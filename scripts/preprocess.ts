import { mkdirSync, writeFileSync, readFileSync, copyFileSync } from "fs";

const DIMS         = 14;
const NUM_CLUSTERS = parseInt(process.env.NUM_CLUSTERS ?? "4000", 10);
const KMEANS_ITERS = parseInt(process.env.KMEANS_ITERS ?? "30",   10);
const SAMPLE_FRAC  = parseFloat(process.env.SAMPLE_FRAC  ?? "0.15"); // 15% ≈ 450K
const MAGIC        = 0x52494e48; // "RINH"

const REFS_PATH  = process.env.REFS_PATH  ?? "./resources/references.json.gz";
const NORM_PATH  = process.env.NORM_PATH  ?? "./resources/normalization.json";
const MCC_PATH   = process.env.MCC_PATH   ?? "./resources/mcc_risk.json";
const OUTPUT_DIR = process.env.DATA_DIR   ?? "./data";
const OUTPUT_BIN = `${OUTPUT_DIR}/index.bin`;

function progress(label: string, i: number, total: number): void {
  process.stdout.write(`\r${label} ${i.toLocaleString()}/${total.toLocaleString()}…`);
}

function tick(label: string): () => void {
  const t0 = performance.now();
  return () =>
    console.log(`${label} — ${((performance.now() - t0) / 1000).toFixed(2)}s`);
}

// Step 1 – Load & parse references

console.log("\n=== Rinha 2026 Preprocessor ===\n");

let done = tick("[1/6] Load references.json.gz");
const compressed = readFileSync(REFS_PATH);
console.log(`     compressed: ${(compressed.byteLength / 1024 / 1024).toFixed(1)} MB`);
const decompressed = Bun.gunzipSync(compressed);
console.log(`     decompressed: ${(decompressed.byteLength / 1024 / 1024).toFixed(1)} MB`);
const refs = JSON.parse(
  new TextDecoder().decode(decompressed)
) as Array<{ vector: number[]; label: string }>;
const N = refs.length;
console.log(`     records: ${N.toLocaleString()}`);
done();

// Step 2 – Extract into typed arrays

done = tick("[2/6] Extract typed arrays");
const allVectors = new Float32Array(N * DIMS);
const allLabels  = new Uint8Array(N);

for (let i = 0; i < N; i++) {
  const ref  = refs[i];
  const base = i * DIMS;
  for (let d = 0; d < DIMS; d++) allVectors[base + d] = ref.vector[d];
  allLabels[i] = ref.label === "fraud" ? 1 : 0;
  if (i % 1_000_000 === 0) progress("  vectors", i, N);
}
console.log();
// Hint GC to free the huge JSON array.
(refs as unknown as null);
done();

// Step 3 – K-Means on a random sample

done = tick("[3/6] K-Means clustering");

const sampleSize = Math.min(Math.floor(N * SAMPLE_FRAC), N);
console.log(`     k=${NUM_CLUSTERS}  sample=${sampleSize.toLocaleString()}  iters=${KMEANS_ITERS}`);

// Reservoir sampling for the K-Means training set.
const sampleIdx = new Uint32Array(sampleSize);
for (let i = 0; i < sampleSize; i++) sampleIdx[i] = i;
for (let i = sampleSize; i < N; i++) {
  const j = Math.floor(Math.random() * (i + 1));
  if (j < sampleSize) sampleIdx[j] = i;
}

const sampleVectors = new Float32Array(sampleSize * DIMS);
for (let i = 0; i < sampleSize; i++) {
  const src = sampleIdx[i] * DIMS;
  sampleVectors.set(allVectors.subarray(src, src + DIMS), i * DIMS);
}

// K-Means++ initialisation: spread centroids optimally.
// First centroid = random. Each subsequent chosen with probability ∝ dist² to nearest centroid.
const centroids = new Float32Array(NUM_CLUSTERS * DIMS);
{
  // Pick first centroid randomly.
  const first = Math.floor(Math.random() * sampleSize);
  centroids.set(sampleVectors.subarray(first * DIMS, first * DIMS + DIMS), 0);

  // minDist[i] = distance² from sample i to its nearest chosen centroid.
  const minDist = new Float64Array(sampleSize);
  minDist.fill(Infinity);

  for (let c = 1; c <= NUM_CLUSTERS; c++) {
    // Update minDist with the centroid just added (c-1).
    const prevBase = (c - 1) * DIMS;
    let totalWeight = 0;
    for (let i = 0; i < sampleSize; i++) {
      const iBase = i * DIMS;
      let dist = 0;
      for (let d = 0; d < DIMS; d++) {
        const diff = sampleVectors[iBase + d] - centroids[prevBase + d];
        dist += diff * diff;
      }
      if (dist < minDist[i]) minDist[i] = dist;
      totalWeight += minDist[i];
    }

    if (c === NUM_CLUSTERS) break; // all centroids placed

    // Weighted random selection (roulette wheel).
    let r = Math.random() * totalWeight;
    let pick = 0;
    for (let i = 0; i < sampleSize; i++) {
      r -= minDist[i];
      if (r <= 0) { pick = i; break; }
    }
    centroids.set(sampleVectors.subarray(pick * DIMS, pick * DIMS + DIMS), c * DIMS);

    if (c % 500 === 0) progress("  K-Means++ init", c, NUM_CLUSTERS);
  }
  console.log();
}

const sampleAssign   = new Uint32Array(sampleSize);
const centroidSumsF64 = new Float64Array(NUM_CLUSTERS * DIMS);
const centroidCounts  = new Uint32Array(NUM_CLUSTERS);

for (let iter = 0; iter < KMEANS_ITERS; iter++) {
  process.stdout.write(`\r  iteration ${iter + 1}/${KMEANS_ITERS}…`);

  // Assignment step.
  for (let i = 0; i < sampleSize; i++) {
    const iBase = i * DIMS;
    let minDist = Infinity;
    let minC    = 0;
    for (let c = 0; c < NUM_CLUSTERS; c++) {
      const cBase = c * DIMS;
      let dist = 0;
      for (let d = 0; d < DIMS; d++) {
        const diff = sampleVectors[iBase + d] - centroids[cBase + d];
        dist += diff * diff;
      }
      if (dist < minDist) { minDist = dist; minC = c; }
    }
    sampleAssign[i] = minC;
  }

  // Update step.
  centroidSumsF64.fill(0);
  centroidCounts.fill(0);
  for (let i = 0; i < sampleSize; i++) {
    const c     = sampleAssign[i];
    const iBase = i * DIMS;
    const cBase = c * DIMS;
    for (let d = 0; d < DIMS; d++) centroidSumsF64[cBase + d] += sampleVectors[iBase + d];
    centroidCounts[c]++;
  }
  for (let c = 0; c < NUM_CLUSTERS; c++) {
    const cnt   = centroidCounts[c];
    const cBase = c * DIMS;
    if (cnt > 0) {
      for (let d = 0; d < DIMS; d++) centroids[cBase + d] = centroidSumsF64[cBase + d] / cnt;
    }
  }
}
console.log();
done();

// Step 4 – Assign ALL 3M vectors to nearest centroid

done = tick("[4/6] Assign all vectors");
const assignments   = new Uint32Array(N);
const clusterSizes  = new Uint32Array(NUM_CLUSTERS);

for (let i = 0; i < N; i++) {
  const iBase = i * DIMS;
  let minDist = Infinity;
  let minC    = 0;
  for (let c = 0; c < NUM_CLUSTERS; c++) {
    const cBase = c * DIMS;
    let dist = 0;
    for (let d = 0; d < DIMS; d++) {
      const diff = allVectors[iBase + d] - centroids[cBase + d];
      dist += diff * diff;
    }
    if (dist < minDist) { minDist = dist; minC = c; }
  }
  assignments[i] = minC;
  clusterSizes[minC]++;
  if (i % 500_000 === 0) progress("  assigned", i, N);
}
console.log();
done();

// Step 5 – Compute offsets and build sorted index

done = tick("[5/6] Sort by cluster");
const clusterOffsets = new Uint32Array(NUM_CLUSTERS);
let runningOffset = 0;
for (let c = 0; c < NUM_CLUSTERS; c++) {
  clusterOffsets[c] = runningOffset;
  runningOffset += clusterSizes[c];
}

// Build sorted order (counting sort by cluster).
const sortedIndices = new Uint32Array(N);
const tempCount     = new Uint32Array(NUM_CLUSTERS);
for (let i = 0; i < N; i++) {
  const c   = assignments[i];
  const pos = clusterOffsets[c] + tempCount[c];
  sortedIndices[pos] = i;
  tempCount[c]++;
}

// Compute per-cluster fraud rates (used by Tier 1 fast-path in search).
const centroidFraudRates = new Float32Array(NUM_CLUSTERS);
for (let c = 0; c < NUM_CLUSTERS; c++) {
  const offset = clusterOffsets[c];
  const size   = clusterSizes[c];
  if (size === 0) { centroidFraudRates[c] = 0.5; continue; }
  let fraudCount = 0;
  for (let i = 0; i < size; i++) {
    if (allLabels[sortedIndices[offset + i]] === 1) fraudCount++;
  }
  centroidFraudRates[c] = fraudCount / size;
}
done();

// Step 5b – Build mini-HNSW over boundary clusters (disabled — reserved for future use)
{
  const HNSW_ENABLED = process.env.HNSW_ENABLED === "true";
  if (!HNSW_ENABLED) {
    console.log("\n[5b] Mini-HNSW: disabled (set HNSW_ENABLED=true to build)");
  } else {
  const HM            = 8;
  const HM0           = HM * 2;
  const HEF           = 100;
  const BOUNDARY_LO   = parseFloat(process.env.HNSW_BOUNDARY_LO  ?? "0.1");
  const BOUNDARY_HI   = parseFloat(process.env.HNSW_BOUNDARY_HI  ?? "0.9");
  const MAX_VECTORS   = parseInt (process.env.HNSW_MAX_VECTORS   ?? "700000", 10);
  const HMAG          = 0x48534e57; // "HNSW"
  const HSEN          = 0xFFFFFFFF;
  const HML           = 1.0 / Math.log(HM);

  // Collect vectors from clusters with mixed fraud rates.
  const origIdx: number[] = [];
  const lblArr:  number[] = [];
  for (let c = 0; c < NUM_CLUSTERS; c++) {
    const rate = centroidFraudRates[c];
    if (rate >= BOUNDARY_LO && rate <= BOUNDARY_HI) {
      const off = clusterOffsets[c], sz = clusterSizes[c];
      for (let i = 0; i < sz; i++) {
        const idx = sortedIndices[off + i];
        origIdx.push(idx);
        lblArr.push(allLabels[idx]);
      }
    }
  }

  const HN = origIdx.length;
  console.log(`\n[5b] Mini-HNSW: ${HN.toLocaleString()} boundary vectors (rate ${BOUNDARY_LO}–${BOUNDARY_HI}), M=${HM}, efC=${HEF}`);

  if (HN === 0) {
    console.log(`     skipped — no boundary vectors found`);
  } else if (HN > MAX_VECTORS) {
    console.log(`     skipped — ${HN.toLocaleString()} > max ${MAX_VECTORS.toLocaleString()}`);
    console.log(`     Hint: tighten HNSW_BOUNDARY_LO/HI or raise HNSW_MAX_VECTORS`);
  } else {
    const t0 = performance.now();

    // Per-node adjacency (build-time only; JS GC handles cleanup after serialise).
    const g0: number[][] = Array.from({ length: HN }, () => []);
    const gu: number[][][] = Array.from({ length: HN }, () => []);
    const nl = new Uint8Array(HN);

    // Epoch-based visited for build-time searches.
    const vis    = new Uint32Array(HN);
    let   visGen = 0;

    let hEP = 0, hMaxLv = 0, hIns = 0;

    const hdist = (a: number, b: number): number => {
      const ab = origIdx[a] * DIMS, bb = origIdx[b] * DIMS;
      let d = 0;
      for (let i = 0; i < DIMS; i++) { const diff = allVectors[ab+i] - allVectors[bb+i]; d += diff*diff; }
      return d;
    };

    const getConns = (node: number, lv: number): number[] => {
      if (lv === 0) return g0[node];
      if (lv > nl[node]) return []; // safety: node not at this level
      return gu[node][lv-1] ?? (gu[node][lv-1] = []);
    };

    // Min-heap helpers (build-time inline closures).
    const mPush = (h: [number,number][], d: number, id: number) => {
      h.push([d, id]);
      let i = h.length - 1;
      while (i > 0) {
        const p = (i-1) >> 1;
        if (h[p][0] <= d) break;
        [h[p], h[i]] = [h[i], h[p]]; i = p;
      }
    };
    const mPop = (h: [number,number][]): [number,number] => {
      const top = h[0], last = h.pop()!;
      if (h.length) {
        h[0] = last; let i = 0;
        for (;;) {
          let s = i, l = 2*i+1, r = 2*i+2;
          if (l < h.length && h[l][0] < h[s][0]) s = l;
          if (r < h.length && h[r][0] < h[s][0]) s = r;
          if (s === i) break;
          [h[i], h[s]] = [h[s], h[i]]; i = s;
        }
      }
      return top;
    };
    // Bounded max-heap push (evicts farthest when full).
    const xPush = (h: [number,number][], ef: number, d: number, id: number) => {
      if (h.length < ef) {
        h.push([d, id]); let i = h.length - 1;
        while (i > 0) {
          const p = (i-1) >> 1;
          if (h[p][0] >= d) break;
          [h[p], h[i]] = [h[i], h[p]]; i = p;
        }
      } else if (d < h[0][0]) {
        h[0] = [d, id]; let i = 0;
        for (;;) {
          let s = i, l = 2*i+1, r = 2*i+2;
          if (l < h.length && h[l][0] > h[s][0]) s = l;
          if (r < h.length && h[r][0] > h[s][0]) s = r;
          if (s === i) break;
          [h[i], h[s]] = [h[s], h[i]]; i = s;
        }
      }
    };

    const searchL = (q: number, ep_: number, ef: number, lv: number): [number,number][] => {
      visGen++;
      vis[ep_] = visGen;
      const cands: [number,number][] = [], res: [number,number][] = [];
      const d0 = hdist(q, ep_);
      mPush(cands, d0, ep_); xPush(res, ef, d0, ep_);
      while (cands.length) {
        const [cD, c] = mPop(cands);
        if (res.length >= ef && cD > res[0][0]) break;
        for (const nb of getConns(c, lv)) {
          if (vis[nb] !== visGen) {
            vis[nb] = visGen;
            const d = hdist(q, nb);
            if (res.length < ef || d < res[0][0]) { mPush(cands, d, nb); xPush(res, ef, d, nb); }
          }
        }
      }
      return res;
    };

    const insert = (q: number) => {
      if (!hIns) { nl[q] = 0; hEP = q; hMaxLv = 0; hIns++; return; }
      const lv = Math.floor(-Math.log(Math.random()) * HML);
      nl[q] = lv;
      let cur = hEP;

      // Phase 1: greedy descent from hMaxLv to lv+1.
      for (let l = hMaxLv; l > lv; l--) {
        const W = searchL(q, cur, 1, l);
        if (W.length) { let b = W[0]; for (const w of W) if (w[0] < b[0]) b = w; cur = b[1]; }
      }

      // Phase 2: connect at each level from min(lv, hMaxLv) down to 0.
      for (let l = Math.min(lv, hMaxLv); l >= 0; l--) {
        const mC = l === 0 ? HM0 : HM;
        const W = searchL(q, cur, HEF, l);
        W.sort((a, b) => a[0] - b[0]);

        // Add edges from q to selected neighbours.
        const qC = getConns(q, l);
        for (const [, nb] of W.slice(0, mC)) if (!qC.includes(nb)) qC.push(nb);

        // Add back-edges from neighbours to q; prune if over capacity.
        for (const [, nb] of W.slice(0, mC)) {
          const nC = getConns(nb, l);
          if (!nC.includes(q)) {
            nC.push(q);
            if (nC.length > mC) {
              const sorted = nC
                .map(id => [hdist(nb, id), id] as [number,number])
                .sort((a, b) => a[0] - b[0]);
              nC.length = 0;
              for (let j = 0; j < mC; j++) nC.push(sorted[j][1]);
            }
          }
        }

        if (W.length) { let b = W[0]; for (const w of W) if (w[0] < b[0]) b = w; cur = b[1]; }
      }

      if (lv > hMaxLv) { hMaxLv = lv; hEP = q; }
      hIns++;
    };

    for (let i = 0; i < HN; i++) {
      insert(i);
      if (i % 10_000 === 0) progress("  inserting", i, HN);
    }
    console.log(`\n     built in ${((performance.now() - t0) / 1000).toFixed(1)}s, maxLevel=${hMaxLv}, ep=${hEP}`);

    // ── Serialise ────────────────────────────────────────────────────────────

    let totalUpperEdges = 0;
    const upperOffsetArr = new Uint32Array(HN).fill(HSEN);
    for (let i = 0; i < HN; i++) {
      if (nl[i] > 0) { upperOffsetArr[i] = totalUpperEdges; totalUpperEdges += nl[i] * HM; }
    }

    const hdrSz  = 28;
    const nlSz   = (HN + 3) & ~3;          // 4-byte padded nodeLevel
    const g0Sz   = HN * HM0 * 4;
    const uoSz   = HN * 4;
    const ueSz   = totalUpperEdges * 4;
    const vecOff = ((hdrSz + nlSz + g0Sz + uoSz + ueSz) + 1) & ~1; // 2-byte align
    const vecSz  = HN * DIMS * 2;
    const lblOff = vecOff + vecSz;
    const total  = lblOff + HN;

    console.log(`     hnsw.bin: ${(total / 1024 / 1024).toFixed(1)} MB`);

    const buf = new ArrayBuffer(total);

    // Header
    const dv = new DataView(buf, 0, hdrSz);
    dv.setUint32( 0, HMAG,            true);
    dv.setUint32( 4, HN,              true);
    dv.setUint32( 8, HM,              true);
    dv.setUint32(12, HM0,             true);
    dv.setUint32(16, hMaxLv,          true);
    dv.setUint32(20, hEP,             true);
    dv.setUint32(24, totalUpperEdges, true);

    // nodeLevel
    new Uint8Array(buf, hdrSz, HN).set(nl);

    // graph0
    const g0Out = new Uint32Array(buf, hdrSz + nlSz, HN * HM0);
    g0Out.fill(HSEN);
    for (let i = 0; i < HN; i++) {
      const conns = g0[i], base = i * HM0;
      for (let j = 0; j < Math.min(conns.length, HM0); j++) g0Out[base + j] = conns[j];
    }

    // upperOffset
    new Uint32Array(buf, hdrSz + nlSz + g0Sz, HN).set(upperOffsetArr);

    // upperEdges
    const ueOut = new Uint32Array(buf, hdrSz + nlSz + g0Sz + uoSz, totalUpperEdges);
    ueOut.fill(HSEN);
    for (let i = 0; i < HN; i++) {
      if (!nl[i]) continue;
      const base = upperOffsetArr[i];
      for (let l = 1; l <= nl[i]; l++) {
        const conns = gu[i][l-1] ?? [], off = base + (l-1) * HM;
        for (let j = 0; j < Math.min(conns.length, HM); j++) ueOut[off + j] = conns[j];
      }
    }

    // Uint16-quantised vectors (same scheme as IVF: v<0→65535, [0,1]→[0,65534])
    const vOut = new Uint16Array(buf, vecOff, HN * DIMS);
    for (let i = 0; i < HN; i++) {
      const srcBase = origIdx[i] * DIMS, dstBase = i * DIMS;
      for (let d = 0; d < DIMS; d++) {
        const v = allVectors[srcBase + d];
        vOut[dstBase + d] = v < 0 ? 65535 : Math.round(v * 65534);
      }
    }

    // Labels
    const lOut = new Uint8Array(buf, lblOff, HN);
    for (let i = 0; i < HN; i++) lOut[i] = lblArr[i];

    mkdirSync(OUTPUT_DIR, { recursive: true });
    writeFileSync(`${OUTPUT_DIR}/hnsw.bin`, new Uint8Array(buf));
    console.log(`     written → ${OUTPUT_DIR}/hnsw.bin`);
    } // end large HN else block
  } // end HNSW_ENABLED else block
} // end step 5b

// Step 6 – Quantise and write binary

done = tick("[6/6] Write index.bin");

const centroidsOff = 16;
const centroidsBytes = NUM_CLUSTERS * DIMS * 4;
const sizesOff       = centroidsOff + centroidsBytes;
const sizesBytes     = NUM_CLUSTERS * 4;
const offsetsOff     = sizesOff + sizesBytes;
const offsetsBytes   = NUM_CLUSTERS * 4;
const ratesOff       = offsetsOff + offsetsBytes;   // centroidFraudRates
const ratesBytes     = NUM_CLUSTERS * 4;
const vectorsOff     = Math.ceil((ratesOff + ratesBytes) / 2) * 2; // 2-byte align
const vectorsBytes   = N * DIMS * 2;
const labelsOff      = vectorsOff + vectorsBytes;
const totalBytes     = labelsOff + N;

console.log(`     total binary size: ${(totalBytes / 1024 / 1024).toFixed(1)} MB`);

const outBuf = new ArrayBuffer(totalBytes);

// Header
const hdrView = new DataView(outBuf, 0, 16);
hdrView.setUint32(0,  MAGIC,        true);
hdrView.setUint32(4,  N,            true);
hdrView.setUint32(8,  NUM_CLUSTERS, true);
hdrView.setUint32(12, DIMS,         true);

// Centroids (Float32, already [0,1])
new Float32Array(outBuf, centroidsOff, NUM_CLUSTERS * DIMS).set(centroids);

// Cluster sizes, offsets, and per-cluster fraud rates
new Uint32Array (outBuf, sizesOff,   NUM_CLUSTERS).set(clusterSizes);
new Uint32Array (outBuf, offsetsOff, NUM_CLUSTERS).set(clusterOffsets);
new Float32Array(outBuf, ratesOff,   NUM_CLUSTERS).set(centroidFraudRates);

// Quantised vectors and labels, sorted by cluster.
const vecOut = new Uint16Array(outBuf, vectorsOff, N * DIMS);
const lblOut = new Uint8Array(outBuf, labelsOff,   N);

for (let pos = 0; pos < N; pos++) {
  const orig    = sortedIndices[pos];
  const srcBase = orig * DIMS;
  const dstBase = pos  * DIMS;
  for (let d = 0; d < DIMS; d++) {
    const v = allVectors[srcBase + d];
    // Sentinel -1 → 65535; [0,1] → [0, 65534]
    vecOut[dstBase + d] = v < 0 ? 65535 : Math.round(v * 65534);
  }
  lblOut[pos] = allLabels[orig];
  if (pos % 500_000 === 0) progress("  written", pos, N);
}
console.log();

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(OUTPUT_BIN, new Uint8Array(outBuf));
done();

// Also copy small resource files.
copyFileSync(NORM_PATH, `${OUTPUT_DIR}/normalization.json`);
copyFileSync(MCC_PATH,  `${OUTPUT_DIR}/mcc_risk.json`);

console.log(`\nDone → ${OUTPUT_BIN}\n`);
