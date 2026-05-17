package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

const (
	DIMS  = 14
	K     = 5
	MAGIC = 0x52494e48 // "RINH"
)

// Parsed from env
var (
	sockPath  string
	dataDir   string
	nprobeBase int
	nprobeMax  int
	nprobeMin  int
)

// Index data (loaded once at startup, read-only thereafter)
var (
	numVectors         int
	numClusters        int
	centroids          []float32 // [numClusters * DIMS]
	clusterSizes       []uint32  // [numClusters]
	clusterOffsets     []uint32  // [numClusters]
	centroidFraudRates []float32 // [numClusters]
	vectors            []uint16  // [numVectors * DIMS]
	labels             []uint8   // [numVectors]
)

// Vectorizer constants (loaded once at startup)
var (
	normMaxAmount           float64
	normMaxInstallments     float64
	normAmountVsAvgRatio    float64
	normMaxMinutes          float64
	normMaxKm              float64
	normMaxTxCount24h       float64
	normMaxMerchantAvgAmt   float64
	mccRisk                 map[string]float64
)

// Per-goroutine scratch to avoid allocation on each request
type searchScratch struct {
	centDists   []float32
	topClusters []int
}

var scratchPool = sync.Pool{
	New: func() any {
		return &searchScratch{
			centDists:   make([]float32, 4096),
			topClusters: make([]int, 12),
		}
	},
}

func getScratch() *searchScratch {
	s := scratchPool.Get().(*searchScratch)
	if len(s.centDists) < numClusters {
		s.centDists = make([]float32, numClusters)
	}
	if len(s.topClusters) < nprobeMax {
		s.topClusters = make([]int, nprobeMax)
	}
	return s
}

func putScratch(s *searchScratch) {
	scratchPool.Put(s)
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envIntOrDefault(key string, def int) int {
	s := os.Getenv(key)
	if s == "" {
		return def
	}
	v, err := strconv.Atoi(s)
	if err != nil {
		return def
	}
	return v
}

func main() {
	sockPath = envOrDefault("SOCK_PATH", "/var/run/api/api.sock")
	dataDir = envOrDefault("DATA_DIR", "/data")
	nprobeBase = envIntOrDefault("NPROBE", 7)
	nprobeMax = envIntOrDefault("NPROBE_MAX", 12)
	nprobeMin = envIntOrDefault("NPROBE_MIN", 5)

	fmt.Println("[startup] Loading normalization constants and MCC risk table...")
	loadNormalization()

	fmt.Println("[startup] Loading index binary...")
	loadIndex()
	fmt.Printf("[startup] Index: %d vectors, %d clusters, nprobe=%d-%d-%d (adaptive)\n",
		numVectors, numClusters, nprobeMin, nprobeBase, nprobeMax)

	// Remove stale socket
	os.Remove(sockPath)

	ln, err := net.Listen("unix", sockPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[fatal] listen: %v\n", err)
		os.Exit(1)
	}
	// chmod 777 so nginx can connect
	os.Chmod(sockPath, 0o777)

	mux := http.NewServeMux()
	mux.HandleFunc("/ready", handleReady)
	mux.HandleFunc("/", handleRoot)

	server := &http.Server{
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
	}

	fmt.Printf("[startup] Serving on %s\n", sockPath)
	if err := server.Serve(ln); err != nil {
		fmt.Fprintf(os.Stderr, "[fatal] serve: %v\n", err)
		os.Exit(1)
	}
}

func handleReady(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(200)
	w.Write([]byte("OK"))
}

var fallbackBody = []byte(`{"approved":true,"fraud_score":0.0}`)

func handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.NotFound(w, r)
		return
	}

	var payload Payload
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(&payload); err != nil {
		// Never return 5xx — fallback to approved
		w.Header().Set("Content-Type", "application/json")
		w.Write(fallbackBody)
		return
	}

	var query [DIMS]float32
	vectorize(&payload, &query)
	fraudCount := searchIVF(&query)
	fraudScore := float64(fraudCount) / float64(K)
	approved := fraudScore < 0.6

	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"approved":%t,"fraud_score":%.1f}`, approved, fraudScore)
}

// ── Index Loading ──────────────────────────────────────────────────────────────

func loadIndex() {
	data, err := os.ReadFile(dataDir + "/index.bin")
	if err != nil {
		fmt.Fprintf(os.Stderr, "[fatal] read index: %v\n", err)
		os.Exit(1)
	}

	magic := binary.LittleEndian.Uint32(data[0:4])
	if magic != MAGIC {
		fmt.Fprintf(os.Stderr, "[fatal] index.bin: bad magic %x\n", magic)
		os.Exit(1)
	}

	numVectors = int(binary.LittleEndian.Uint32(data[4:8]))
	numClusters = int(binary.LittleEndian.Uint32(data[8:12]))
	// dims at data[12:16] should be 14

	centOffset := 16
	centBytes := numClusters * DIMS * 4
	sizesOffset := centOffset + centBytes
	offsOffset := sizesOffset + numClusters*4
	ratesOffset := offsOffset + numClusters*4
	vecOffset := ((ratesOffset + numClusters*4) + 1) &^ 1 // 2-byte align
	lblOffset := vecOffset + numVectors*DIMS*2

	// Parse centroids
	centroids = make([]float32, numClusters*DIMS)
	for i := range centroids {
		centroids[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[centOffset+i*4:]))
	}

	// Parse cluster sizes
	clusterSizes = make([]uint32, numClusters)
	for i := range clusterSizes {
		clusterSizes[i] = binary.LittleEndian.Uint32(data[sizesOffset+i*4:])
	}

	// Parse cluster offsets
	clusterOffsets = make([]uint32, numClusters)
	for i := range clusterOffsets {
		clusterOffsets[i] = binary.LittleEndian.Uint32(data[offsOffset+i*4:])
	}

	// Parse centroid fraud rates
	centroidFraudRates = make([]float32, numClusters)
	for i := range centroidFraudRates {
		centroidFraudRates[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[ratesOffset+i*4:]))
	}

	// Parse vectors (Uint16)
	vectors = make([]uint16, numVectors*DIMS)
	for i := range vectors {
		vectors[i] = binary.LittleEndian.Uint16(data[vecOffset+i*2:])
	}

	// Parse labels
	labels = make([]uint8, numVectors)
	copy(labels, data[lblOffset:lblOffset+numVectors])
}

func loadNormalization() {
	// normalization.json
	normData, err := os.ReadFile(dataDir + "/normalization.json")
	if err != nil {
		fmt.Fprintf(os.Stderr, "[fatal] read normalization: %v\n", err)
		os.Exit(1)
	}
	var norm struct {
		MaxAmount          float64 `json:"max_amount"`
		MaxInstallments    float64 `json:"max_installments"`
		AmountVsAvgRatio   float64 `json:"amount_vs_avg_ratio"`
		MaxMinutes         float64 `json:"max_minutes"`
		MaxKm              float64 `json:"max_km"`
		MaxTxCount24h      float64 `json:"max_tx_count_24h"`
		MaxMerchantAvgAmt  float64 `json:"max_merchant_avg_amount"`
	}
	if err := json.Unmarshal(normData, &norm); err != nil {
		fmt.Fprintf(os.Stderr, "[fatal] parse normalization: %v\n", err)
		os.Exit(1)
	}
	normMaxAmount = norm.MaxAmount
	normMaxInstallments = norm.MaxInstallments
	normAmountVsAvgRatio = norm.AmountVsAvgRatio
	normMaxMinutes = norm.MaxMinutes
	normMaxKm = norm.MaxKm
	normMaxTxCount24h = norm.MaxTxCount24h
	normMaxMerchantAvgAmt = norm.MaxMerchantAvgAmt

	// mcc_risk.json
	mccData, err := os.ReadFile(dataDir + "/mcc_risk.json")
	if err != nil {
		fmt.Fprintf(os.Stderr, "[fatal] read mcc_risk: %v\n", err)
		os.Exit(1)
	}
	mccRisk = make(map[string]float64)
	if err := json.Unmarshal(mccData, &mccRisk); err != nil {
		fmt.Fprintf(os.Stderr, "[fatal] parse mcc_risk: %v\n", err)
		os.Exit(1)
	}
}

// ── IVF Search ─────────────────────────────────────────────────────────────────

func searchIVF(query *[DIMS]float32) int {
	sc := getScratch()
	defer putScratch(sc)
	centDists := sc.centDists[:numClusters]
	topClusters := sc.topClusters[:nprobeMax]

	// Scale query to Uint16 space
	var scaledQuery [DIMS]float32
	for d := 0; d < DIMS; d++ {
		if query[d] < 0 {
			scaledQuery[d] = 65535
		} else {
			scaledQuery[d] = query[d] * 65534
		}
	}

	hasSentinel := query[5] < 0

	// Step 1: distance from query to every centroid (float space)
	for c := 0; c < numClusters; c++ {
		base := c * DIMS
		var dist float32
		for d := 0; d < DIMS; d++ {
			diff := query[d] - centroids[base+d]
			dist += diff * diff
		}
		centDists[c] = dist
	}

	// Select top NPROBE_MAX clusters
	for i := 0; i < nprobeMax; i++ {
		minDist := float32(math.MaxFloat32)
		minIdx := 0
		for c := 0; c < numClusters; c++ {
			if centDists[c] < minDist {
				minDist = centDists[c]
				minIdx = c
			}
		}
		topClusters[i] = minIdx
		centDists[minIdx] = float32(math.MaxFloat32)
	}

	// Adaptive NPROBE via distance ratio
	var nearDist, farDist float32
	{
		nearC := topClusters[0]
		nearBase := nearC * DIMS
		for d := 0; d < DIMS; d++ {
			diff := query[d] - centroids[nearBase+d]
			nearDist += diff * diff
		}
		farC := topClusters[nprobeBase]
		farBase := farC * DIMS
		for d := 0; d < DIMS; d++ {
			diff := query[d] - centroids[farBase+d]
			farDist += diff * diff
		}
	}

	var nprobe int
	ratio := float32(100.0)
	if nearDist > 0 {
		ratio = farDist / nearDist
	}
	if ratio < 2.5 {
		nprobe = nprobeMax
	} else if ratio > 4.0 {
		nprobe = nprobeMin
	} else {
		nprobe = nprobeBase
	}

	// Tier 1: pure-legit isolated cluster fast-path
	if ratio > 10.0 && centroidFraudRates[topClusters[0]] == 0.0 {
		return 0
	}

	// Step 2: linear scan through probed clusters
	var topDist [K]float64
	var topLbl [K]uint8
	for j := 0; j < K; j++ {
		topDist[j] = math.MaxFloat64
	}
	worstDist := math.MaxFloat64
	worstIdx := 0

	if hasSentinel {
		// Sentinel path: dims 5,6 corrected
		for ci := 0; ci < nprobe; ci++ {
			c := topClusters[ci]
			offset := int(clusterOffsets[c])
			size := int(clusterSizes[c])
			vecBase := offset * DIMS

			for i := 0; i < size; i++ {
				base := vecBase + i*DIMS
				var dist float64
				for d := 0; d < 5; d++ {
					diff := float64(scaledQuery[d]) - float64(vectors[base+d])
					dist += diff * diff
				}
				// Dim 5: sentinel query → (65534 + refVal)²
				rv5 := vectors[base+5]
				if rv5 != 65535 {
					cd5 := float64(65534 + int(rv5))
					dist += cd5 * cd5
				}
				// Dim 6: sentinel query → (65534 + refVal)²
				rv6 := vectors[base+6]
				if rv6 != 65535 {
					cd6 := float64(65534 + int(rv6))
					dist += cd6 * cd6
				}
				for d := 7; d < DIMS; d++ {
					diff := float64(scaledQuery[d]) - float64(vectors[base+d])
					dist += diff * diff
				}
				if dist < worstDist {
					topDist[worstIdx] = dist
					topLbl[worstIdx] = labels[offset+i]
					worstDist = 0
					for j := 0; j < K; j++ {
						if topDist[j] > worstDist {
							worstDist = topDist[j]
							worstIdx = j
						}
					}
				}
			}
		}
	} else {
		// Normal path: check for sentinel refs
		sq5 := float64(scaledQuery[5])
		sq6 := float64(scaledQuery[6])
		sentDist5 := (65534 + sq5) * (65534 + sq5)
		sentDist6 := (65534 + sq6) * (65534 + sq6)

		for ci := 0; ci < nprobe; ci++ {
			c := topClusters[ci]
			offset := int(clusterOffsets[c])
			size := int(clusterSizes[c])
			vecBase := offset * DIMS

			for i := 0; i < size; i++ {
				base := vecBase + i*DIMS
				var dist float64
				for d := 0; d < 5; d++ {
					diff := float64(scaledQuery[d]) - float64(vectors[base+d])
					dist += diff * diff
				}
				// Dim 5
				if vectors[base+5] == 65535 {
					dist += sentDist5
				} else {
					diff := sq5 - float64(vectors[base+5])
					dist += diff * diff
				}
				// Dim 6
				if vectors[base+6] == 65535 {
					dist += sentDist6
				} else {
					diff := sq6 - float64(vectors[base+6])
					dist += diff * diff
				}
				for d := 7; d < DIMS; d++ {
					diff := float64(scaledQuery[d]) - float64(vectors[base+d])
					dist += diff * diff
				}
				if dist < worstDist {
					topDist[worstIdx] = dist
					topLbl[worstIdx] = labels[offset+i]
					worstDist = 0
					for j := 0; j < K; j++ {
						if topDist[j] > worstDist {
							worstDist = topDist[j]
							worstIdx = j
						}
					}
				}
			}
		}
	}

	fraudCount := 0
	for j := 0; j < K; j++ {
		if topDist[j] < math.MaxFloat64 && topLbl[j] == 1 {
			fraudCount++
		}
	}
	return fraudCount
}
