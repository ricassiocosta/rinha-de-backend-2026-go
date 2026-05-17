package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net"
	"os"
	"runtime"
	"strconv"
	"sync"

	"github.com/tidwall/gjson"
	"github.com/valyala/fasthttp"
)

const (
	DIMS  = 14
	K     = 5
	MAGIC = 0x52494e48 // "RINH"
)

var (
	sockPath   string
	dataDir    string
	nprobeBase int
	nprobeMax  int
	nprobeMin  int
)

// Index data (loaded once, read-only)
var (
	numVectors         int
	numClusters        int
	centroids          []float32
	clusterSizes       []uint32
	clusterOffsets     []uint32
	centroidFraudRates []float32
	vectors            []uint16
	labels             []uint8
)

// Vectorizer constants
var (
	normMaxAmount         float64
	normMaxInstallments   float64
	normAmountVsAvgRatio  float64
	normMaxMinutes        float64
	normMaxKm             float64
	normMaxTxCount24h     float64
	normMaxMerchantAvgAmt float64
	mccRisk               map[string]float64
)

// Pre-computed response bytes
var (
	respApproved00 = []byte(`{"approved":true,"fraud_score":0.0}`)
	respApproved02 = []byte(`{"approved":true,"fraud_score":0.2}`)
	respApproved04 = []byte(`{"approved":true,"fraud_score":0.4}`)
	respDenied06   = []byte(`{"approved":false,"fraud_score":0.6}`)
	respDenied08   = []byte(`{"approved":false,"fraud_score":0.8}`)
	respDenied10   = []byte(`{"approved":false,"fraud_score":1.0}`)
	contentTypeJSON = []byte("application/json")
)

// Per-goroutine scratch buffers
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

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envIntOrDefault(key string, def int) int {
	if s := os.Getenv(key); s != "" {
		if v, err := strconv.Atoi(s); err == nil {
			return v
		}
	}
	return def
}

func main() {
	runtime.GOMAXPROCS(1)

	sockPath = envOrDefault("SOCK_PATH", "/var/run/api/api.sock")
	dataDir = envOrDefault("DATA_DIR", "/data")
	nprobeBase = envIntOrDefault("NPROBE", 7)
	nprobeMax = envIntOrDefault("NPROBE_MAX", 12)
	nprobeMin = envIntOrDefault("NPROBE_MIN", 5)

	fmt.Println("[startup] Loading normalization constants and MCC risk table...")
	loadNormalization()

	fmt.Println("[startup] Loading index binary...")
	loadIndex()
	fmt.Printf("[startup] Index: %d vectors, %d clusters, nprobe=%d-%d-%d\n",
		numVectors, numClusters, nprobeMin, nprobeBase, nprobeMax)

	os.Remove(sockPath)
	ln, err := net.Listen("unix", sockPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[fatal] listen: %v\n", err)
		os.Exit(1)
	}
	os.Chmod(sockPath, 0o777)

	server := &fasthttp.Server{
		Handler:               requestHandler,
		DisableKeepalive:      false,
		NoDefaultServerHeader: true,
		NoDefaultContentType:  true,
		NoDefaultDate:         true,
		ReduceMemoryUsage:     false,
	}

	fmt.Printf("[startup] Serving on %s (GOMAXPROCS=%d, GOGC=off)\n", sockPath, runtime.GOMAXPROCS(0))
	if err := server.Serve(ln); err != nil {
		fmt.Fprintf(os.Stderr, "[fatal] serve: %v\n", err)
		os.Exit(1)
	}
}

func requestHandler(ctx *fasthttp.RequestCtx) {
	path := ctx.Path()
	if len(path) == 6 && path[1] == 'r' { // "/ready"
		ctx.SetStatusCode(200)
		ctx.SetBody([]byte("OK"))
		return
	}

	if !ctx.IsPost() {
		ctx.SetStatusCode(404)
		return
	}

	body := ctx.PostBody()
	if len(body) == 0 {
		ctx.SetStatusCode(200)
		ctx.Response.Header.SetContentTypeBytes(contentTypeJSON)
		ctx.SetBody(respApproved00)
		return
	}

	var query [DIMS]float32
	vectorizeGJSON(body, &query)
	fraudCount := searchIVF(&query)

	ctx.Response.Header.SetContentTypeBytes(contentTypeJSON)
	switch fraudCount {
	case 0:
		ctx.SetBody(respApproved00)
	case 1:
		ctx.SetBody(respApproved02)
	case 2:
		ctx.SetBody(respApproved04)
	case 3:
		ctx.SetBody(respDenied06)
	case 4:
		ctx.SetBody(respDenied08)
	default:
		ctx.SetBody(respDenied10)
	}
}

// ── Zero-alloc vectorize using gjson ───────────────────────────────────────────

func vectorizeGJSON(body []byte, out *[DIMS]float32) {
	raw := gjson.ParseBytes(body)

	txAmount := raw.Get("transaction.amount").Float()
	txInstallments := raw.Get("transaction.installments").Float()
	requestedAt := raw.Get("transaction.requested_at").Str

	custAvgAmount := raw.Get("customer.avg_amount").Float()
	custTxCount24h := raw.Get("customer.tx_count_24h").Float()

	merchID := raw.Get("merchant.id").Str
	merchMCC := raw.Get("merchant.mcc").Str
	merchAvgAmount := raw.Get("merchant.avg_amount").Float()

	termKmFromHome := raw.Get("terminal.km_from_home").Float()
	termIsOnline := raw.Get("terminal.is_online").Bool()
	termCardPresent := raw.Get("terminal.card_present").Bool()

	out[0] = clamp01(txAmount / normMaxAmount)
	out[1] = clamp01(txInstallments / normMaxInstallments)

	avgRatio := 0.0
	if custAvgAmount > 0 {
		avgRatio = txAmount / custAvgAmount / normAmountVsAvgRatio
	}
	out[2] = clamp01(avgRatio)

	out[3] = float32(parseHourUTC(requestedAt)) / 23.0
	out[4] = float32(dayOfWeekZeller(requestedAt)) / 6.0

	lastTx := raw.Get("last_transaction")
	if !lastTx.Exists() || lastTx.Type == gjson.Null {
		out[5] = -1
		out[6] = -1
	} else {
		lastTimestamp := lastTx.Get("timestamp").Str
		lastKm := lastTx.Get("km_from_current").Float()
		currentMs := parseEpochMs(requestedAt)
		lastMs := parseEpochMs(lastTimestamp)
		minutesBetween := float64(currentMs-lastMs) / 60000.0
		out[5] = clamp01(minutesBetween / normMaxMinutes)
		out[6] = clamp01(lastKm / normMaxKm)
	}

	out[7] = clamp01(termKmFromHome / normMaxKm)
	out[8] = clamp01(custTxCount24h / normMaxTxCount24h)

	if termIsOnline {
		out[9] = 1
	} else {
		out[9] = 0
	}
	if termCardPresent {
		out[10] = 1
	} else {
		out[10] = 0
	}

	// Unknown merchant check
	out[11] = 1
	knownMerchants := raw.Get("customer.known_merchants")
	if knownMerchants.IsArray() {
		knownMerchants.ForEach(func(_, v gjson.Result) bool {
			if v.Str == merchID {
				out[11] = 0
				return false
			}
			return true
		})
	}

	if risk, ok := mccRisk[merchMCC]; ok {
		out[12] = float32(risk)
	} else {
		out[12] = 0.5
	}

	out[13] = clamp01(merchAvgAmount / normMaxMerchantAvgAmt)
}

func clamp01(v float64) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return float32(v)
}

// parseHourUTC extracts hour from ISO "2026-03-11T18:45:53Z"
func parseHourUTC(iso string) int {
	if len(iso) < 13 {
		return 0
	}
	return int(iso[11]-'0')*10 + int(iso[12]-'0')
}

// dayOfWeekZeller uses Zeller's congruence (no time.Date allocation)
// Returns Mon=0 ... Sun=6
func dayOfWeekZeller(iso string) int {
	if len(iso) < 10 {
		return 0
	}
	y := int(iso[0]-'0')*1000 + int(iso[1]-'0')*100 + int(iso[2]-'0')*10 + int(iso[3]-'0')
	m := int(iso[5]-'0')*10 + int(iso[6]-'0')
	d := int(iso[8]-'0')*10 + int(iso[9]-'0')

	// Adjust for Zeller's: Jan/Feb are months 13/14 of prev year
	if m < 3 {
		m += 12
		y--
	}
	k := y % 100
	j := y / 100
	// Zeller: h = (d + floor(13*(m+1)/5) + k + floor(k/4) + floor(j/4) - 2*j) mod 7
	// h=0→Sat, h=1→Sun, h=2→Mon, ..., h=6→Fri
	h := (d + (13*(m+1))/5 + k + k/4 + j/4 - 2*j) % 7
	if h < 0 {
		h += 7
	}
	// Convert: h=2→Mon=0, h=3→Tue=1, ..., h=6→Fri=4, h=0→Sat=5, h=1→Sun=6
	return (h + 5) % 7
}

// parseEpochMs converts ISO timestamp to Unix milliseconds (no time.Date)
func parseEpochMs(iso string) int64 {
	if len(iso) < 19 {
		return 0
	}
	y := int(iso[0]-'0')*1000 + int(iso[1]-'0')*100 + int(iso[2]-'0')*10 + int(iso[3]-'0')
	mo := int(iso[5]-'0')*10 + int(iso[6]-'0')
	d := int(iso[8]-'0')*10 + int(iso[9]-'0')
	h := int(iso[11]-'0')*10 + int(iso[12]-'0')
	mi := int(iso[14]-'0')*10 + int(iso[15]-'0')
	s := int(iso[17]-'0')*10 + int(iso[18]-'0')

	// Days from epoch (1970-01-01) to date
	// Convert year/month/day to days since epoch
	dy := y - 1970
	days := dy*365 + dy/4 - dy/100 + dy/400
	// Adjust for leap year overcounting
	if y > 1970 {
		// Account for century adjustments
		days -= (1970/4 - 1970/100 + 1970/400)
		days += (y-1)/4 - (y-1)/100 + (y-1)/400
		days = (y-1970)*365 + ((y-1)/4 - 1969/4) - ((y-1)/100 - 1969/100) + ((y-1)/400 - 1969/400)
	}

	monthDays := [12]int{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
	isLeap := (y%4 == 0 && y%100 != 0) || y%400 == 0
	for i := 0; i < mo-1; i++ {
		days += monthDays[i]
	}
	if isLeap && mo > 2 {
		days++
	}
	days += d - 1

	return int64(days)*86400000 + int64(h)*3600000 + int64(mi)*60000 + int64(s)*1000
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
		fmt.Fprintf(os.Stderr, "[fatal] bad magic %x\n", magic)
		os.Exit(1)
	}

	numVectors = int(binary.LittleEndian.Uint32(data[4:8]))
	numClusters = int(binary.LittleEndian.Uint32(data[8:12]))

	centOffset := 16
	centBytes := numClusters * DIMS * 4
	sizesOffset := centOffset + centBytes
	offsOffset := sizesOffset + numClusters*4
	ratesOffset := offsOffset + numClusters*4
	vecOffset := ((ratesOffset + numClusters*4) + 1) &^ 1
	lblOffset := vecOffset + numVectors*DIMS*2

	centroids = make([]float32, numClusters*DIMS)
	for i := range centroids {
		centroids[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[centOffset+i*4:]))
	}

	clusterSizes = make([]uint32, numClusters)
	for i := range clusterSizes {
		clusterSizes[i] = binary.LittleEndian.Uint32(data[sizesOffset+i*4:])
	}

	clusterOffsets = make([]uint32, numClusters)
	for i := range clusterOffsets {
		clusterOffsets[i] = binary.LittleEndian.Uint32(data[offsOffset+i*4:])
	}

	centroidFraudRates = make([]float32, numClusters)
	for i := range centroidFraudRates {
		centroidFraudRates[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[ratesOffset+i*4:]))
	}

	vectors = make([]uint16, numVectors*DIMS)
	for i := range vectors {
		vectors[i] = binary.LittleEndian.Uint16(data[vecOffset+i*2:])
	}

	labels = make([]uint8, numVectors)
	copy(labels, data[lblOffset:lblOffset+numVectors])
}

func loadNormalization() {
	normData, err := os.ReadFile(dataDir + "/normalization.json")
	if err != nil {
		fmt.Fprintf(os.Stderr, "[fatal] read normalization: %v\n", err)
		os.Exit(1)
	}
	var norm struct {
		MaxAmount         float64 `json:"max_amount"`
		MaxInstallments   float64 `json:"max_installments"`
		AmountVsAvgRatio  float64 `json:"amount_vs_avg_ratio"`
		MaxMinutes        float64 `json:"max_minutes"`
		MaxKm             float64 `json:"max_km"`
		MaxTxCount24h     float64 `json:"max_tx_count_24h"`
		MaxMerchantAvgAmt float64 `json:"max_merchant_avg_amount"`
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
	sc := scratchPool.Get().(*searchScratch)
	centDists := sc.centDists[:numClusters]
	topClusters := sc.topClusters[:nprobeMax]

	// Scale query to Uint16 integer space (avoid float in inner loop)
	var sq [DIMS]int32
	for d := 0; d < DIMS; d++ {
		if query[d] < 0 {
			sq[d] = 65535
		} else {
			sq[d] = int32(query[d] * 65534)
		}
	}

	hasSentinel := query[5] < 0

	// Distance from query to every centroid (float32 space — centroids are float32)
	for c := 0; c < numClusters; c++ {
		base := c * DIMS
		cent := centroids[base : base+DIMS : base+DIMS] // BCE
		var dist float32
		for d := 0; d < DIMS; d++ {
			diff := query[d] - cent[d]
			dist += diff * diff
		}
		centDists[c] = dist
	}

	// Select top NPROBE_MAX clusters (partial selection sort)
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
	nearC := topClusters[0]
	nearCent := centroids[nearC*DIMS : nearC*DIMS+DIMS : nearC*DIMS+DIMS]
	for d := 0; d < DIMS; d++ {
		diff := query[d] - nearCent[d]
		nearDist += diff * diff
	}
	farC := topClusters[nprobeBase]
	farCent := centroids[farC*DIMS : farC*DIMS+DIMS : farC*DIMS+DIMS]
	for d := 0; d < DIMS; d++ {
		diff := query[d] - farCent[d]
		farDist += diff * diff
	}

	ratio := float32(100.0)
	if nearDist > 0 {
		ratio = farDist / nearDist
	}

	var nprobe int
	if ratio < 2.5 {
		nprobe = nprobeMax
	} else if ratio > 4.0 {
		nprobe = nprobeMin
	} else {
		nprobe = nprobeBase
	}

	// Tier 1: pure-legit fast-path
	if ratio > 10.0 && centroidFraudRates[topClusters[0]] == 0.0 {
		scratchPool.Put(sc)
		return 0
	}

	// Linear scan using pure integer arithmetic (no float64 conversions)
	var topDist [K]int64
	var topLbl [K]uint8
	const maxDist = int64(math.MaxInt64)
	for j := 0; j < K; j++ {
		topDist[j] = maxDist
	}
	worstDist := maxDist
	worstIdx := 0

	// Early termination threshold: if all K=5 neighbors are within this distance,
	// the classification is already confident. ~0.14 normalized = 0.14² × 14 × 65534² ≈ 1.18B
	const earlyTermThreshold int64 = 1_180_000_000

	// Pre-compute sentinel distances (integer)
	sq5 := int64(sq[5])
	sq6 := int64(sq[6])
	sentDist5 := (65534 + sq5) * (65534 + sq5)
	sentDist6 := (65534 + sq6) * (65534 + sq6)

	for ci := 0; ci < nprobe; ci++ {
		// Early termination: if worst of K=5 is already below threshold, stop
		if ci >= 2 && worstDist < earlyTermThreshold {
			break
		}

		c := topClusters[ci]
		offset := int(clusterOffsets[c])
		size := int(clusterSizes[c])
		vecBase := offset * DIMS

		// Bounds check elimination: validate entire cluster is accessible
		if vecBase+size*DIMS > len(vectors) {
			continue
		}
		clusterVecs := vectors[vecBase : vecBase+size*DIMS]
		clusterLabels := labels[offset : offset+size]

		if hasSentinel {
			for i := 0; i < size; i++ {
				base := i * DIMS
				vec := clusterVecs[base : base+DIMS : base+DIMS] // BCE hint

				var dist int64
				for d := 0; d < 5; d++ {
					diff := int64(sq[d]) - int64(vec[d])
					dist += diff * diff
				}
				// Sentinel query dims 5,6: distance = (65534+refVal)² if ref is not sentinel
				rv5 := vec[5]
				if rv5 != 65535 {
					cd5 := int64(65534 + int64(rv5))
					dist += cd5 * cd5
				}
				rv6 := vec[6]
				if rv6 != 65535 {
					cd6 := int64(65534 + int64(rv6))
					dist += cd6 * cd6
				}
				for d := 7; d < DIMS; d++ {
					diff := int64(sq[d]) - int64(vec[d])
					dist += diff * diff
				}
				if dist < worstDist {
					topDist[worstIdx] = dist
					topLbl[worstIdx] = clusterLabels[i]
					worstDist = 0
					for j := 0; j < K; j++ {
						if topDist[j] > worstDist {
							worstDist = topDist[j]
							worstIdx = j
						}
					}
				}
			}
		} else {
			for i := 0; i < size; i++ {
				base := i * DIMS
				vec := clusterVecs[base : base+DIMS : base+DIMS] // BCE hint

				var dist int64
				for d := 0; d < 5; d++ {
					diff := int64(sq[d]) - int64(vec[d])
					dist += diff * diff
				}
				// Non-sentinel query: if ref is sentinel, use precomputed large distance
				if vec[5] == 65535 {
					dist += sentDist5
				} else {
					diff := sq5 - int64(vec[5])
					dist += diff * diff
				}
				if vec[6] == 65535 {
					dist += sentDist6
				} else {
					diff := sq6 - int64(vec[6])
					dist += diff * diff
				}
				for d := 7; d < DIMS; d++ {
					diff := int64(sq[d]) - int64(vec[d])
					dist += diff * diff
				}
				if dist < worstDist {
					topDist[worstIdx] = dist
					topLbl[worstIdx] = clusterLabels[i]
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

	scratchPool.Put(sc)

	fraudCount := 0
	for j := 0; j < K; j++ {
		if topDist[j] < maxDist && topLbl[j] == 1 {
			fraudCount++
		}
	}
	return fraudCount
}
