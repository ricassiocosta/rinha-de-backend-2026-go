package main

import (
	"time"
)

// Payload represents the JSON body of POST /fraud-score
type Payload struct {
	Transaction struct {
		Amount       float64 `json:"amount"`
		Installments float64 `json:"installments"`
		RequestedAt  string  `json:"requested_at"`
	} `json:"transaction"`
	Customer struct {
		AvgAmount      float64  `json:"avg_amount"`
		TxCount24h     float64  `json:"tx_count_24h"`
		KnownMerchants []string `json:"known_merchants"`
	} `json:"customer"`
	Merchant struct {
		ID        string  `json:"id"`
		MCC       string  `json:"mcc"`
		AvgAmount float64 `json:"avg_amount"`
	} `json:"merchant"`
	Terminal struct {
		KmFromHome  float64 `json:"km_from_home"`
		IsOnline    bool    `json:"is_online"`
		CardPresent bool    `json:"card_present"`
	} `json:"terminal"`
	LastTransaction *struct {
		Timestamp    string  `json:"timestamp"`
		KmFromCurrent float64 `json:"km_from_current"`
	} `json:"last_transaction"`
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

// parseHourUTC extracts hour (0-23) from ISO "2026-03-11T18:45:53Z"
func parseHourUTC(iso string) int {
	if len(iso) < 13 {
		return 0
	}
	return int(iso[11]-'0')*10 + int(iso[12]-'0')
}

// parseDayOfWeekUTC returns Mon=0 ... Sun=6
func parseDayOfWeekUTC(iso string) int {
	if len(iso) < 10 {
		return 0
	}
	y := int(iso[0]-'0')*1000 + int(iso[1]-'0')*100 + int(iso[2]-'0')*10 + int(iso[3]-'0')
	m := int(iso[5]-'0')*10 + int(iso[6]-'0')
	d := int(iso[8]-'0')*10 + int(iso[9]-'0')
	t := time.Date(y, time.Month(m), d, 0, 0, 0, 0, time.UTC)
	// Go Weekday: Sunday=0 ... Saturday=6 → shift to Mon=0 ... Sun=6
	return (int(t.Weekday()) + 6) % 7
}

// parseMs parses ISO timestamp to Unix milliseconds
func parseMs(iso string) int64 {
	if len(iso) < 19 {
		return 0
	}
	y := int(iso[0]-'0')*1000 + int(iso[1]-'0')*100 + int(iso[2]-'0')*10 + int(iso[3]-'0')
	mo := int(iso[5]-'0')*10 + int(iso[6]-'0')
	d := int(iso[8]-'0')*10 + int(iso[9]-'0')
	h := int(iso[11]-'0')*10 + int(iso[12]-'0')
	mi := int(iso[14]-'0')*10 + int(iso[15]-'0')
	s := int(iso[17]-'0')*10 + int(iso[18]-'0')
	t := time.Date(y, time.Month(mo), d, h, mi, s, 0, time.UTC)
	return t.UnixMilli()
}

func vectorize(p *Payload, out *[DIMS]float32) {
	tx := &p.Transaction
	cust := &p.Customer
	merch := &p.Merchant
	term := &p.Terminal

	out[0] = clamp01(tx.Amount / normMaxAmount)
	out[1] = clamp01(tx.Installments / normMaxInstallments)

	avgRatio := 0.0
	if cust.AvgAmount > 0 {
		avgRatio = tx.Amount / cust.AvgAmount / normAmountVsAvgRatio
	}
	out[2] = clamp01(avgRatio)

	out[3] = float32(parseHourUTC(tx.RequestedAt)) / 23.0
	out[4] = float32(parseDayOfWeekUTC(tx.RequestedAt)) / 6.0

	if p.LastTransaction == nil {
		out[5] = -1
		out[6] = -1
	} else {
		currentMs := parseMs(tx.RequestedAt)
		lastMs := parseMs(p.LastTransaction.Timestamp)
		minutesBetween := float64(currentMs-lastMs) / 60000.0
		out[5] = clamp01(minutesBetween / normMaxMinutes)
		out[6] = clamp01(p.LastTransaction.KmFromCurrent / normMaxKm)
	}

	out[7] = clamp01(float64(term.KmFromHome) / normMaxKm)
	out[8] = clamp01(cust.TxCount24h / normMaxTxCount24h)

	if term.IsOnline {
		out[9] = 1
	} else {
		out[9] = 0
	}
	if term.CardPresent {
		out[10] = 1
	} else {
		out[10] = 0
	}

	// Unknown merchant → 1
	out[11] = 1
	for _, km := range cust.KnownMerchants {
		if km == merch.ID {
			out[11] = 0
			break
		}
	}

	if risk, ok := mccRisk[merch.MCC]; ok {
		out[12] = float32(risk)
	} else {
		out[12] = 0.5
	}

	out[13] = clamp01(merch.AvgAmount / normMaxMerchantAvgAmt)
}
