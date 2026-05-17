package main

import (
"testing"
"time"
)

func TestParseEpochMs(t *testing.T) {
tests := []string{
"2026-03-11T18:45:53Z",
"2026-01-01T00:00:00Z",
"2026-12-31T23:59:59Z",
"2026-02-28T12:30:00Z",
"2024-02-29T06:15:30Z", // leap year
}
for _, iso := range tests {
got := parseEpochMs(iso)
// Parse with time.Date for comparison
y := int(iso[0]-'0')*1000 + int(iso[1]-'0')*100 + int(iso[2]-'0')*10 + int(iso[3]-'0')
mo := int(iso[5]-'0')*10 + int(iso[6]-'0')
d := int(iso[8]-'0')*10 + int(iso[9]-'0')
h := int(iso[11]-'0')*10 + int(iso[12]-'0')
mi := int(iso[14]-'0')*10 + int(iso[15]-'0')
s := int(iso[17]-'0')*10 + int(iso[18]-'0')
expected := time.Date(y, time.Month(mo), d, h, mi, s, 0, time.UTC).UnixMilli()
if got != expected {
t.Errorf("parseEpochMs(%q) = %d, want %d (diff=%d)", iso, got, expected, got-expected)
}
}
}

func TestDayOfWeekZeller(t *testing.T) {
tests := []string{
"2026-03-11T18:45:53Z", // Wednesday
"2026-01-01T00:00:00Z", // Thursday
"2026-05-17T10:00:00Z", // Sunday
"2024-02-29T06:15:30Z", // Thursday (leap year)
}
for _, iso := range tests {
got := dayOfWeekZeller(iso)
y := int(iso[0]-'0')*1000 + int(iso[1]-'0')*100 + int(iso[2]-'0')*10 + int(iso[3]-'0')
m := int(iso[5]-'0')*10 + int(iso[6]-'0')
d := int(iso[8]-'0')*10 + int(iso[9]-'0')
expected := (int(time.Date(y, time.Month(m), d, 0, 0, 0, 0, time.UTC).Weekday()) + 6) % 7
if got != expected {
t.Errorf("dayOfWeekZeller(%q) = %d, want %d", iso, got, expected)
}
}
}
