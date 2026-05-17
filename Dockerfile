# ====================================================================
FROM oven/bun:1 AS preprocessor

WORKDIR /build

COPY scripts/ ./scripts/
COPY resources/references.json.gz  ./resources/
COPY resources/normalization.json  ./resources/
COPY resources/mcc_risk.json       ./resources/

RUN DATA_DIR=/data \
    REFS_PATH=./resources/references.json.gz \
    NORM_PATH=./resources/normalization.json \
    MCC_PATH=./resources/mcc_risk.json \
    bun run scripts/preprocess.ts

# ====================================================================
FROM golang:1.24-alpine AS builder

WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download

COPY *.go ./

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /app/server .

# ====================================================================
FROM alpine:3.21 AS runner

COPY --from=builder      /app/server /app/server
COPY --from=preprocessor /data       /data

ENV DATA_DIR=/data
ENV SOCK_PATH=/var/run/api/api.sock
ENV NPROBE=7
ENV NPROBE_MIN=5
ENV NPROBE_MAX=12
ENV GOGC=off
ENV GOMAXPROCS=4

ENTRYPOINT ["/app/server"]
