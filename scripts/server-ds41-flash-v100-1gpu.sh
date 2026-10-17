#!/usr/bin/env bash
set -euo pipefail

# DeepSeek V4.1 Flash on one V100 with SSD streaming. V4.1 CUDA supports only
# one GPU per rank, so routed experts stream from the GGUF. On AC922 the host
# page cache (631 GiB) holds the whole file and NVLink feeds the GPU, so keep
# reads in the page cache: direct I/O and page dropping force SATA SSD reads
# and cut decode from ~4.5 to ~0.2 t/s.
MODEL=${DS4_MODEL:-$HOME/ds4.1/gguf/DeepSeek-V4.1-Flash-Q2.gguf}
BIN=${DS4_BIN:-$HOME/ds4.1/ds4-server}
HOST=${DS4_HOST:-0.0.0.0}
PORT=${DS4_PORT:-8082}
CTX=${DS4_CTX:-32768}
KV_DIR=${DS4_KV_DIR:-$HOME/ds4.1-kv}
KV_MB=${DS4_KV_MB:-8192}

[ -f /opt/rh/gcc-toolset-13/enable ] && source /opt/rh/gcc-toolset-13/enable

export CUDA_VISIBLE_DEVICES=${DS4_CUDA_DEVICE_ID:-0}
export DS4_CUDA_NO_DIRECT_IO=${DS4_CUDA_NO_DIRECT_IO:-1}
export DS4_CUDA_KEEP_MODEL_PAGES=${DS4_CUDA_KEEP_MODEL_PAGES:-1}

exec "$BIN" \
    --cuda \
    --ssd-streaming \
    --model "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --ctx "$CTX" \
    --kv-disk-dir "$KV_DIR" \
    --kv-disk-space-mb "$KV_MB" \
    "$@"
