#!/usr/bin/env bash
set -euo pipefail

MODEL=${DS4_MODEL:-$HOME/models/deepseek-v4-flash/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}
BIN=${DS4_BIN:-$HOME/ds4/ds4-server}
HOST=${DS4_HOST:-0.0.0.0}
PORT=${DS4_PORT:-8080}
CTX=${DS4_CTX:-262144}
KV_DIR=${DS4_KV_DIR:-$HOME/ds4-kv}
KV_MB=${DS4_KV_MB:-8192}

[ -f /opt/rh/gcc-toolset-13/enable ] && source /opt/rh/gcc-toolset-13/enable

export DS4_CUDA_SPLIT_OUTPUT_HEAD=${DS4_CUDA_SPLIT_OUTPUT_HEAD:-1}
export DS4_CUDA_DEVICES=${DS4_CUDA_DEVICES:-0,1,2,3}
export DS4_CUDA_TENSOR_SPLIT=${DS4_CUDA_TENSOR_SPLIT:-8.2,8.2,8.2,8.2}
export DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=${DS4_CUDA_WEIGHT_ARENA_CHUNK_MB:-512}

exec "$BIN" \
    --cuda \
    --model "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --ctx "$CTX" \
    --kv-disk-dir "$KV_DIR" \
    --kv-disk-space-mb "$KV_MB" \
    "$@"
