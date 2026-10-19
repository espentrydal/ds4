#!/usr/bin/env bash
set -euo pipefail

# DeepSeek V4.1 Flash on one AC922 node (4x V100-32GB) with SSD streaming and
# expert-parallel decode: GPU 0 runs the dense model and prefill with a small
# expert cache; GPUs 1-3 hold only large expert caches (DS4_EP_SLOTS each, about
# 9.5 MiB per slot) and run the six routed experts of each decoded token. Output
# is identical to the one-GPU path. One server uses all four GPUs. Decode is about
# 20 t/s at 32k-64k context, prefill about 400 t/s. See
# docs/ppc64le-v41-flash-streaming.md and scripts/server-ds41-flash-v100-1gpu.sh
# for the streaming and memory reasoning.
MODEL=${DS4_MODEL:-$HOME/ds4.1/gguf/DeepSeek-V4.1-Flash-Q2.gguf}
BIN=${DS4_BIN:-$HOME/ds4.1/ds4-server}
HOST=${DS4_HOST:-0.0.0.0}
PORT=${DS4_PORT:-8801}
CTX=${DS4_CTX:-65536}
KV_DIR=${DS4_KV_DIR:-$HOME/ds4.1-kv}
KV_MB=${DS4_KV_MB:-16384}
CACHE_EXPERTS=${DS4_CACHE_EXPERTS:-448}
KV_COLD_MAX=${DS4_KV_COLD_MAX:-$CTX}
EP_GPUS=${DS4_EP_GPUS:-1,2,3}
EP_SLOTS=${DS4_EP_SLOTS:-1800}

[ -f /opt/rh/gcc-toolset-13/enable ] && source /opt/rh/gcc-toolset-13/enable
mkdir -p "$KV_DIR"

export CUDA_VISIBLE_DEVICES=${DS4_CUDA_DEVICE_ID:-0,1,2,3}
export DS4_CUDA_NO_DIRECT_IO=${DS4_CUDA_NO_DIRECT_IO:-1}
export DS4_CUDA_KEEP_MODEL_PAGES=${DS4_CUDA_KEEP_MODEL_PAGES:-1}
export DS4_CUDA_SSD_CACHE_RESERVE_MB=${DS4_CUDA_SSD_CACHE_RESERVE_MB:-1024}
export DS4_CUDA_SSD_PAGEABLE_DMA=${DS4_CUDA_SSD_PAGEABLE_DMA:-1}
export DS4_CUDA_SSD_PREFAULT_MODEL=${DS4_CUDA_SSD_PREFAULT_MODEL:-1}
export DS4_METAL_DISABLE_V41_8K_CHUNK=${DS4_METAL_DISABLE_V41_8K_CHUNK:-1}
export DS4_V41_STREAM_GRAPHS=${DS4_V41_STREAM_GRAPHS:-1}
export DS4_V41_EP_GPUS=$EP_GPUS
export DS4_V41_EP_SLOTS=$EP_SLOTS

exec "$BIN" \
    --cuda \
    --ssd-streaming \
    --ssd-streaming-cache-experts "$CACHE_EXPERTS" \
    --model "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --ctx "$CTX" \
    --kv-disk-dir "$KV_DIR" \
    --kv-disk-space-mb "$KV_MB" \
    --kv-cache-cold-max-tokens "$KV_COLD_MAX" \
    "$@"
