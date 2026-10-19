#!/usr/bin/env bash
set -euo pipefail

# DeepSeek V4.1 Flash on one V100 with SSD streaming. V4.1 CUDA supports only
# one GPU per rank, so routed experts stream from the GGUF. On AC922 the host
# page cache (631 GiB) holds the whole file and NVLink feeds the GPU, so keep
# reads in the page cache: direct I/O and page dropping force SATA SSD reads
# and cut decode from ~4.5 to ~0.2 t/s.
#
# POWER9 reads the mmap through the host page tables (ATS), so selected experts
# upload directly from it once the mapping is prefaulted. The expert cache is
# sized for long contexts: 448 slots leave ~1.7 GiB of VRAM at 64k context,
# while filling VRAM to the default 8 GiB reserve (~345 slots) wastes it and a
# 1 GiB reserve with automatic sizing (~900 slots) starves long prefill. Prefill
# uses 4096-row sweeps; the 8192-row default needs ~4 GiB more than a 32 GB
# V100 has once the cache is sized this way.
MODEL=${DS4_MODEL:-$HOME/ds4.1/gguf/DeepSeek-V4.1-Flash-Q2.gguf}
BIN=${DS4_BIN:-$HOME/ds4.1/ds4-server}
HOST=${DS4_HOST:-0.0.0.0}
PORT=${DS4_PORT:-8801}
CTX=${DS4_CTX:-32768}
KV_DIR=${DS4_KV_DIR:-$HOME/ds4.1-kv}
KV_MB=${DS4_KV_MB:-8192}
CACHE_EXPERTS=${DS4_CACHE_EXPERTS:-448}
# A 30k-token default would skip the prefix snapshot of long first prompts, so a
# resent or edited long prompt would prefill again from token zero (~160 s per
# 34k tokens here).
KV_COLD_MAX=${DS4_KV_COLD_MAX:-$CTX}

[ -f /opt/rh/gcc-toolset-13/enable ] && source /opt/rh/gcc-toolset-13/enable

export CUDA_VISIBLE_DEVICES=${DS4_CUDA_DEVICE_ID:-0}
export DS4_CUDA_NO_DIRECT_IO=${DS4_CUDA_NO_DIRECT_IO:-1}
export DS4_CUDA_KEEP_MODEL_PAGES=${DS4_CUDA_KEEP_MODEL_PAGES:-1}
export DS4_CUDA_SSD_CACHE_RESERVE_MB=${DS4_CUDA_SSD_CACHE_RESERVE_MB:-1024}
export DS4_CUDA_SSD_PAGEABLE_DMA=${DS4_CUDA_SSD_PAGEABLE_DMA:-1}
export DS4_CUDA_SSD_PREFAULT_MODEL=${DS4_CUDA_SSD_PREFAULT_MODEL:-1}
export DS4_METAL_DISABLE_V41_8K_CHUNK=${DS4_METAL_DISABLE_V41_8K_CHUNK:-1}

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
