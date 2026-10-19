#!/bin/bash
# Qwen 3.6 35B-A3B Q8_K_XL - GPUs 2,3, port 8602, 256K ctx
# Slot 2 of the ai-smil model layout (node port 8602, tailnet 10082).
# Kept on 3.6: Qwen 3.8 has no 35B-A3B variant.
# Service: qwen-gpu23.service on ai-smil1.
set -euo pipefail
[ -f /opt/rh/gcc-toolset-13/enable ] && source /opt/rh/gcc-toolset-13/enable
export CUDA_VISIBLE_DEVICES=2,3
MODEL=$HOME/models/qwen3.6-35b-a3b-MTP/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf
BIN=$HOME/llama.cpp/build/bin/llama-server
exec "$BIN" \
    --model "$MODEL" \
    --host 0.0.0.0 --port 8602 \
    --n-gpu-layers 99 \
    --tensor-split 1,1 \
    --alias qwen3.6-35b-a3b \
    --ctx-size 262144 \
    --flash-attn on \
    --temp 0.7 --top-p 0.8 --top-k 20 \
    "$@"
