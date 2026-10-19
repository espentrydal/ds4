#!/bin/bash
# Qwen 3.8 27B Q6_K_XL - GPUs 0,1, port 8601, 256K ctx, MTP, reasoning medium
# Slot 1 of the ai-smil model layout (node port 8601, tailnet 10081).
# Service: qwen-gpu01.service on ai-smil1.
set -euo pipefail
[ -f /opt/rh/gcc-toolset-13/enable ] && source /opt/rh/gcc-toolset-13/enable
export CUDA_VISIBLE_DEVICES=0,1
MODEL=$HOME/models/qwen3.8-27b/Qwen3.8-27B-UD-Q6_K_XL.gguf
BIN=$HOME/llama.cpp/build/bin/llama-server
exec "$BIN" \
    --model "$MODEL" \
    --host 0.0.0.0 --port 8601 \
    --n-gpu-layers 99 \
    --tensor-split 1,1 \
    --alias qwen3.8-27b \
    --ctx-size 262144 \
    --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --spec-type draft-mtp --spec-draft-n-max 4 \
    --chat-template-kwargs "{\"reasoning_effort\":\"medium\"}" \
    --temp 0.7 --top-p 0.8 --top-k 20 \
    "$@"
