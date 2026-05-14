# ppc64le V100 Performance Notes

This setup is one node with four V100 32 GB GPUs. The measured fast decode
configuration is:

```sh
DS4_CUDA_SPLIT_OUTPUT_HEAD=1
DS4_CUDA_DEVICES=0,1,2,3
DS4_CUDA_TENSOR_SPLIT=8.2,8.2,8.2,8.2
DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=512
```

Use `scripts/server-ds4-flash-v100-1node.sh` for OpenAI/Anthropic-compatible
serving, or `scripts/profile-ds4-flash-v100-1node.sh` for a reproducible decode
profile run.

## Current Profile

Profile command used the same fast CUDA split plus:

```sh
DS4_METAL_GRAPH_TOKEN_PROFILE=1
DS4_METAL_DECODE_STAGE_PROFILE=1
DS4_CUDA_MOE_PROFILE=1
```

Short profile run result:

```text
ds4: prefill: 0.13 t/s, generation: 9.41 t/s
```

The profile is slower than normal generation because the stage profiler
synchronizes frequently. The normal 96-token generation benchmark for this
build was:

```text
ds4: prefill: 0.19 t/s, generation: 10.72 t/s
```

Decode stage totals from the synchronized profile:

```text
routed_moe          243.236 ms total, 0.707 ms avg
attn_output         130.105 ms total, 0.378 ms avg
q_path              120.330 ms total, 0.350 ms avg
shared_down          61.547 ms total, 0.179 ms avg
compressor_indexer   51.913 ms total, 0.151 ms avg
ffn_hc_pre           42.682 ms total, 0.124 ms avg
attn_hc_pre          41.548 ms total, 0.121 ms avg
shared_gate_up       35.037 ms total, 0.102 ms avg
attention            33.485 ms total, 0.097 ms avg
router               28.487 ms total, 0.083 ms avg
kv_path              13.797 ms total, 0.040 ms avg
```

MoE subtotals:

```text
total  216.788 ms total, 0.630 ms avg
down   107.295 ms total, 0.312 ms avg
gateup  98.202 ms total, 0.285 ms avg
xq       5.592 ms total, 0.016 ms avg
midq     4.257 ms total, 0.012 ms avg
sort     0.647 ms total, 0.002 ms avg
sum      0.635 ms total, 0.002 ms avg
```

## CUDA Graph Island Triage

The llama.cpp commit named "accelerate DeepSeek V4 CUDA graph islands" mainly
adds custom DSV4 CUDA op islands to GGML:

- HC split/sinkhorn
- HC weighted sum
- HC expand
- FP8 KV quantize
- rope tail

Standalone DS4 already has equivalent or more specialized CUDA kernels for
those paths, including fused HC split/weighted-sum/norm, compressor update,
qkv RMS fusion, FP8 KV quantize, and rope tail.

An opt-in experiment tried actual CUDA Graph capture around the current hot
one-token F16 cuBLAS path. CUDA/cuBLAS rejected capture on this path:

```text
ds4: CUDA f16 cuBLAS graph capture disabled: operation not permitted when stream is capturing
```

The run fell back to the normal path and measured:

```text
ds4: prefill: 0.19 t/s, generation: 10.70 t/s
```

That experiment was reverted. Future graph work should target a pure-kernel
island rather than cuBLAS stream capture, or use a cuBLASLt path known to be
capture-compatible on the deployed CUDA/V100 stack.
