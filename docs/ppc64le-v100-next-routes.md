# ppc64le V100 Next Performance Routes

Current best verified one-node decode result:

```text
ds4: prefill: 0.19 t/s, generation: 12.76 t/s
```

Getting to 15 tok/s requires roughly a 15 percent token-latency reduction from
the current path. The latest synchronized profile still has most remaining time
inside the layer loop:

```text
routed_moe          0.662 ms/layer
attn_output         0.277 ms/layer
q_path              0.188 ms/layer
shared_down         0.175 ms/layer
compressor_indexer  0.152 ms/layer
```

## Most Realistic Routes

1. Routed-MoE down/gate-up kernel work.

   This remains the largest bucket. The fused midq kernel helped, but MoE is
   still dominated by down and gate/up work:

   ```text
   down    0.298 ms/layer
   gateup  0.258 ms/layer
   ```

   Simple variants have already regressed: direct sum6, atomic down, no-LUT
   gate/up, constant-memory LUT lookup, and half/full-warp down kernels. Further
   gains probably need a new decode-specialized down accumulation strategy that
   preserves per-expert parallelism while reducing intermediate traffic.

2. Output/logits path instrumentation.

   The current stage profile is layer-focused and does not break down the final
   output-head/logits path clearly. Before changing more output code, add timing
   around output projection, split-device synchronization, sampling, and any
   device-host transfer. Previous opt-in F16 output-split caching was flat, but
   the full output/logits path may still hide non-layer latency.

3. Attention-output projection fusion.

   The current reference attention-output path is faster than the fused HC path,
   but `attn_output` is still the second-largest layer bucket. One-token cuBLAS
   for `attn_output_a` was flat, so the likely route is a purpose-built
   out-a/out-b/HC expansion fusion that avoids extra intermediate traffic
   without repeating the slower older fused path.

4. Compressor/indexer path.

   `compressor_indexer` is smaller than MoE and attention output, but still large
   enough to matter. Any change here must preserve the compressed-attention
   selection behavior. Candidate work is reducing top-k/indexer overhead or
   fusing small compressor/indexer kernels.

5. Pure-kernel CUDA graph islands.

   CUDA graph capture around cuBLAS failed on this stack. Graph work may still
   help for pure-kernel islands, but the current hot path now includes more
   cuBLAS GEMV, so graphing is less likely to be the biggest win unless scoped
   to non-cuBLAS groups.

6. Two-node execution.

   Cross-node tensor parallelism over InfiniBand is unlikely to improve
   single-stream decode latency. A layer-pipeline split could be explored, but
   it is a larger scheduler/runtime change and should be treated separately from
   one-node kernel work.

## Low-Priority Or Already Tested

- MTP/speculative decoding: no useful single-stream win on this setup.
- vLLM: may help serving/batching, but not expected to beat this one-stream DS4
  CUDA path without substantial PPC64/CUDA integration work.
- Existing MoE env toggles: already swept; no better mode found.
- F16 output split cache: flat.
- Shared gate/up F16 fallback: slower than the native pair kernel.

## 2026-05-15 Follow-Up Sweep Toward 15 tok/s

After restoring service-path decode to roughly 13 tok/s server-side, another
dev-node sweep did not find a safe route to 15 tok/s:

- Lowering `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB` to 2048 or 1024 did not improve
  warm decode. The 1024 MiB reserve triggered CUDA arena allocation failures.
- Skewing `DS4_CUDA_TENSOR_SPLIT` from `8.2,8.2,8.2,8.2` to
  `7.0,8.2,9.4,8.2` moved memory pressure to GPU2 and regressed throughput to
  about 11.5 tok/s server-side.
- Rechecking `DS4_CUDA_MOE_DIRECT_DOWN_SUM6=1` still regressed.
- Disabling attention-output F16 caching freed VRAM but did not improve total
  speed; MoE arena allocation still failed on the dev node.
- Replacing MoE `expf` with explicit `__expf` did not help; the V100 CUDA build
  already uses `--use_fast_math`.
- A decode-only shared-`midq` MoE-down kernel was tested and regressed. The
  added shared-memory staging overhead outweighed the reduced `midq` reloads.

The latest dev profile still points to routed MoE as the only realistic large
single-node target, especially the down projection. Reaching 15 tok/s likely
needs a new down-projection algorithm rather than another environment toggle.
