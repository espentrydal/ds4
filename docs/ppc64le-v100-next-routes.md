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
   gate/up, constant-memory LUT lookup, half/full-warp down kernels, and a
   5-row x 6-slot direct-sum down kernel. Further gains probably need a new
   decode-specialized down accumulation strategy that preserves per-expert
   parallelism while reducing intermediate traffic.

2. Split-output cold-start instrumentation.

   The stage profile is layer-focused, so an opt-in `DS4_OUTPUT_HEAD_PROFILE=1`
   profiler was added for the final output head. It synchronizes before the
   output head and between its substeps, so it is diagnostic only. A second
   opt-in, `DS4_CUDA_SPLIT_OUTPUT_PROFILE=1`, breaks down split-output enqueue
   and synchronize time. This corrected the initial output-logits interpretation:
   the large `~67 ms/token` average included the first lazy output-weight cache
   during prefill, not warm decode.

   ```text
   first split-output call:  1052.8 ms total, about 263 ms enqueue per device
   warm split-output call:      1.35 ms total, about 1.14 ms waiting on dev0
   ```

   A narrow opt-in
   `DS4_CUDA_OUTPUT_F16_CACHE=1` path can cache `output`/`output_split` Q8
   weights as F16; with a lower cache reserve it did cache the four output
   segments, but it regressed on V100. Output logits are therefore not a
   meaningful warm-decode route toward 15 tok/s.

3. Attention-output projection fusion.

   The current reference attention-output path is faster than the fused HC path,
   but `attn_output` is still the second-largest layer bucket. One-token cuBLAS
   for `attn_output_a` was flat, so the likely route is a purpose-built
   out-a/out-b/HC expansion fusion that avoids extra intermediate traffic
   without repeating the slower older fused path.

4. Compressor/indexer path.

   `compressor_indexer` is smaller than MoE and attention output, but still large
   enough to matter. The current short-prompt decode profile does not exercise
   top-k selection yet: `DS4_N_INDEXER_TOP_K` is 512, and ratio-4 layers need
   roughly 2048 tokens before `n_comp > top_k`. In the current 12.6-13 tok/s
   benchmark, this bucket is therefore mostly compressor projection/update and
   compressed-cache maintenance, not score/top-k. Any change here must preserve
   the compressed-attention selection behavior. Candidate work is fusing small
   compressor kernels or reducing cache-update traffic; top-k/indexer scoring is
   more relevant to long-context decode than to the current short-context speed
   target.

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
- Output-head instrumentation showed that final logits are a major decode
  bucket only if the first cold split-output cache call is included. Warm
  split-output logits are about `1.35 ms`, so this is not a major generation
  target.
- A temporary half-warp output Q8 kernel shape was flat against the same-node
  baseline (`67.02 ms` vs `67.10 ms` logits) and was reverted.
- A temporary full-block DP4A output Q8 specialization was also flat
  (`67.09 ms` including the cold call) and was reverted.
- MoE row-span toggles were flat or worse: `DS4_CUDA_MOE_DOWN_ROW1024=1`
  measured `down=0.296 ms` and `total=0.587 ms`, `DOWN_ROW512` regressed to
  `down=0.300 ms`, and gate row 512/2048 both left `gateup` around
  `0.261 ms`.
- `nvprof` confirms the warm decode MoE kernels are
  `moe_down_qwarp32_kernel` at roughly `293 us/layer` and
  `moe_gate_up_midq_decode_lut_qwarp32_kernel` at roughly `256 us/layer`.
  `ncu` hardware counters are blocked by `ERR_NVGPUCTRPERM` on these nodes.
- A temporary `__forceinline__` test on the Q2 down dot helpers was flat to
  slightly worse (`down=0.298-0.300 ms`) and was reverted.
- A temporary 5-row x 6-slot direct-sum down kernel kept the six experts
  parallel and removed most explicit sum time (`sum 0.009 -> 0.002 ms`), but
  increased down time (`down 0.299 -> 0.320 ms`) and total MoE time
  (`0.590 -> 0.599 ms`), so it was reverted.

The latest dev profile still points to routed MoE as the realistic large
single-node target. Reaching 15 tok/s likely needs a new MoE down projection
algorithm rather than another environment toggle.

## 2026-05-16 Indexer Check

`DS4_METAL_INDEXER_STAGE_PROFILE=1` was rechecked after the MoE/output sweeps.
The script now aggregates `metal indexer stage` lines when they appear, but the
standard short decode profile emits none because the top-k path only starts
after the ratio-4 compressed row count exceeds 512. The layer-level profile still
shows `compressor_indexer` around `0.151-0.152 ms/layer`, but for the current
benchmark that should be read as compressor/update overhead, not top-k overhead.

This keeps the 15 tok/s priority order unchanged:

1. new routed-MoE down/gate-up algorithm,
2. attention-output projection fusion,
3. compressor/cache-update fusion as a secondary route,
4. long-context indexer top-k work only if long-context decode becomes the main
   target.
