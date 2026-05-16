# ppc64le V100 Next Performance Routes

Current best verified one-node decode result:

```text
ds4: prefill: 0.15 t/s, generation: 14.03 t/s
```

Getting to 15 tok/s requires roughly a 6-7 percent token-latency reduction
from the current path. The latest synchronized profile still has most remaining
time inside the layer loop:

```text
routed_moe          0.579 ms/layer
attn_output         0.278 ms/layer
q_path              0.188 ms/layer
compressor_indexer  0.153 ms/layer
shared_gate_up      0.102 ms/layer
shared_down         0.069 ms/layer
```

## Most Realistic Routes

0. Keep `DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION=1` enabled.

   This is the current fastest simple route. It measured `12.64 -> 13.20 t/s`
   on ai-smil1 and `12.55 -> 13.09 t/s` on ai-smil2. The separate shared-down
   and HC-post path is faster than the older fused path.

   Keep the adjacent shared gate/up/SwiGLU fusion enabled. Its opt-out,
   `DS4_METAL_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION=1`, worsened the profiled
   `shared_gate_up` bucket (`0.100 -> 0.114 ms/layer`) and did not show a clear
   direct-generation win. The narrower
   `DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR=1` opt-out also worsened the same
   bucket (`0.100 -> 0.115 ms/layer`).

   Keep QKV pair projection enabled as well. The
   `DS4_CUDA_DISABLE_QKV_PAIR_PROJ=1` opt-out raised `q_path` from
   `0.186 -> 0.253 ms/layer` and dropped direct decode to `12.89 t/s`.

1. Keep the V100 register cap in the `cuda-v100` build.

   A 2026-05-16 compile-flag sweep found that ptxas register pressure matters
   on V100. `-Xptxas=-maxrregcount=64` raised direct 200-token decode to
   `13.40-13.43 t/s`. Nearby caps were weaker or not reproducibly better:
   56 regs measured `13.26 t/s`, 60 regs ranged `13.37-13.49 t/s`, 62 regs
   `13.43 t/s`, 68 regs `13.35 t/s`, 72 regs `13.37 t/s`, 80 regs
   `13.33 t/s`, and 96 regs regressed to `12.86 t/s`. Longer 500-token swapped
   checks kept 64 as the safer default: ai-smil1 measured `13.46 t/s` with 64
   versus `13.44 t/s` with 60, while ai-smil2 measured `13.33 t/s` with both.
   The `cuda-v100` target applies the 64-register cap through
   `NVCC_PTXAS_FLAGS`; other CUDA targets remain unchanged.

2. Keep the 64-row MoE decode down kernel.

   The default decode down path now processes 64 output rows per block with 512
   threads. This moved the profiled MoE down bucket from about
   `0.318 ms/layer` to `0.236 ms/layer` and reduced routed MoE from about
   `0.662 ms/layer` to `0.579 ms/layer`. Direct 200-token checks measured
   `13.96 t/s` on ai-smil1 and `14.03 t/s` on ai-smil2; a 500-token ai-smil2
   run measured `14.01 t/s`. The old 32-row shape remains available with
   `DS4_CUDA_MOE_DOWN_ROWS32=1` for comparison.

3. Routed-MoE gate-up kernel work.

   MoE remains the largest bucket. After the 64-row down change, gate/up and
   down are roughly tied:

   ```text
   down    0.236 ms/layer
   gateup  0.236 ms/layer
   ```

   Simple earlier variants already regressed: direct sum6, atomic down, no-LUT
   gate/up, constant-memory LUT lookup, half/full-warp down kernels, and a
   5-row x 6-slot direct-sum down kernel. Further gains probably need work on
   the fused gate/up+midq kernel, or another down shape that beats the new
   64-row default.

4. Split-output cold-start instrumentation.

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

5. Attention-output projection fusion.

   The current reference attention-output path is faster than the fused HC path,
   but `attn_output` is still the second-largest layer bucket. One-token cuBLAS
   for `attn_output_a` was flat, so the likely route is a purpose-built
   out-a/out-b/HC expansion fusion that avoids extra intermediate traffic
   without repeating the slower older fused path. Disabling attention-output
   F16 caching improved the synchronized `attn_output` stage, but same-node
   direct generation regressed (`12.64 -> 12.55 t/s`), so it is not a
   production-default route. CUDA event timing with
   `DS4_CUDA_ATTENTION_OUTPUT_PROFILE=1` showed the default path split almost
   evenly between `attention_output_a` (`0.142 ms/layer`) and
   `attention_output_b` (`0.137 ms/layer`), so meaningful work here probably
   needs to reduce the whole A/B/HC pipeline rather than one subprojection. The
   existing `DS4_METAL_ENABLE_ATTN_OUT_HC_FUSION=1` path was rechecked on the
   current build and still regressed (`12.64 -> 12.19 t/s` direct decode).

6. Compressor/indexer path.

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

7. Pure-kernel CUDA graph islands.

   CUDA graph capture around cuBLAS failed on this stack. Graph work may still
   help for pure-kernel islands, but the current hot path now includes more
   cuBLAS GEMV, so graphing is less likely to be the biggest win unless scoped
   to non-cuBLAS groups.

8. Two-node execution.

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
- After the shared-down/HC opt-out became default, reserve values of 3072 and
  2048 MiB were rechecked. They measured `13.21-13.22 t/s` versus a same-node
  default rerun at `13.19 t/s`, which is not enough to justify a default change.
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
- A V100 ptxas register-cap sweep did find a small build-level win:
  `maxrregcount=64` measured `13.40-13.43 t/s` and is now the `cuda-v100`
  default. Higher and lower caps were weaker or not reproducibly better.
- ptxas `-dlcm=ca` and `-dlcm=cg` cache modifiers did not improve the
  64-register build. `ca` matched the best ai-smil1 long run but regressed
  ai-smil2; `cg` was also below the default path.
- `--extra-device-vectorization` was mixed and should not be a default. It
  improved one short ai-smil1 run but missed on longer/swap checks.
- ptxas verbose output showed no spills in the active one-token MoE decode
  kernels under the 64-register cap. Spilling exists in some batch/tile MoE
  kernels, but those are not the current single-token decode bottleneck.
- A 64-row/512-thread MoE decode down shape produced the next real gain and is
  now the default. The 16-row shape regressed to `12.90 t/s`; the 64-row shape
  repeated at `13.96-14.03 t/s` on 200-token checks and `14.01 t/s` on a
  500-token ai-smil2 check. A same-binary 32-token output comparison with the
  old path matched exactly.

The latest dev profile still points to routed MoE as the realistic large
single-node target. Reaching 15 tok/s likely needs another MoE or
attention-output gain rather than another simple environment toggle or build
flag.

## 2026-05-16 Indexer Check

`DS4_METAL_INDEXER_STAGE_PROFILE=1` was rechecked after the MoE/output sweeps.
The script now aggregates `metal indexer stage` lines when they appear, but the
standard short decode profile emits none because the top-k path only starts
after the ratio-4 compressed row count exceeds 512. The layer-level profile still
shows `compressor_indexer` around `0.151-0.152 ms/layer`, but for the current
benchmark that should be read as compressor/update overhead, not top-k overhead.

After the V100 register cap and 64-row MoE down change, the 15 tok/s priority
order is:

1. fused MoE gate/up+midq work, plus any further down shape that beats rows64,
2. attention-output projection fusion,
3. compressor/cache-update fusion as a secondary route,
4. long-context indexer top-k work only if long-context decode becomes the main
   target.
