# ppc64le V100 Next Performance Routes

Current best verified default one-node decode result:

```text
ai-smil1: ds4: prefill: 0.13 t/s, generation: 17.26 t/s
ai-smil2: ds4: prefill: 0.13 t/s, generation: 17.26 t/s
```

The 17 tok/s direct target is reached by keeping routed-MoE, attention-output,
and Q/KV scratch tensors local to each layer's assigned GPU with
`DS4_CUDA_LOCAL_ROUTED_SCRATCH=1`, `DS4_CUDA_LOCAL_ATTN_SCRATCH=1`, and
`DS4_CUDA_LOCAL_QKV_SCRATCH=1`. The 512-thread fused MoE gate/up+midq decode
kernel remains part of the fast path; the old 256-thread kernel remains
available for comparison with
`DS4_CUDA_MOE_GATEUP_THREADS256=1`.

The best reserve-tuning one-off measured `14.91 t/s`, but follow-up lower
reserve checks landed at `14.82-14.83 t/s`, so reserve tuning is not a default
route. The latest synchronized profile still has most remaining time inside the
layer loop:

```text
routed_moe          0.483 ms/layer
attn_output         0.284 ms/layer
q_path              0.189 ms/layer
compressor_indexer  0.154 ms/layer
shared_gate_up      0.100 ms/layer
shared_down         0.068 ms/layer
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

2. Keep the 128-row MoE decode down kernel.

   The default decode down path now processes 128 output rows per block with
   1024 threads. This moved the profiled MoE down bucket from the old
   `0.318 ms/layer` to `0.162 ms/layer` and reduced routed MoE to about
   `0.506 ms/layer`. Direct checks measured `14.61-14.66 t/s` on 200-token
   runs, and `14.47 t/s` on an ai-smil2 500-token run. Rows64 and rows32 remain
   available with
   `DS4_CUDA_MOE_DOWN_ROWS64=1` and `DS4_CUDA_MOE_DOWN_ROWS32=1` for
   comparison. `DS4_CUDA_MOE_DOWN_ROWS96=1` also exists as a comparison path,
   but it was flat with rows64 and is not documented as a tuning target.

3. Keep the 512-thread MoE gate/up+midq decode kernel.

   MoE remains the largest bucket, but the 512-thread gate/up variant is the
   change that pushed the main benchmark over 15 tok/s. It keeps the same
   256-row Q8_K output tile, but uses 64 quarter-warps per block and four
   serial rows per quarter-warp instead of 32 quarter-warps and eight serial
   rows. The old path is available with `DS4_CUDA_MOE_GATEUP_THREADS256=1`.

   ```text
   gateup  0.212 ms/layer
   down    0.165 ms/layer
   total   0.407 ms/layer
   ```

   Simple earlier variants already regressed: direct sum6, atomic down, no-LUT
   gate/up, constant-memory LUT lookup, half/full-warp down kernels, and a
   5-row x 6-slot direct-sum down kernel. Further gains probably need another
   fused gate/up+midq shape or a down shape that beats rows128.

3a. Keep routed-MoE scratch local to layer devices.

   `DS4_CUDA_LOCAL_ROUTED_SCRATCH=1` allocates routed-MoE scratch/output tensors
   per layer on the layer's assigned CUDA device. This avoids running the
   largest decode bucket through scratch buffers owned by the primary GPU. It
   preserved the existing deterministic 64-token and 200-token output hashes,
   and raised direct 32K decode to `16.56 t/s` for 64 tokens and `16.47 t/s`
   for 200 tokens on ai-smil1. ai-smil2 matched the 64-token hash at
   `16.48 t/s`.

3b. Keep attention-output scratch local to layer devices.

   `DS4_CUDA_LOCAL_ATTN_SCRATCH=1` applies the same locality idea to attention
   heads, low-rank attention-output scratch, and attention-output buffers. In
   combination with local routed-MoE scratch, it preserved deterministic output
   hashes and measured `16.64 t/s` for 64 tokens and `16.54 t/s` for 200 tokens
   on ai-smil1. ai-smil2 matched the 64-token hash at `16.68 t/s`.

3c. Keep Q/KV scratch local to layer devices.

   `DS4_CUDA_LOCAL_QKV_SCRATCH=1` extends the same locality fix to the front of
   the attention path: attention pre-HC scratch, normalized attention input, QR,
   normalized QR, Q, raw KV, and KV tensors. With routed-MoE and
   attention-output locality already enabled, it preserved the deterministic
   64-token and 200-token hashes and measured `17.23 t/s` for 64 tokens and
   `17.26 t/s` for 200 tokens on ai-smil1. ai-smil2 matched the 64-token hash
   at `17.26 t/s`.

3d. Keep the NVLink-pair-aware device order.

   The runtime assigns contiguous layer ranges according to
   `DS4_CUDA_DEVICES` and `DS4_CUDA_TENSOR_SPLIT`; it does not dynamically
   model topology. On these nodes the correct topology-aware order is
   `0,1,2,3`, because GPU0-GPU1 and GPU2-GPU3 are NVLink pairs. Equal split
   weights map the 43 layers as 0-10, 11-21, 22-31, and 32-42, so two of the
   three inter-GPU layer handoffs stay inside NVLink pairs and only the middle
   handoff crosses the slower `SYS` path.

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

5. Keep one-token cuBLAS for attention-output A.

   The `attn_output_a` F16/cuBLAS path now runs for one-token decode, not just
   prefill/batch. Direct 200-token checks measured `14.85 t/s` on ai-smil1 and
   `14.79 t/s` on ai-smil2; a 500-token ai-smil1 run measured `14.75 t/s`. A
   same-binary 64-token output comparison against the old Q8 decode path matched
   exactly and the cuBLAS-A run measured `15.06 t/s` on that short check.

6. Attention-output projection fusion.

   The current reference attention-output path is faster than the fused HC path,
   but `attn_output` is still the second-largest layer bucket. One-token cuBLAS
   for `attn_output_a` is now the default and gave a small real win, so the
   remaining likely route is `attention_output_b` or a purpose-built
   out-a/out-b/HC expansion pipeline that avoids extra intermediate traffic
   without repeating the slower older fused path. Disabling attention-output F16
   caching improved the synchronized `attn_output` stage, but same-node direct
   generation regressed (`12.64 -> 12.55 t/s`), so it is not a
   production-default route. CUDA event timing with
   `DS4_CUDA_ATTENTION_OUTPUT_PROFILE=1` showed the default path split almost
   evenly between `attention_output_a` (`0.142 ms/layer`) and
   `attention_output_b` (`0.137 ms/layer`), so meaningful work here probably
   needs to reduce the whole A/B/HC pipeline rather than one subprojection. The
   existing `DS4_METAL_ENABLE_ATTN_OUT_HC_FUSION=1` path was rechecked on the
   current build and still regressed (`12.64 -> 12.19 t/s` direct decode).
   A later temporary variant let the opt-in HC-fused path use the cached
   F16/cuBLAS A projection before the fused Q8 B/HC expand. It still regressed
   (`14.15` vs `14.62 t/s` on the short ai-smil1 check) and did not match the
   reference output (`cmp` exit 1), so it was reverted.

7. Compressor/indexer path.

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

8. Pure-kernel CUDA graph islands.

   CUDA graph capture around cuBLAS failed on this stack. Graph work may still
   help for pure-kernel islands, but the current hot path now includes more
   cuBLAS GEMV, so graphing is less likely to be the biggest win unless scoped
   to non-cuBLAS groups.

9. Two-node execution.

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
- A 64-row/512-thread MoE decode down shape produced a real gain, and the
  follow-up 128-row/1024-thread shape improved it further. Rows16 regressed to
  `12.90 t/s`, rows64 reached roughly `14.0 t/s`, rows96 was flat with rows64,
  and rows128 repeated at `14.47-14.66 t/s` on longer checks. A same-binary
  32-token output comparison between rows64 and rows128 matched exactly.
- Q8 warp-row geometry for attention-output/general Q8 kernels was checked and
  reverted. Rows16 measured `14.51 t/s` and rows32 measured `14.39 t/s`, both
  below the rows128 MoE default path.
- One-token cuBLAS for attention-output A produced a small real gain and is now
  the default. It repeated at `14.75-14.85 t/s` on longer checks and matched
  the old Q8 decode path exactly on a 64-token output comparison.
- Rechecking the V100 register cap after rows128/cuBLAS-A did not improve the
  default. `maxrregcount=60` was flat at `14.83 t/s`; `maxrregcount=72` is
  invalid with the 1024-thread rows128 down kernel because it exceeds launch
  resources.
- Rechecking F16-cache reserve after rows128/cuBLAS-A also did not produce a
  defaultable win. `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=3072` measured
  `14.84 t/s`, `2048` produced one `14.91 t/s` run but was not established as
  better than noise, and lower `1536`/`1024` checks measured `14.83` and
  `14.82 t/s`.

The latest dev profile still points to routed MoE as the largest remaining
single-node target, with attention output also significant. The main 200-token
benchmark now reaches 15 tok/s, but further headroom would likely come from MoE
gate/up, attention output B, or compressor/cache-update work.

## 2026-05-16 Indexer Check

`DS4_METAL_INDEXER_STAGE_PROFILE=1` was rechecked after the MoE/output sweeps.
The script now aggregates `metal indexer stage` lines when they appear, but the
standard short decode profile emits none because the top-k path only starts
after the ratio-4 compressed row count exceeds 512. The layer-level profile still
shows `compressor_indexer` around `0.151-0.152 ms/layer`, but for the current
benchmark that should be read as compressor/update overhead, not top-k overhead.

After the V100 register cap, 128-row MoE down, one-token cuBLAS-A, and
512-thread gate/up changes, the next priority order is:

1. further fused MoE gate/up+midq work, plus any down shape that beats rows128,
2. attention-output B or full attention-output projection pipeline work,
3. compressor/cache-update fusion as a secondary route,
4. long-context indexer top-k work only if long-context decode becomes the main
   target.

## 2026-05-16 Post-Upstream 18 tok/s Probes

After cherry-picking selected upstream runtime/server commits from `origin/main`,
the 4-GPU direct benchmark still measured `15.06 t/s` on ai-smil1 and
`15.05 t/s` on ai-smil2.

`git cherry -v ppc64le origin/main` was rechecked after the later performance
probes. Several useful runtime/server subjects still show as patch-positive only
because the local branch carries conflict-adjusted cherry-pick equivalents with
different SHAs. Subject checks found local equivalents for the runtime/server
commits we wanted; the remaining main-only work is eval/model-card/imatrix or
README-oriented and remains intentionally skipped for the ppc64le runtime path.

Three quick routes did not justify default changes:

- `DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_B=1` was tested as a temporary B-only
  attention-output opt-out. Output matched the default, but the short check was
  slightly slower (`15.24` vs `15.27 t/s`), so the generic cached-F16/cuBLAS B
  path should stay enabled.
- A temporary 1024-thread fused MoE gate/up+midq kernel matched output but
  regressed the short check (`15.10` vs `15.25 t/s`). The 512-thread kernel
  remains the V100 sweet spot.
- A temporary 768-thread fused MoE gate/up+midq kernel was also rejected. The
  short check was flat (`14.63` vs `14.62 t/s`) and the same-binary 64-token
  output comparison did not match (`cmp` exit 1), so it was reverted.
- Rechecking the V100 ptxas cap after the 512-thread kernel and upstream picks
  measured `15.13 t/s` with `maxrregcount=60` and `15.10 t/s` with the forced
  default `64` build. This is still noise-level; keep the safer 64-register
  default unless a longer cross-node check proves otherwise.

The 18 tok/s target still needs a structural reduction in layer-loop latency,
not another small flag. The most plausible remaining source-level routes are a
new MoE gate/up or down design, an attention-output A/B/HC pipeline fusion that
does not repeat the older regressing fusion, or compressor/cache-update kernel
fusion.

Additional topology and projection checks also did not produce a route:

- The node topology is pairwise NVLink: GPU0<->GPU1 and GPU2<->GPU3 are `NV3`,
  while all cross-pair GPU paths are `SYS`. The default
  `DS4_CUDA_DEVICES=0,1,2,3` already keeps the 43 layers in contiguous chunks,
  so the layer handoffs are NVLink, cross-pair, NVLink. Alternate device orders
  did not beat the default: `1,0,2,3` measured `15.06 t/s`, `0,1,3,2`
  measured `15.02 t/s`, `1,0,3,2` measured `14.99 t/s`, and `2,3,0,1`
  measured `14.81 t/s`.
- Three-GPU runs on `0,1,2` and `0,1,3` caused model-cache arena allocation
  failures and then generated only `7.10 t/s` and `6.45 t/s`. The fourth GPU is
  needed for resident weights/cache headroom; removing a cross-pair handoff does
  not pay for losing that memory.
- `DS4_CUDA_SYNC_DEVICE_SWITCH=1` was flat against the default event-based
  cross-device handoff in a direct ai-smil1 200-token check (`14.60 t/s` for
  both modes). The device-switch synchronization mode should stay on the default
  event path.
- A small ratio-4 compressor shift cleanup removed redundant stores back into
  the already-current half of the compressor state. It matched output exactly
  against the previous binary on a 64-token check and measured `14.62 t/s` on a
  200-token ai-smil1 run, so it is kept as a cleanup but not counted as a
  meaningful route toward 18 tok/s.
- A temporary `DS4_CUDA_COMPRESSOR_EMIT_FUSED=1` one-row emit kernel fused
  compressor pool, RMS+weight, and RoPE. It matched default output exactly on
  64-token and 200-token checks, but the 200-token direct run only moved from
  `14.64` to `14.66 t/s`. That is noise-level, so the source was reverted.
- Existing MoE down-kernel toggles were rejected. `DS4_CUDA_MOE_DIRECT_DOWN_SUM6=1`,
  `DS4_CUDA_MOE_DOWN_ROWS32=1`, `DS4_CUDA_MOE_DOWN_ROWS64=1`, and
  `DS4_CUDA_MOE_DOWN_ROWS96=1` all matched the default output, but measured
  `13.31`, `13.38`, `13.99`, and `14.04 t/s` respectively on 64-token checks.
  The default 128-row decode down path remains best among the existing knobs.
- `DS4_CUDA_MOE_NO_DECODE_LUT_GATE=1` matched output exactly but fell to
  `9.88 t/s`, so the decode LUT gate path is required.
- `DS4_CUDA_FUSE_Q_NORM_ROPE=1` was tested as a temporary source change using
  the existing fused Q head-RMS+RoPE CUDA kernel. It matched output exactly and
  shaved only about `0.004 ms/layer` from `q_path`; the 200-token direct run
  regressed from `14.64` to `14.58 t/s`, so the source was reverted.
- Additional attention-output fallbacks were rejected. `DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A=1`
  preserved output exactly but measured only `14.41 t/s`, and
  `DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1` changed output while measuring
  `14.40 t/s`. Keep the cached F16/cuBLAS A projection and attention-output F16
  cache enabled.
- `DS4_CUDA_Q8_F32_LARGE=1` was rejected for the attention-output B-sized path:
  it changed output, caused a CUDA model arena allocation failure for `moe_down`,
  and measured only `12.31 t/s`.
- `DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ=1` measured `15.03 t/s`, below the
  current default band, so compressor pair projection should stay enabled.
- `DS4_CUDA_F16_PAIR_MATMUL=1` measured `11.76 t/s`, far below the current
  default. The existing two one-token F16/cuBLAS calls are faster than the
  ordered-chunks F16 pair kernel on V100.
- A temporary qwarp64 MoE gate/up reduction change started the Q8 max reduction
  at `CUDA_QK_K >> 1` instead of `blockDim.x >> 1`, skipping the empty
  256-thread half of the 512-thread block. It preserved the deterministic output
  hashes for 64-token and 200-token checks, but measured only `14.65 t/s` and
  `14.55 t/s`, so the source was reverted.
- Existing compressor/indexer fallbacks were spot-checked and rejected.
  `DS4_CUDA_NO_INDEXER_DIRECT_ONE=1`, `DS4_CUDA_NO_TOPK1024=1`, and
  `DS4_CUDA_NO_TOPK2048=1` all preserved the deterministic 64-token output hash,
  but measured only `14.61`, `14.62`, and `14.61 t/s` respectively.
- Moving the CUDA primary device for shared scratch allocation did not help.
  With the normal layer split still set to `DS4_CUDA_DEVICES=0,1,2,3`,
  `DS4_CUDA_DEVICE=1` preserved the deterministic 64-token output hash but
  measured `14.48 t/s`, and `DS4_CUDA_DEVICE=2` measured `14.36 t/s`.
  A real per-device scratch-buffer design is still a possible larger route, but
  simply changing the primary device is not.
- A per-layer profile showed that layers 22-42 are much heavier than layers
  0-21, especially in routed MoE and compressor/indexer work. A cost-balanced
  contiguous split would be roughly 15/11/8/9 layers instead of the default
  11/11/10/11, but current tensor-split tuning is limited by VRAM/cache
  pressure. `DS4_CUDA_TENSOR_SPLIT=15,11,8,9` fell to `7.23 t/s`,
  `12,12,9,10` fell to `13.60 t/s`, and `12,11,10,10` fell to `13.34 t/s`;
  all changed the deterministic 64-token output hash because cache coverage
  changed. Keep the default `8.2,8.2,8.2,8.2` split.

## 2026-05-16 Luce Megakernel Takeaways

Reference: <https://github.com/Luce-Org/lucebox-hub/tree/main/megakernel>

The Luce megakernel work is useful as direction, but not as a directly portable
implementation. Their kernel is a hand-written, batch-1, single-model,
single-architecture path for Qwen 3.5-0.8B, using BF16/FP16 weights and no GGUF
quantization. The README also notes that the approach is a research
proof-of-concept, and that benefits shrink as model size grows and compute
begins to dominate. DS4 on this setup is a much larger 43-layer routed-MoE,
quantized, multi-GPU decode path, so a whole-model megakernel is not a realistic
near-term route.

The practical lessons for this V100 DS4 branch are still relevant:

1. Minimize CPU launch and layer-boundary overhead where the profile shows many
   small pure-kernel steps. We already tried cuBLAS graph capture and it failed
   on this CUDA/cuBLAS stack, but pure-kernel islands remain plausible.
2. Prefer scoped persistent/fused islands over a full forward-pass megakernel.
   Candidate islands are the compressor/update path, attention-output A/B/HC
   tail if it can avoid the older regressing fusion shape, and parts of routed
   MoE that are not already fused.
3. Treat register pressure as a first-class constraint. This matches our V100
   results: `-Xptxas=-maxrregcount=64` helped, 1024-thread gate/up regressed,
   and the 128-row MoE down shape is already near a launch-resource boundary.
4. Avoid grid-wide synchronization inside tight recurrence/token loops. If a
   future persistent island uses cooperative groups, synchronize at coarse
   phase boundaries and prove it with correctness comparisons plus watchdog-safe
   tests.

For the 18 tok/s target, the takeaway is that launch overhead may still matter,
but the most likely useful version is a small number of DS4-specific persistent
or fused islands, not a generic vLLM-style scheduler and not a monolithic
whole-model rewrite.
