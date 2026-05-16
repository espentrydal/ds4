# ppc64le V100 Performance Notes

This setup is one node with four V100 32 GB GPUs. The measured fast decode
configuration is:

```sh
DS4_CUDA_SPLIT_OUTPUT_HEAD=1
DS4_CUDA_DEVICES=0,1,2,3
DS4_CUDA_TENSOR_SPLIT=8.2,8.2,8.2,8.2
DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=512
DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION=1
```

Use `scripts/server-ds4-flash-v100-1node.sh` for OpenAI/Anthropic-compatible
serving, or `scripts/profile-ds4-flash-v100-1node.sh` for a reproducible decode
profile run.

Production serving now defaults to `DS4_CTX=131072` to reduce VRAM pressure and
protect decode speed. The historical measurements below were mostly taken at
262K context unless otherwise noted; use them as kernel/runtime comparisons, not
as a claim that 262K is the preferred production context.

Service-path check on 2026-05-15, after retiring the stale llama.cpp
`rpc-server` process, measured warm 200-token non-thinking chat requests at
12.5-12.6 tok/s wall-clock. The server-side decode log reported 12.98-13.05
tok/s. The stale RPC process had been reserving about 360 MiB on each GPU, which
was enough to reduce DS4 CUDA cache headroom.

A later 2026-05-15 dev sweep toward 15 tok/s tested lower F16-cache reserves,
skewed tensor split, direct MoE down+sum, disabled attention-output F16 cache,
explicit fast MoE SILU, and a shared-`midq` decode down kernel. None improved
the service path beyond the roughly 13 tok/s server-side result; several
regressed.

The first simple route that did improve current direct decode was disabling the
shared-down/HC fusion with `DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION=1`. Same
prompt 200-token checks measured `12.64 -> 13.20 t/s` on ai-smil1 and
`12.55 -> 13.09 t/s` on ai-smil2. The stage profile shows why: default fused
shared-down/HC measured about `shared_down=0.175 ms/layer` plus
`ffn_hc_post=0.008 ms/layer`, while the opt-out measured
`shared_down=0.068 ms/layer` plus `ffn_hc_post=0.064 ms/layer`. The separate
path is faster overall.

The same setting was verified through `ds4-server.service` on ai-smil2. The
systemd process environment included `DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION=1`;
a warm 128K-context 200-token request logged `13.28 t/s` for the first chunk and
`12.99 t/s` average over 200 generated tokens.

The adjacent shared gate/up/SwiGLU fusion should stay enabled. Testing
`DS4_METAL_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION=1` with the fast shared-down
setting produced only noise-level direct throughput (`13.14 t/s` on ai-smil2)
and worsened the profiled `shared_gate_up` bucket from about `0.100 ms/layer`
to `0.114 ms/layer`.

The narrower `DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR=1` check also should not be a
default. It left direct throughput at the ai-smil2 baseline (`13.09 t/s`) and
similarly worsened the profiled `shared_gate_up` bucket to `0.115 ms/layer`.

A 2026-05-16 follow-up added `DS4_OUTPUT_HEAD_PROFILE=1` to isolate the final
output head. This profiler synchronizes before the output head and between
substeps, so it is for diagnosis rather than speed measurement. A later
`DS4_CUDA_SPLIT_OUTPUT_PROFILE=1` check showed that the earlier `~67 ms/token`
output-logits average was dominated by the first lazy output-weight cache during
prefill. Warm generated-token split output is only about `1.35 ms`, so output
logits are not a meaningful route to 15 tok/s.

The same follow-up checked MoE row-span variants. `DS4_CUDA_MOE_DOWN_ROW1024=1`
was flat (`down=0.296 ms`, `total=0.587 ms`), `DS4_CUDA_MOE_DOWN_ROW512=1`
regressed (`down=0.300 ms`, `total=0.594 ms`), and gate row 512/2048 variants
left `gateup` around `0.261 ms`. The row-span toggles are not a route to 15
tok/s.

A later decode-only MoE down experiment tried a 5-row x 6-slot direct-sum
kernel, guarded during the test as `DS4_CUDA_MOE_DIRECT_DOWN_SUM6_ROWS5=1`.
It preserved expert-slot parallelism while avoiding the intermediate
`6 * out_dim` down buffer. It still regressed: the explicit sum stage dropped
from about `0.009 ms/layer` to `0.002 ms/layer`, but `down` rose from
`0.299 ms/layer` to `0.320 ms/layer`, increasing total MoE time. The temporary
kernel was reverted.

`nvprof` on a short decode run confirmed the active warm-decode MoE kernels:
`moe_down_qwarp32_kernel` averaged about `293 us/layer`, and
`moe_gate_up_midq_decode_lut_qwarp32_kernel` averaged about `256 us/layer`.
The larger `moe_*expert_tile8_row32*` kernels in the same summary are prefill,
not generated-token decode. `ncu` could not collect hardware counters because
GPU performance counters are permission-blocked (`ERR_NVGPUCTRPERM`). A
temporary `__forceinline__` test on the Q2 down dot helpers was flat to slightly
worse and was reverted.

## Current Profile

Profile command used the same fast CUDA split plus:

```sh
DS4_METAL_GRAPH_TOKEN_PROFILE=1
DS4_METAL_DECODE_STAGE_PROFILE=1
DS4_CUDA_MOE_PROFILE=1
```

Current short profile run result with the fast shared-down/HC opt-out:

```text
ds4: prefill: 0.15 t/s, generation: 10.76 t/s
```

The profile is slower than normal generation because the stage profiler
synchronizes frequently. The current normal 200-token generation benchmark for
this build is:

```text
ds4: prefill: 0.15 t/s, generation: 13.20 t/s
```

Later decode-path tuning on the same model and CUDA split measured:

```text
MTP disabled baseline:                 generation: 11.03 t/s
MTP draft=1:                           generation: 10.98 t/s
MTP draft=2:                           generation:  4.71 t/s
generic MoE down path default:          generation: 11.20 t/s
reference attention-output path default: generation: 11.48 t/s
fused attention-output opt-in:          generation: 11.24 t/s
```

The optional MTP GGUF is usable with `--mtp`, but on this one-node V100 setup it
did not produce a useful decode-speed win. `--mtp-draft 2` also triggered CUDA
allocation failures for the MTP MoE ranges and should not be used as a default.

The current attention-output default uses the reference HC expansion path.
Set `DS4_METAL_ENABLE_ATTN_OUT_HC_FUSION=1` only for comparison with the older
fused path.

The tested `attn_q_b` cache variant
`DS4_CUDA_ATTN_Q_B_F32_CACHE=1 DS4_CUDA_NO_ATTN_Q_B_F16_CACHE=1` was flat with
the new attention-output default: both the default and q-path variant measured
`11.62 t/s` in the 48-token check.

The decode q/k/v path now pairs the `attn_q_a` and `attn_kv` Q8 projections
that share the same normalized input. The old two-projection path is available
with `DS4_CUDA_DISABLE_QKV_PAIR_PROJ=1`. In the 96-token check, paired measured
`11.54 t/s` versus `11.50 t/s` for the old path. The synchronized profile
showed `q_path` moving from `0.355 ms/layer` to `0.334 ms/layer`.

The QKV pair projection was rechecked after the shared-down/HC opt-out became
the fast default. It is still required for speed: `DS4_CUDA_DISABLE_QKV_PAIR_PROJ=1`
raised `q_path` from about `0.186 ms/layer` to `0.253 ms/layer`, and direct
200-token decode on ai-smil2 dropped to `12.89 t/s`.

F16 cache reserve tuning was also rechecked after the shared-down/HC opt-out.
`DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=3072` measured `13.21 t/s`, and `2048`
measured `13.22 t/s`, while a same-node default rerun measured `13.19 t/s`.
That is within the current run-to-run band, so reserve tuning remains a
low-priority route rather than a production-default change.

One-token decode now uses cached F16/cuBLAS for eligible Q8 projections by
default, with `DS4_CUDA_NO_Q8_F16_GEMV=1` as the opt-out. This moved the
96-token check from `11.48 t/s` with the opt-out to `12.58 t/s` by cutting the
profiled `q_path` to `0.187 ms/layer`.

Follow-up MoE checks did not find a better decode mode: write-gate/up,
direct-sum6, no-LUT gate/up, and a temporary atomic-down experiment all
regressed. A 5-row x 6-slot direct-sum down kernel also regressed by trading a
smaller sum stage for a larger down stage. Attention-output follow-ups were
also flat: one-token cuBLAS for `attn_output_a` and opt-in F16 output-head
caching did not produce a meaningful speedup.

A later recheck of `DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1` showed why this
should not become a production default. The synchronized profile improved
`attn_output` from `0.279 ms/layer` to `0.269 ms/layer`, but same-node direct
200-token generation moved the wrong way, from `12.64 t/s` default to
`12.55 t/s` with attention-output F16 caching disabled.

`DS4_CUDA_ATTENTION_OUTPUT_PROFILE=1` is available for opt-in CUDA event timing
inside the attention-output projection. On warm decode it split the default
path almost evenly between `attention_output_a` and `attention_output_b`:
`a=0.142 ms/layer`, `b=0.137 ms/layer`, `total=0.280 ms/layer`. With
`DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1`, CUDA event timing dropped to
`a=0.140`, `b=0.115`, `total=0.256 ms/layer`, but the synchronized stage and
direct generation did not improve. Treat this profiler as a routing diagnostic,
not as a production speed measurement.

The older fused attention-output/HC path was also rechecked with
`DS4_METAL_ENABLE_ATTN_OUT_HC_FUSION=1` on the current build. It is still not a
production route: direct 200-token generation fell to `12.19 t/s` from the
current `12.64 t/s` default. The synchronized profile moved `attn_hc_post` down
to `0.009 ms/layer`, but pushed `attn_output` up to `0.372 ms/layer`, so the
combined A/B/HC pipeline regressed.

Output-head follow-up timings from direct CLI runs with `DS4_OUTPUT_HEAD_PROFILE=1`:

```text
first split-output call:  1052.8 ms total, about 263 ms enqueue per device
warm split-output call:      1.35 ms total, about 1.14 ms waiting on dev0
output F16 cache:        generation 13.29 t/s, cold-inclusive logits 69.181 ms/token
half-warp output Q8:     generation 13.39 t/s, cold-inclusive logits 67.024 ms/token
full-block DP4A output:  generation 13.36 t/s, cold-inclusive logits 67.092 ms/token
```

The F16 output cache run used `DS4_CUDA_OUTPUT_F16_CACHE=1` and
`DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=512`; logs confirmed four cached
`252.50 MiB` output split segments. It still regressed, so this is not a
promising production default. A temporary half-warp output Q8 kernel was flat
within measurement noise and was reverted. A temporary full-block DP4A output
Q8 specialization was also flat and was reverted.

The routed MoE decode path now fuses the gate/up result directly into Q8_K
`midq` blocks, avoiding the separate global `mid` materialization and quantize
kernel. `DS4_CUDA_MOE_NO_FUSE_MIDQ=1` restores the older path. The 96-token
check measured `12.76 t/s` with the fused default versus `12.47 t/s` with the
opt-out. A synchronized profile moved `routed_moe` from `0.709 ms/layer` to
`0.662 ms/layer`, with MoE total moving from `0.632 ms/layer` to
`0.586 ms/layer`. Other MoE kernel experiments, including constant-memory LUT
lookup and half/full-warp down kernels, regressed.

Decode stage totals from the synchronized profile:

```text
routed_moe          227.840 ms total, 0.662 ms avg
attn_output          95.423 ms total, 0.277 ms avg
q_path               64.504 ms total, 0.188 ms avg
shared_down          60.081 ms total, 0.175 ms avg
compressor_indexer   52.271 ms total, 0.152 ms avg
ffn_hc_pre           42.125 ms total, 0.122 ms avg
attn_hc_pre          41.613 ms total, 0.121 ms avg
shared_gate_up       33.676 ms total, 0.098 ms avg
attention            33.182 ms total, 0.096 ms avg
router               23.302 ms total, 0.068 ms avg
attn_hc_post         22.093 ms total, 0.064 ms avg
kv_path              13.998 ms total, 0.041 ms avg
```

MoE subtotals:

```text
total  201.649 ms total, 0.586 ms avg
down   102.571 ms total, 0.298 ms avg
gateup  88.870 ms total, 0.258 ms avg
xq       5.748 ms total, 0.017 ms avg
sum      2.964 ms total, 0.009 ms avg
sort     0.683 ms total, 0.002 ms avg
midq     0.650 ms total, 0.002 ms avg
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
