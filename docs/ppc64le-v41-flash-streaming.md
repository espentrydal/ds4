# DeepSeek V4.1 Flash on one V100 (AC922, SSD streaming)

Measured on two IBM AC922 nodes (POWER9, 4x V100-SXM2-32GB, 631 GiB RAM,
CUDA 12.4). Model: `DeepSeek-V4.1-Flash-Q2.gguf` (341 GiB; 152 GiB main
weights, 189 GiB Engram tables). One GPU holds the dense weights and a cache of
routed experts; the rest of the file is read from the page cache.

## Run it

```sh
source /opt/rh/gcc-toolset-13/enable && make cuda-v100 -j40
DS4_CTX=65536 scripts/server-ds41-flash-v100-1gpu.sh
```

Warm the GGUF into the page cache first (`cat file > /dev/null`, about 2.5 min).
The launcher sets:

| Setting | Value | Why |
|---|---|---|
| `DS4_CUDA_NO_DIRECT_IO`, `DS4_CUDA_KEEP_MODEL_PAGES` | 1 | Direct I/O or dropping pages forces SATA reads: 0.2 t/s. |
| `DS4_CUDA_SSD_PAGEABLE_DMA` | 1 | POWER9 NVLink reads the mmap through host page tables (ATS). Scalar expert loads DMA straight from it: 67 GB/s versus 11 GB/s through the pinned staging ring. Ignored where the GPU lacks host page tables. Batched prefill keeps staging, which is faster there. |
| `DS4_CUDA_SSD_PREFAULT_MODEL` | 1 | Touches the main mapping once (about 2 s) so the first requests do not fault page by page. Without it the same run decodes at 7.7 instead of 16.1 t/s. |
| `DS4_CUDA_SSD_CACHE_RESERVE_MB` | 1024 | Default VRAM reserve for the expert cache is 8 GiB. |
| `--ssd-streaming-cache-experts` | 448 | See memory below. |
| `DS4_METAL_DISABLE_V41_8K_CHUNK` | 1 | Prefill sweeps of 4096 rows. The 8192-row default ran out of memory even with 384 slots at 32k context. |
| `--kv-cache-cold-max-tokens` | the context size | The 30000 default saves no prefix snapshot for longer first prompts, so a resent prompt prefilled again from token 0. |

`DS4_CUDA_SSD_STREAM_STATS=1` prints batch and selected-expert upload totals at
exit.

## Measurements

All with the Q2 model, one GPU, warm page cache, temperature 0.

**Short prompt (8k context):** 16.1 generated tokens/s, 6.3 prefill tokens/s.
The expert cache reserve change alone moved the earlier 5.2 t/s to 7.0 t/s;
direct mmap uploads and prefaulting did the rest.

**Long context** (non-repeating C source as prompt, fresh prefill of the whole
prompt, then 128 teacher-forced decode tokens):

| Prompt | Schedule | Prefill | Decode |
|---|---|---:|---:|
| 32k | 4096 rows + decoder suffix | 233 tok/s (about 140 s) | |
| 32k | 4096 rows, no suffix | 120 tok/s | |
| 64k | 4096 rows + decoder suffix, 512 slots | 238 tok/s (about 275 s) | 12.86 tok/s |
| 64k | same, 448 slots | 238 tok/s | 12.69 tok/s |
| 64k | 2048 rows, no suffix, 512 slots | 111 tok/s (about 590 s) | 12.89 tok/s |

Decode at 64k is about 5% slower than at 8k (12.94 versus 13.55 tok/s in an
earlier incremental sweep, which also gave 13.3 tok/s at 32k). Peak VRAM sampled
once per second at 64k context: 31419 MiB with 512 slots, 30811 MiB with 448,
29872 MiB for the 2048-row schedule. The card exposes 32510 MiB, so 512 leaves
about 1.1 GiB and 448 about 1.7 GiB; 448 is the launcher default. Logits at 64k are bit-identical for 448
and 512 slots: the cache size does not change results.

**Through the HTTP server** (`DS4_CTX=65536`, defaults above, one slot):

| Request | Prompt tokens | Time to first token |
|---|---:|---:|
| First, 33.8k tokens | 33797 | 162 s (209 tok/s) |
| Identical resend (prefix snapshot at 32768 restored) | 33797 | 21 s, same output |
| Chat follow-up turn (live session reused) | 33859 | 2.7 s |
| Second long prompt sharing the first 32768 tokens | 58967 | 140 s for the 26k new tokens |

Decode in the server was 13.4-13.5 t/s at 34k. Peak VRAM 30857 MiB. A cold
59k first prompt was not timed through HTTP; the offline 64k prefill above
(about 275 s) is the estimate.

## Numerical behavior

Different prefill schedules do not give bit-identical logits, and no schedule
is a unique reference. The cause was traced layer by layer:

- Batched kernels (F16 cuBLAS GEMM for the HC mixers, Q8 matrix products for
  the K/V projections, attention) sum in an order that depends on the number
  of rows in the sweep.
- V4.1 rounds activations to BF16 after most operations, so a one-ulp
  difference can flip a rounding downstream and spread: at layer 0 two
  schedules differ in one value in each of 25 of 2048 `kv` rows, and by layer 2
  about 12% of the published compressed KV rows differ.
- Even a prefill of 1 token per sweep differs from one of 8 tokens per sweep
  at every vocabulary entry.

What was held fixed: the same schedule gives the same result. Fresh and
incrementally appended 64k prompts were bit-identical on the 2048-row schedule,
where sweeps start on the same row boundaries, and the expert cache size never
changed a result.

How far the schedules are from each other, as KL divergence of the next-token
distribution against a prefill of one token per sweep (non-repeating C source,
single sample each; argmax matched in every run, 16-20 of the top 20 tokens
overlap):

| Schedule | KL at 16k | KL at 32k | KL at 64k |
|---|---:|---:|---:|
| 8 tokens per sweep | 7.4e-4 | | |
| 2048 rows, no suffix | 2.9e-3 | 5.5e-3 | 4.4e-6 |
| 4096 rows, no suffix | 1.6e-3 | 6.2e-3 | |
| 4096 rows + suffix | 1.6e-3 | 6.1e-3 | 1.4e-5 |
| 2048 rows + suffix | 8.5e-3 | | |

The decoder-suffix shortcut (re-running only the last 2541 tokens through the
decoder layers) adds almost nothing at 4096 rows (KL 8e-6 against 4096 rows
without it at 32k) and roughly doubles prefill speed. The 2048-row suffix
schedule was the least close at 16k, so it is not used.

A per-token HC projection (`DS4_CUDA_V41_HC_PER_TOKEN`) made the mixer values
independent of the sweep size and moved layer 0 and 1 to bit identity, but it
cost 9% of prefill and did not make later layers agree, so it was not kept.

The spread between samples is larger than the spread between schedules, so
these numbers show that the fast schedule stays in the same range, not that it
is closer or farther than the others. The 1-token run takes 52 minutes at 64k.

Debug switches (off by default):

- `DS4_METAL_GRAPH_DUMP_PREFIX` and `DS4_METAL_GRAPH_DUMP_POS` now also dump the
  sweep rows of the selected position for the V4.1 stages (`v41_*`), including
  HC mixers, projections, indexer inputs and the published caches.
- `DS4_V41_TRACE_WHOLE_BATCH` dumps every row of that sweep.
- `DS4_V41_PREFILL_LIMIT=N` caps the sweep size; 1 gives a token-by-token
  prefill (about 21 tok/s) to compare schedules against.

## Two ranks (experimental)

`DS4_CUDA_V41_TP_REPLICATE_ATTN=1 DS4_CUDA_V41_TP_ORDERED_SUM=1` with two
streaming ranks, one GPU each, reproduced the single-GPU outputs in a 16-step
check and in 6800 intermediate tensors byte for byte, at 17.85 generated and
15.72 prefill tokens/s on an NVLink pair. `DS4_CUDA_V41_TP_GATHER_ATTN=1` is
also exact but slower. Two independent pairs on one node served two requests at
a combined 33.9 t/s; that is throughput, not a faster single request. Not
validated at long context, with more than two ranks, or across nodes (the
InfiniBand link currently negotiates 10 Gb/s).

## Check a change

Short, exact (about 1 minute once the cache is warm):

```sh
./ds4 --cuda -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ssd-streaming --ctx 8192 \
  --nothink --temp 0 -n 16 --dump-logprobs new.json -p 'Explain how a hash table handles collisions.'
```

Compare `new.json` with a saved run from the previous build; the 16 steps and
their top-20 logits should be identical.

Long context (about 10 min at 64k):

```sh
DS4_METAL_DISABLE_V41_8K_CHUNK=1 DS4_CUDA_SSD_CACHE_RESERVE_MB=1024 \
DS4_CUDA_SSD_PAGEABLE_DMA=1 DS4_CUDA_SSD_PREFAULT_MODEL=1 \
./ds4-bench --cuda -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ssd-streaming \
  --ssd-streaming-cache-experts 448 --prompt-file prompt.txt \
  --ctx-start 65536 --ctx-max 65536 --ctx-alloc 66049 \
  --gen-tokens 128 --teacher-forced-decode --dump-frontier-logits-dir out
```

Expect about 238 prefill tokens/s and 12.7 decode tokens/s. Compare
`out/frontier_065536.logits.json` between builds with the same flags (identical
is expected); against a different schedule, expect the KL values above, not
equality.
