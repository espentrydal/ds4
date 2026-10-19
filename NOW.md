# NOW: DeepSeek V4.1 Flash on AC922 (ppc64le-v41)

State as of 2026-10-19. Details: `docs/ppc64le-v41-flash-streaming.md`.

## Running

- ai-smil2: `ds41-ep.service` (systemd user unit, started, not enabled at boot).
  Port 8082, 64k context, all four GPUs, expert-parallel decode.
  Script `scripts/server-ds41-flash-v100-ep.sh`, KV snapshots in `~/ds4.1-kv`.
- Stop it before using GPUs for anything else: `systemctl --user stop ds41-ep`.
  The unit conflicts with `ds4-server.service` and `qwen-gpu01.service`.
- Logs: `journalctl -b -f _SYSTEMD_USER_UNIT=ds41-ep.service`. First start prefaults the 341 GiB
  model into RAM (several minutes before the port opens).
- Stable V4 in `~/ds4` on both nodes is untouched. ai-smil1 is idle.
- Enable at boot (not done): `systemctl --user enable ds41-ep`.

## Numbers (graphs + expert-parallel, 3 helper GPUs, 1800 slots each)

| Context | Prefill t/s | Decode t/s |
|---|---:|---:|
| 8k | | 21-22.5 |
| 32k | 396 | 20.6 (22 steady) |
| 64k | 404 | 19.6 (20.9 steady) |

Logprobs are bit-identical to the one-GPU eager path at 8k, 32k and 64k.
One GPU without these features: 14-15 t/s.

## Why not 25 t/s yet

A decoded layer is a serial chain of about 10 hops (leader dense work, copies to
helpers, expert lookup, compute, copy back, sum) of 15-30 us each, 40 layers:
about 1.1 ms per layer. Uploads are no longer the limit (hit rate 88-93%;
2400 and 3000 slots gave no gain). Each single fix is worth 1-3%.

## Next, in order of value

1. **Speculative decoding** (only route clearly past 25 t/s). No drafter exists
   for V4.1 with SSD streaming (DSpark/MTP files on the nodes are for V4 Flash).
   n-gram / prompt-lookup drafting needs no model and helps repetitive text
   (code, tool output); batched verification is not bit-identical to normal
   decode (different summation order), so ship it opt-in and say so.
2. **Expert kernel** `moe_gate_up_mid_decode_lut_qwarp32_kernel` and
   `moe_down_sum_qwarp32_kernel`: 40-140 us for one or two experts that need
   about 8 us of memory traffic (about 15% efficient). Must keep the exact
   summation order. Est. 3-5 ms/token.
3. **Dispatch**: nine small peer copies per layer cost 15 us of host time each.
   Packing x, ids and weights into one buffer saves 30-50 us per layer at best.
   An earlier try (host-read ids) was exact but no faster:
   `notes/ds41-ep-host-ids-and-profile.patch` (outside the repo).
4. **Chat latency**: short tails after a snapshot prefill at about 5.5 tok/s;
   snapshots align at 2048; thinking/tool turns miss on token mismatch. Run the
   cold 58k test and a realistic tool/thinking client against the service.
5. **Prefill**: shared/routed FFN is 51% of a 16k prefill; token-tile it as was
   done for attention output (exact check at 16k and 32k).
6. **Operations**: enable at boot; health check/restart policy; two-node use
   (a second request on ai-smil1; one request across 10 Gb/s IB does not help).

## Known limits

- Expert-parallel needs all four GPUs of one node for one request. The earlier
  two-pair setup has higher aggregate throughput (33.9 t/s) but lower
  single-request speed.
- Helper failure falls back to one GPU for the rest of the process (tested with
  injected failures, `DS4_V41_EP_TEST_FAIL_JOB`); restart the service to regain
  expert-parallel.
- Cache sizing: the leader keeps 448 slots (headroom for 64k prefill); do not
  raise it. Helpers use 1800 slots with a 2 GiB reserve.
- Diagnostics: `DS4_V41_EP_*`, `DS4_V41_STREAM_GRAPHS`, `DS4_V41_NO_SHARED_SIDE`,
  `DS4_CUDA_TP_GATE_STATS`.
