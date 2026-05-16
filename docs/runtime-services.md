# Runtime Services

Production serving should use systemd user services. Scripts are still kept for
development, one-off overrides, and customized launches.

## Production Defaults

### DS4

- Node: `ai-smil2`
- Service: `ds4-server.service`
- Script: `~/ds4/scripts/server-ds4-flash-v100-1node.sh`
- Port: `8080`
- Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- Context: `131072`
- KV disk budget: `16384 MiB`
- GPU split: `DS4_CUDA_TENSOR_SPLIT=8.2,8.2,8.2,8.2`
- Decode-speed override: `DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION=1`

The DS4 context default is intentionally 128K. DS4 decode speed is sensitive to
VRAM pressure because larger context buffers compete with CUDA-side caches. Use
262K only when the task needs it:

```bash
DS4_CTX=262144 systemctl --user restart ds4-server.service
```

For one-off development runs:

```bash
DS4_CTX=262144 ~/ds4/scripts/server-ds4-flash-v100-1node.sh
```

Do not run the deprecated llama.cpp `rpc-server` beside production DS4. On
2026-05-15 it was still using about 360 MiB per GPU; after stopping it and
restarting `ds4-server.service`, warm 200-token non-thinking chat requests
improved from about 11.5-11.7 tok/s wall-clock to 12.5-12.6 tok/s wall-clock,
with server-side decode logging 12.98-13.05 tok/s.

On 2026-05-16, after making
`DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION=1` the DS4 script default, the systemd
environment confirmed the override and a warm 200-token request at 128K context
logged `13.28 t/s` for the first 50-token chunk and `12.99 t/s` average over
200 tokens. The direct 32K CLI benchmark remains faster at about `13.20 t/s`.

Later on 2026-05-16, after the 64-row MoE decode down kernel became the default,
a warm 200-token `ds4-server.service` request at 128K context logged
`14.30 t/s` for the first 50-token chunk and `13.81 t/s` average over 200
tokens. The direct 32K CLI benchmark measured about `14.04-14.09 t/s`.

### Qwen

- Node: `ai-smil1`
- Primary service: `qwen-gpu01.service`
- Optional second service: `qwen-gpu23.service`
- Script: `~/scripts/server-qwen-gpu01.sh`
- Primary port: `8080`
- Optional second port: `8081`
- Context: `262144`

Qwen uses 256K context by default because it has enough speed headroom for
convenience. The older comments that called this "1M ctx" were stale; the actual
runtime argument is 256K.

## Deprecated Runtime

The old llama.cpp DeepSeek/DSV4 services and scripts are retired. They are kept
only as historical references under `~/scripts/deprecated/` and
`~/.config/systemd/user/deprecated/`.

Current DeepSeek production path is the `~/ds4` runtime, not the older
`~/llama.cpp-deepseek-v4-flash` runtime.
