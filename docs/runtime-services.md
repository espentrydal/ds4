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
