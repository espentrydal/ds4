# DS4 Notes

This directory is the Hermes (`jarvis`) canonical copy for DS4 operational and
performance notes.

Human/agent-facing copies should also be present on the GPU nodes at:

- `ai-smil1:~/ds4/docs/`
- `ai-smil2:~/ds4/docs/`

The active DeepSeek-V4 Flash runtime branch is `ppc64le` in `~/ds4`.

Files:

- `ppc64le-v100-performance.md`: measured configuration, commits, and profile
  results.
- `ppc64le-v41-flash-streaming.md`: DeepSeek V4.1 Flash on one V100 with SSD
  streaming: launcher settings, 32k-64k measurements, numerical behavior.
- `ppc64le-v100-next-routes.md`: ranked remaining routes for larger speedups.
- `runtime-services.md`: production systemd services and launch-script layout.
