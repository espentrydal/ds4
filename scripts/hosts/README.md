# Host launcher scripts

Copies of the launcher scripts that live on the model hosts, kept here so their
changes are reviewable and recoverable. The hosts have no chezmoi and their Git
remote authentication is broken, so these are installed by copying:

    scp scripts/hosts/ai-smil1-qwen-gpu01.sh ai-smil1:scripts/server-qwen-gpu01.sh
    scp scripts/hosts/ai-smil1-qwen-gpu23.sh ai-smil1:scripts/server-qwen-gpu23.sh

- `ai-smil1-qwen-gpu01.sh` → `~/scripts/server-qwen-gpu01.sh`, `qwen-gpu01.service`,
  Qwen3.8-27B Q6_K_XL on GPUs 0,1, port 8601.
- `ai-smil1-qwen-gpu23.sh` → `~/scripts/server-qwen-gpu23.sh`, `qwen-gpu23.service`,
  Qwen3.6-35B-A3B Q8 on GPUs 2,3, port 8602.

DeepSeek V4.1 Flash on ai-smil2 uses `scripts/server-ds41-flash-v100-ep.sh`
(port 8801), which is already in this repository.
