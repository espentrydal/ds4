#!/usr/bin/env bash
set -euo pipefail

MODEL=${DS4_MODEL:-$HOME/models/deepseek-v4-flash/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}
BIN=${DS4_BIN:-$HOME/ds4/ds4}
OUT=${DS4_PROFILE_OUT:-/tmp/ds4-profile.out}
ERR=${DS4_PROFILE_ERR:-/tmp/ds4-profile.err}
TOKENS=${DS4_PROFILE_TOKENS:-8}
PROMPT=${DS4_PROFILE_PROMPT:-Write a comma-separated list of common English nouns.}

[ -f /opt/rh/gcc-toolset-13/enable ] && source /opt/rh/gcc-toolset-13/enable

export DS4_CUDA_SPLIT_OUTPUT_HEAD=${DS4_CUDA_SPLIT_OUTPUT_HEAD:-1}
export DS4_CUDA_DEVICES=${DS4_CUDA_DEVICES:-0,1,2,3}
export DS4_CUDA_TENSOR_SPLIT=${DS4_CUDA_TENSOR_SPLIT:-8.2,8.2,8.2,8.2}
export DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=${DS4_CUDA_WEIGHT_ARENA_CHUNK_MB:-512}
export DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION=${DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION:-1}
export DS4_METAL_GRAPH_TOKEN_PROFILE=${DS4_METAL_GRAPH_TOKEN_PROFILE:-1}
export DS4_METAL_DECODE_STAGE_PROFILE=${DS4_METAL_DECODE_STAGE_PROFILE:-1}
export DS4_CUDA_MOE_PROFILE=${DS4_CUDA_MOE_PROFILE:-1}

"$BIN" --cuda --model "$MODEL" --ctx 32768 --think --temp 1 --seed 1 -n "$TOKENS" -p "$PROMPT" >"$OUT" 2>"$ERR"

printf "generation\n"
grep "generation:" "$ERR" || true
printf "\ndecode_stage_total_ms count avg_ms\n"
awk '/metal layer stage part=decode/ {for(i=1;i<=NF;i++){if($i ~ /=/){split($i,a,"="); key=a[1]; val=a[2]; if(key!="part"&&key!="layer"&&key!="pos"&&key!="tokens"){sum[key]+=val; cnt[key]++}}}} END{for(k in sum) printf "%s %.3f %d %.3f\n", k, sum[k], cnt[k], sum[k]/cnt[k]}' "$ERR" | sort -k2,2nr
printf "\nmoe_total_ms count avg_ms\n"
awk '/CUDA MoE profile tokens=1/ {for(i=1;i<=NF;i++){if($i ~ /=/){split($i,a,"="); key=a[1]; val=a[2]; if(key!="tokens"&&key!="pairs"){sum[key]+=val; cnt[key]++}}}} END{for(k in sum) printf "%s %.3f %d %.3f\n", k, sum[k], cnt[k], sum[k]/cnt[k]}' "$ERR" | sort -k2,2nr
printf "\nindexer_stage_total_ms count avg_ms\n"
awk '/metal indexer stage/ {for(i=1;i<=NF;i++){if($i ~ /=/){split($i,a,"="); key=a[1]; val=a[2]; if(key!="layer"&&key!="pos"&&key!="tokens"&&key!="comp"){sum[key]+=val; cnt[key]++}}}} END{for(k in sum) printf "%s %.3f %d %.3f\n", k, sum[k], cnt[k], sum[k]/cnt[k]}' "$ERR" | sort -k2,2nr
printf "\nattention_output_cuda_total_ms count avg_ms\n"
awk '/CUDA attention output profile/ {for(i=1;i<=NF;i++){if($i ~ /=/){split($i,a,"="); key=a[1]; val=a[2]; if(key=="a"||key=="b"||key=="total"){sum[key]+=val; cnt[key]++}}}} END{for(k in sum) printf "%s %.3f %d %.3f\n", k, sum[k], cnt[k], sum[k]/cnt[k]}' "$ERR" | sort -k2,2nr
printf "\ntoken_tail\n"
grep "metal graph token" "$ERR" | tail -12 || true
printf "\nalloc_failures\n"
grep "alloc failed" "$ERR" || true
