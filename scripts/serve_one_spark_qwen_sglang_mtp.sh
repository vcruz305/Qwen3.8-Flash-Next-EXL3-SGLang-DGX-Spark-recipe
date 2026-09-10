#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next EXL3 on one DGX Spark GB10 via SGLang with the
# EAGLE/MTP draft enabled. Requires the sglang-exl3 plugin fix that loads
# the draft's MoE/linear layers as native EXL3 (SGLANG_EXL3_MTP_EXL3=1,
# the default as of the fix -- see README "MTP fix" section).
#
# speculative_eagle_topk is HARD-CAPPED AT 1 for this architecture
# (Qwen4-Exp QSA MTP raises NotImplementedError above topk=1). Raising
# speculative_num_steps or speculative_num_draft_tokens above the verified
# steps=3/draft_tokens=4 pair crashes the QSA sparse indexer with a CUDA
# device-side assert (out-of-bounds gather) -- the shared MTP sparse-index
# buffer is sized for steps=3 and does not resize with the flag. Do not
# raise these without first fixing QSAMTPSharedSparseIndices sizing in
# sglang's eagle_worker_v2.py / qwen_sparse_attn_backend.py.
#
# Env overrides: same as serve_one_spark_qwen_sglang.sh, plus
#   SPEC_STEPS        (default 3, do not raise -- see above)
#   SPEC_TOPK         (default 1, MUST stay 1 -- see above)
#   SPEC_DRAFT_TOKENS (default 4, do not raise -- see above)
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the local Qwen3.8-Flash-Next-exl3 pack directory}"
PORT="${PORT:-30000}"
CTX_LEN="${CTX_LEN:-4096}"
MEM_FRAC="${MEM_FRAC:-0.95}"
MAX_REQS="${MAX_REQS:-1}"
LOG="${LOG:-./sglang_qwen.log}"
SPEC_STEPS="${SPEC_STEPS:-3}"
SPEC_TOPK="${SPEC_TOPK:-1}"
SPEC_DRAFT_TOKENS="${SPEC_DRAFT_TOKENS:-4}"

if [[ "$SPEC_TOPK" != "1" ]]; then
  echo "REFUSING: speculative_eagle_topk must be 1 for Qwen4-Exp QSA MTP" \
       "(got $SPEC_TOPK) -- see script header" >&2
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/lib/python3.12/dist-packages/nvidia/cu13}"
export PATH="$CUDA_HOME/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:$CUDA_HOME/lib"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SGLANG_EXL3_MOE_KERNEL="${SGLANG_EXL3_MOE_KERNEL:-exllamav3}"
export EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}"
export SGLANG_EXL3_NGRAM="${SGLANG_EXL3_NGRAM:-$MODEL_DIR/ngram_embedding.safetensors}"
# Native-EXL3 draft loading; see README "MTP fix" -- required or the draft
# loads dense random weights and accept rate stays 0.
export SGLANG_EXL3_MTP_EXL3="${SGLANG_EXL3_MTP_EXL3:-1}"

: > "$LOG"

python3 -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --served-model-name Qwen3.8-Flash-Next \
  --host 127.0.0.1 \
  --port "$PORT" \
  --quantization exl3 \
  --load-format exl3_mmap \
  --trust-remote-code \
  --context-length "$CTX_LEN" \
  --mem-fraction-static "$MEM_FRAC" \
  --max-running-requests "$MAX_REQS" \
  --disable-radix-cache \
  --ple-offload-embedding \
  --ple-offload-backend file \
  --ple-offload-dir "${PLE_OFFLOAD_DIR:-./ple-qwen38}" \
  --disable-cuda-graph \
  --linear-attn-backend triton \
  --linear-attn-prefill-backend triton \
  --linear-attn-decode-backend triton \
  --mamba-ssm-dtype float32 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps "$SPEC_STEPS" \
  --speculative-eagle-topk "$SPEC_TOPK" \
  --speculative-num-draft-tokens "$SPEC_DRAFT_TOKENS" \
  --skip-server-warmup \
  >"$LOG" 2>&1 &

echo "$!" > ./sglang_qwen.pid
echo "sglang_pid=$(cat ./sglang_qwen.pid) log=$LOG mtp=steps${SPEC_STEPS}_topk${SPEC_TOPK}_draft${SPEC_DRAFT_TOKENS}"

# Optional: run the memory-guard watchdog alongside (see memory-guard.sh).
# FLOOR_GIB=5 GUARD_GIB=6 POLL_INTERVAL_S=0.5 KILL_MODE=immediate \
#   PIDFILE=./sglang_qwen.pid LOGFILE=./guard.log \
#   setsid bash ./memory-guard.sh </dev/null >/dev/null 2>&1 &
