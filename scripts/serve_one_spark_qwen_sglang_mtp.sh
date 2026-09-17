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
# Speed knobs (env, all optional). Defaults match the 2026-09-16 measured
# 34 tok/s path (EAGLE 3/1/4 + ReplaySSM + fused exl3_moe). Safe extras:
#   LINEAR_ATTN_DECODE / LINEAR_ATTN_PREFILL  triton|flashinfer|cute_dsl
#   MAMBA_SSM_DTYPE                           float32|bfloat16
#   MAMBA_SCHEDULER                           extra_buffer|extra_buffer_lazy|no_buffer
#   SPEC_ALGO                                 EAGLE|NEXTN
#   SPEC_ATTN_MODE                            decode|prefill
#   ENABLE_REPLAYSSM                          0|1   (--enable-linear-replayssm-spec)
#   RADIX                                     0|1   (0 = --disable-radix-cache)
#   CUDA_GRAPH                                0|1   (0 required with PLE file offload)
#   CHUNKED_PREFILL                           e.g. 1024|4096|8192
#   SGLANG_EXL3_MOE_KERNEL                    exllamav3|native|auto
#   SGLANG_OPT_MOE_QUANT_ONCE                 1|0
#   EXTRA_ARGS                                extra argv appended to launch_server
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
SPEC_ALGO="${SPEC_ALGO:-EAGLE}"
SPEC_ATTN_MODE="${SPEC_ATTN_MODE:-decode}"
LINEAR_ATTN_BACKEND="${LINEAR_ATTN_BACKEND:-triton}"
LINEAR_ATTN_PREFILL="${LINEAR_ATTN_PREFILL:-$LINEAR_ATTN_BACKEND}"
LINEAR_ATTN_DECODE="${LINEAR_ATTN_DECODE:-$LINEAR_ATTN_BACKEND}"
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-float32}"
MAMBA_SCHEDULER="${MAMBA_SCHEDULER:-}"
ENABLE_REPLAYSSM="${ENABLE_REPLAYSSM:-1}"
RADIX="${RADIX:-0}"
CUDA_GRAPH="${CUDA_GRAPH:-0}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "$SPEC_TOPK" != "1" ]]; then
  echo "REFUSING: speculative_eagle_topk must be 1 for Qwen4-Exp QSA MTP" \
       "(got $SPEC_TOPK) -- see script header" >&2
  exit 1
fi

PYTHON="${PYTHON:-$(command -v python3)}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:${PATH:-/usr/bin:/bin}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/local/lib/python3.12/dist-packages/torch/lib:$CUDA_HOME/lib}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
export SGLANG_EXL3_MOE_KERNEL="${SGLANG_EXL3_MOE_KERNEL:-exllamav3}"
export EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}"
export SGLANG_EXL3_NGRAM="${SGLANG_EXL3_NGRAM:-$MODEL_DIR/ngram_embedding.safetensors}"
export SGLANG_EXL3_MTP_EXL3="${SGLANG_EXL3_MTP_EXL3:-1}"
export SGLANG_OPT_MOE_QUANT_ONCE="${SGLANG_OPT_MOE_QUANT_ONCE:-1}"
export SGLANG_EXL3_REGISTER=1

: > "$LOG"

args=(
  --model-path "$MODEL_DIR"
  --served-model-name Qwen3.8-Flash-Next
  --host 127.0.0.1
  --port "$PORT"
  --quantization exl3
  --load-format exl3_mmap
  --trust-remote-code
  --context-length "$CTX_LEN"
  --mem-fraction-static "$MEM_FRAC"
  --max-running-requests "$MAX_REQS"
  --ple-offload-embedding
  --ple-offload-backend file
  --ple-offload-dir "${PLE_OFFLOAD_DIR:-./ple-qwen38}"
  --linear-attn-backend "$LINEAR_ATTN_BACKEND"
  --linear-attn-prefill-backend "$LINEAR_ATTN_PREFILL"
  --linear-attn-decode-backend "$LINEAR_ATTN_DECODE"
  --mamba-ssm-dtype "$MAMBA_SSM_DTYPE"
  --speculative-algorithm "$SPEC_ALGO"
  --speculative-num-steps "$SPEC_STEPS"
  --speculative-eagle-topk "$SPEC_TOPK"
  --speculative-num-draft-tokens "$SPEC_DRAFT_TOKENS"
  --speculative-attention-mode "$SPEC_ATTN_MODE"
  --skip-server-warmup
)

if [[ "$RADIX" != "1" ]]; then
  args+=(--disable-radix-cache)
fi
if [[ "$CUDA_GRAPH" != "1" ]]; then
  args+=(--disable-cuda-graph)
fi
if [[ -n "$MAMBA_SCHEDULER" ]]; then
  args+=(--mamba-scheduler-strategy "$MAMBA_SCHEDULER")
fi
if [[ "$ENABLE_REPLAYSSM" == "1" ]]; then
  args+=(--enable-linear-replayssm-spec)
fi
if [[ -n "$CHUNKED_PREFILL" ]]; then
  args+=(--chunked-prefill-size "$CHUNKED_PREFILL")
fi
# shellcheck disable=SC2206
if [[ -n "$EXTRA_ARGS" ]]; then
  extra=($EXTRA_ARGS)
  args+=("${extra[@]}")
fi

"$PYTHON" -m sglang_exl3.launch_server "${args[@]}" >"$LOG" 2>&1 &

echo "$!" > ./sglang_qwen.pid
echo "sglang_pid=$(cat ./sglang_qwen.pid) log=$LOG mtp=${SPEC_ALGO}_steps${SPEC_STEPS}_topk${SPEC_TOPK}_draft${SPEC_DRAFT_TOKENS} decode=$LINEAR_ATTN_DECODE mamba=$MAMBA_SSM_DTYPE radix=$RADIX replayssm=$ENABLE_REPLAYSSM moe=$SGLANG_EXL3_MOE_KERNEL"
