#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next EXL3 on one DGX Spark GB10 via SGLang, no MTP draft.
#
# Env overrides:
#   MODEL_DIR   (required) local pack directory
#   PORT        (default 30000)
#   CTX_LEN     (default 4096)
#   MEM_FRAC    (default 0.95)
#   MAX_REQS    (default 1)
#   PYTHON      interpreter that has sglang_exl3 (default: python3)
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the local Qwen3.8-Flash-Next-exl3 pack directory}"
PORT="${PORT:-30000}"
CTX_LEN="${CTX_LEN:-4096}"
MEM_FRAC="${MEM_FRAC:-0.95}"
MAX_REQS="${MAX_REQS:-1}"
LOG="${LOG:-./sglang_qwen.log}"
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
export SGLANG_EXL3_REGISTER=1

: > "$LOG"

"$PYTHON" -m sglang_exl3.launch_server \
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
  --skip-server-warmup \
  >"$LOG" 2>&1 &

echo "$!" > ./sglang_qwen.pid
echo "sglang_pid=$(cat ./sglang_qwen.pid) log=$LOG"
