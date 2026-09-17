#!/usr/bin/env bash
# Overlay the two SGLang-pin edits required for EXL3 MTP on Qwen4-Exp.
# Pin ships nn.Linear fc_* (packs never bind) and eagle_worker_v2.init_lm_head
# that reads lm_head.weight (EXL3 ParallelLMHead has trellis, not .weight).
#
# Usage:
#   SGLANG_PIN=/path/to/sglang/python bash scripts/apply_sglang_pin_overlays.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIN="${SGLANG_PIN:?set SGLANG_PIN to the SGLang python/ tree (contains sglang/srt/models)}"
MTP_DST="$PIN/sglang/srt/models/qwen4_exp_mtp.py"
EAGLE="$PIN/sglang/srt/speculative/eagle_worker_v2.py"
test -f "$MTP_DST"
test -f "$EAGLE"
PYTHON="${PYTHON:-python3}"
cp "$ROOT/patches/qwen4_exp_mtp.py" "$MTP_DST"
"$PYTHON" "$ROOT/scripts/patch_eagle_worker_lm_head.py" "$EAGLE"
echo "overlaid $MTP_DST"
echo "patched  $EAGLE"
