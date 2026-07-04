#!/bin/bash
# smoke_test_opd_4xh200.sh — ONE-COMMAND OPD run on chankhavu/aimo-opd-sft:v2, to validate the
# container after any change. Captures the whole flow: pull env code -> download models ->
# run Nguyen's reference 4xH200 experiment. Save this file on the box and re-run it any time.
#
# Usage — set the 3 required vars, then run:
#   export HF_TOKEN=hf_...  WANDB_API_KEY=...  MODELS_DIR=/root/models
#   bash smoke_test_opd_4xh200.sh
#
# Optional overrides:
#   MAX_TRAIN_STEPS=2      # fast sanity run (default 30)
#   OUTPUT_DIR=...         # default $MODELS_DIR/opd_out
#   WANDB_PROJECT=...      # default olmo3-prime-rl
#   POLICY_MODEL / TEACHER_MODEL / EXTRA_MODELS   # see setup_and_run_opd_4xh200.sh
#   DATASET_PATH=...       # default: bundled proofbench_v3.csv
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY}"
export MODELS_DIR="${MODELS_DIR:?set MODELS_DIR to a disk with >=130GB free (run: df -h)}"
export OUTPUT_DIR="${OUTPUT_DIR:-$MODELS_DIR/opd_out}"
export WANDB_PROJECT="${WANDB_PROJECT:-olmo3-prime-rl}"

echo "=== OPD smoke test on $(hostname) — $(date -u) ==="
command -v nvidia-smi >/dev/null && nvidia-smi -L | sed 's/^/[gpu] /' || echo "  (no nvidia-smi)"
df -h "$MODELS_DIR" 2>/dev/null | tail -1 || df -h | tail -3

# 1. pull env code (proof_opd_env + train.py + operator scripts) + set PYTHONPATH
eval "$(opd-env-sync)"
REPO=/tmp/imochallenge/opd-env/aimo-proof-pilot

# 2. use the bundled proof dataset unless overridden (no external data needed)
export DATASET_PATH="${DATASET_PATH:-$REPO/src/proof_opd_env/proofbench_v3.csv}"
ls -la "$DATASET_PATH"

# 3. download models (cached after first run) + run the 4xH200 experiment
mkdir -p "$MODELS_DIR" "$OUTPUT_DIR"
echo "=== launch: models=$MODELS_DIR out=$OUTPUT_DIR wandb=$WANDB_PROJECT steps=${MAX_TRAIN_STEPS:-30} ==="
exec bash "$REPO/operator_commands/setup_and_run_opd_4xh200.sh"
