#!/bin/bash
# setup_and_run_opd_4xh200.sh — pull + organize the OPD models from HF, then run the single-node
# 4xH200 OPD experiment (Nguyen's reference) on chankhavu/aimo-opd-sft:v2. Run in a relay shell.
set -euo pipefail

# ============ config (export before running, or edit here) ============
MODELS_DIR="${MODELS_DIR:-/workspace/models}"                 # download target (use a big mounted disk!)
HF_BUNDLE="${HF_BUNDLE:-ycchen/proof-pilot-deploy-bundle}"
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY}"
export DATASET_PATH="${DATASET_PATH:?set DATASET_PATH to your proof CSV}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/opd_out}"         # checkpoints+logs (mounted volume)

# Which sub-bundle is POLICY (trained) vs TEACHER (frozen). Nguyen's reference used opd-32b-deploy
# for BOTH (self-distillation). Override to use opd-32b-v33-s150 for either.
POLICY_MODEL="${POLICY_MODEL:-opd-32b-deploy}"
TEACHER_MODEL="${TEACHER_MODEL:-opd-32b-deploy}"
# Extra bundles to pre-download but not necessarily use this run (space-separated). Empty to skip.
EXTRA_MODELS="${EXTRA_MODELS:-opd-32b-v33-s150}"

# ============ 1. download + organize the model bundles ============
export HF_XET_HIGH_PERFORMANCE=1
fetch_model() {   # <sub-bundle>
  local sub="$1"; local dest="$MODELS_DIR/$sub"
  if [ -f "$dest/config.json" ]; then echo "[skip] $sub already at $dest"; return; fi
  echo "[download] $HF_BUNDLE :: $sub/*  ->  $dest"
  hf download --repo-type model --local-dir "$dest" "$HF_BUNDLE" --include "$sub/*"
  # hf nests files under the repo path ($dest/$sub/*); flatten to $dest/*
  if [ -d "$dest/$sub" ]; then
    echo "[organize] flatten $dest/$sub -> $dest"
    ( shopt -s dotglob nullglob; mv "$dest/$sub"/* "$dest"/ ) && rmdir "$dest/$sub"
  fi
  [ -f "$dest/config.json" ] || { echo "  ERROR: no config.json in $dest (check bundle layout)" >&2; exit 1; }
}

mkdir -p "$MODELS_DIR"
for m in $(printf '%s\n' "$POLICY_MODEL" "$TEACHER_MODEL" $EXTRA_MODELS | sort -u); do
  fetch_model "$m"
done

# ============ 2. run the 4xH200 experiment ============
export MODEL_PATH="$MODELS_DIR/$POLICY_MODEL"
export TEACHER_MODEL_PATH="$MODELS_DIR/$TEACHER_MODEL"
echo "[run] policy=$MODEL_PATH"
echo "[run] teacher=$TEACHER_MODEL_PATH"
echo "[run] data=$DATASET_PATH  out=$OUTPUT_DIR"
eval "$(opd-env-sync)"
REPO=/tmp/imochallenge/opd-env/aimo-proof-pilot
exec bash "$REPO/operator_commands/run_opd_4xh200_ours.sh"
