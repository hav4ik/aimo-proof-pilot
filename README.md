# AIMO Proof Pilot

Public training and container assets for proof-oriented OLMo3 / OLMo3Sink experiments. The main maintained path in this snapshot is Prime-RL OPD training with a DeepSeekMath-V2-style proof environment.

Large model weights, checkpoints, caches, `.sif` files, W&B runs, and private credentials are intentionally not committed.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/train.py` | Training wrapper used inside Docker/Singularity. It can fetch runtime updates and dispatch to SFT, Prime-RL, VERL, or operator mode. |
| `src/train_engine_rl.py` | Prime-RL launcher and config writer for OLMo3Sink / OPD training. |
| `src/proof_opd_env.py` | Current OPD environment: proof generation, verifier, meta-verifier, optional refinement, and verifiable-answer metrics. |
| `src/olmo3_sink/` | OLMo3Sink model, vLLM adapter, FA3 sink attention, and conversion helpers. |
| `operator_commands/` | Reproducible launch scripts for Modal or cluster/container runs. |
| `imo_data_1959_2024.csv` | Proof-style IMO data with `question` and `solution` columns. |
| `astralbench.csv` | Verifiable answer data with `problem` and `answer` columns, mixed into OPD training for boxed-answer accuracy tracking. |
| `Dockerfile` | CUDA 13 / Torch 2.11 image definition for Prime-RL and VERL experiments. |
| `*.def`, `scripts/build_sif_and_upload.sh` | Singularity/Apptainer build files and helpers. |

## Quick Checks

Run from the repository root:

```bash
python -m py_compile src/train.py src/train_engine.py src/train_engine_rl.py src/proof_opd_env.py
bash -n operator_commands/prime_rl_opd_4xh200_muon_imo_ctx16384_2train_1policy_1teacher.sh
```

The command filename is historical; the current default in that script is a 20,480-token trainer context.

## Current Best OPD Pipeline: 4xH200, 20k Context

Use:

```bash
bash operator_commands/prime_rl_opd_4xh200_muon_imo_ctx16384_2train_1policy_1teacher.sh
```

Default topology:

- GPU 0: policy vLLM rollout server.
- GPUs 1-2: trainer, `CP=2`, Ulysses context parallelism.
- GPU 3: frozen OPD teacher vLLM server.
- Optimizer: `muon`.
- Trainer FP8: enabled.
- Policy and teacher vLLM quantization: FP8.
- Trainer context length: `20480`.
- Rollout max completion tokens: `20480`.
- vLLM max model length: `40960`.
- vLLM `max_num_batched_tokens`: `16384`.
- Policy max concurrent sequences: `16`.
- Teacher max concurrent sequences: `8`.

Default data mix:

- `imo_data_1959_2024.csv` supplies proof-only tasks.
- `astralbench.csv` supplies verifiable tasks.
- `PRIME_OPD_VERIFIABLE_DATASET_PATH` selects the verifiable CSV, defaulting to `/workspace/submissions-instructions/astralbench.csv`.
- `PRIME_OPD_VERIFIABLE_FRACTION=0.20` mixes 20% verifiable rows into the train environment.
- `PRIME_OPD_VERIFIABLE_MIX_SEED=34521` makes the mixed proof/verifiable ordering reproducible.
- `PRIME_PROOF_MAX_EXAMPLES=20` keeps the default launch cheap. Increase it for real runs, for example `PRIME_PROOF_MAX_EXAMPLES=1481`.

Example container-style launch:

```bash
export PRIME_OPD_MODEL_PATH=/vol/olmo_train_assets/models/opd-32b-deploy/opd-32b-deploy
export PRIME_OPD_TEACHER_MODEL_PATH="$PRIME_OPD_MODEL_PATH"
export PRIME_OPD_DATASET_PATH=/workspace/submissions-instructions/imo_data_1959_2024.csv
export PRIME_OPD_VERIFIABLE_DATASET_PATH=/workspace/submissions-instructions/astralbench.csv
export PRIME_OPD_VERIFIABLE_FRACTION=0.20
export PRIME_OPD_VERIFIABLE_MIX_SEED=34521
export PRIME_PROOF_MAX_EXAMPLES=1481
export MAX_TRAIN_STEPS=30
export WANDB_MODE=online
export WANDB_PROJECT=olmo3-prime-rl

bash operator_commands/prime_rl_opd_4xh200_muon_imo_ctx16384_2train_1policy_1teacher.sh
```

The script expects `/app/train.py` inside the image and writes outputs/logs under `/vol/olmo_train_assets/`. Mount this repository at `/workspace/submissions-instructions` and mount a writable volume at `/vol/olmo_train_assets`.

At startup, the script prints the proof dataset path, verifiable dataset path, verifiable fraction, mix seed, max example count, context length, and rollout completion-token cap. Check these lines first when validating that a run is using the intended mixer settings.

## OPD Environment

`src/proof_opd_env.py` uses a staged pipeline:

1. Proof generation prompt.
2. Extract `## Solution` from the assistant output.
3. Verifier prompt over the extracted proof.
4. Meta-verifier prompt over the verifier analysis.
5. Optional refinement round if the selected reward is below the early-stop threshold.

For verifiable tasks, the proof-generation prompt additionally asks the model to include one final answer in `\boxed{...}` inside the `## Solution` section. The boxed answer is used only for metrics; the OPD proof/verifier/meta reward path stays unchanged.

Important W&B metrics:

- `proof_opd_reward`: final proof reward used by the environment.
- `proof_opd_format_score`: format compliance score.
- `proof_opd_proof_score`: verifier score.
- `proof_opd_meta_score`: meta-verifier score.
- `proof_opd_task_is_verifiable`: `1` for AstralBench-style rows, `0` for proof-only rows.
- `proof_opd_verifiable_accuracy`: `1` if the boxed answer matches, `0` if wrong or missing, `-1` for proof-only rows.
- `proof_opd_boxed_present`: whether a boxed answer was found in the solution section.

## Data Formats

Proof data should provide one of:

- `question`
- `problem`
- `messages` as a fallback source for the first user problem text

Optional `solution` is kept for reference. Verifiable data should provide:

- `problem` or `question`
- `answer`

`astralbench.csv` follows this format.

## Docker

Build:

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile -t aimo-proof-pilot:cu130 .
```

Run with four H200 GPUs:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace/submissions-instructions \
  -v /path/to/olmo_train_assets:/vol/olmo_train_assets \
  -e HF_TOKEN \
  -e WANDB_API_KEY \
  -e WANDB_MODE=online \
  aimo-proof-pilot:cu130 \
  bash /workspace/submissions-instructions/operator_commands/prime_rl_opd_4xh200_muon_imo_ctx16384_2train_1policy_1teacher.sh
```

## Singularity / Apptainer

Build and upload helpers are in `scripts/`:

```bash
bash scripts/build_sif_and_upload.sh
```

Manual run shape:

```bash
singularity run --nv container.sif \
  --backend prime_rl \
  --model_path /path/to/model \
  --dataset_path /path/to/imo_data_1959_2024.csv \
  --prime_env_id proof-opd-env
```

For the provided OPD shell script, bind this repo to `/workspace/submissions-instructions` and bind a writable model/output volume to `/vol/olmo_train_assets`.

## Secrets and Artifacts

Use environment variables for credentials:

- `HF_TOKEN`
- `WANDB_API_KEY`
- `OPENROUTER_API_KEY` if using API-judge paths

Do not commit model weights, checkpoints, generated caches, `.sif` files, W&B directories, private tokens, or presigned URLs.
