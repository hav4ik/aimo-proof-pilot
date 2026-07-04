# `aimo-opd-sft` — the baked OPD container (design & rationale)

This is **hav4ik's alternative container** to the runtime-fetch image at the repo root. Same goal
(run the OPD proof-pilot training), opposite build philosophy: **everything baked and immutable**,
with **prime-rl kept bit-for-bit official**. Published as `chankhavu/aimo-opd-sft:v2` on Docker Hub.

If you only read one thing: **the root image *fetches and installs* prime-rl at runtime; this one
*bakes* it.** That single choice cascades into every difference below.

---

## Why we built it this way

### 1. Bake everything, fetch/install nothing at runtime
The root container's `train.py` git-clones and `pip install`s prime-rl (and friends) at container
start. That's flexible, but it has three failure modes we wanted to eliminate:

- **Unforeseen runtime errors.** A runtime `pip install` can fail on a bad wheel, a transient network
  blip, a dependency-resolution change, or a compiler mismatch — *while you're holding a GPU
  allocation*. We move every one of those failures to **build time**, where they're cheap and
  reproducible.
- **Read-only filesystems.** On NII (Singularity/Apptainer) the container FS is **read-only except
  `/tmp` and `$HOME`**. A runtime `pip install` into site-packages simply **cannot work** there. A
  baked image runs as-is; runtime writes go only to `/tmp/imochallenge/*`.
- **Reproducibility.** A baked image is a fixed artifact — the exact stack that ran last week is the
  exact stack that runs today. A runtime-fetch image is "whatever `main` resolves to *this* boot,"
  which is how the worker bug below silently entered.

The one deliberate exception: **the OPD env code** (`proof_opd_env` — the rubric/reward) is
git-pulled at runtime via `opd-env-sync`, because it's iterated constantly and is pure-Python glue
whose dependencies are already baked. That's the *right* thing to fetch; the heavy stack is not.

### 2. prime-rl stays bit-for-bit official (cu128), because it's the primary vehicle
prime-rl is our main RL tool, so we refuse to deviate its stack. We keep the **official**
`torch 2.11+cu128 / vLLM 0.23-line / transformers 5.6.2` exactly as PrimeIntellect ships it. The
root image instead runs prime-rl on **cu130 + a vLLM *dev nightly*** — convenient for its multi-engine
goals, but that's a non-reproducible, nightly-regression-exposed stack on the component we least want
surprises in.

We do run on **cu13 host drivers** (NII/VastAI Blackwell/Hopper nodes) — cu128 code runs fine on a
cu13 driver via CUDA's backward compatibility (only `libcuda.so.1`, the driver stub, comes from the
host via `--nv`/`--gpus`). So "official cu128 image" and "cu13 cluster" are not in conflict.

### 3. Built from `dev-vllm023` — avoids a brand-new, breaking bug
Our prime-rl source is `hav4ik/prime-rl-aimo@dev-vllm023`, which is the fork's state **just before the
2026-07-03 vLLM-0.24 sync merge** (`6ee9a5dc`). That merge introduced a real regression (see
[The worker bug](#the-worker-bug-the-critical-one)). Building from the pre-merge state means we're on
the **proven** `prime_rl 0.6.0 @ vLLM 0.23` stack — the same one Nguyen's *working* W&B runs used —
with **none of the 0.24 breakage** and no need for a base-image `uv sync`.

### 4. One environment: prime-rl **and** olmo-core in the same venv
We fold the SFT engine (olmo-core) into prime-rl's `/app/.venv` instead of a second venv. This was
non-obvious but turned out clean:
- **transformers:** no conflict — prime-rl's 5.6.2 already has `Olmo3Config`; olmo-core is unpinned.
- **CUDA/kernels:** olmo-core reuses prime-rl's flash-attn / quack / liger; dense Olmo3 needs no
  grouped-gemm — so it's a near-pure pip install.
- **rich:** the one real clash — prime-rl needs `rich>=14` (via textual), olmo-core's cached-path caps
  `<14`. Resolved by keeping prime-rl's `rich 15` and forcing cached-path onto it (verified safe).
- **torchao:** bumped `0.15→0.17` so its C++ ext loads on torch 2.11 (needed for olmo-core FP8);
  torchao is olmo-core-only, so it can't affect prime-rl.

Result: one image does both **OPD (RL, primary)** and **SFT (backup)** with no glibc transplant, no
second toolkit, no SFT recompile.

### 5. NII-hardened compiler toolchain + tidy caches
On NII, host tools sit on **noexec** mounts, so any compiler resolved from a host path (or a host CUDA
bound onto `/usr/local/cuda`) dies with a `PermissionError` on first JIT. We ship a **bind-safe** nvcc:
the cu12.8 toolkit is copied to `/opt/opd/cuda` (an *unbindable* path), `CUDA_HOME` points there, and
host CUDA dirs go **last** on `PATH`. Every cache/JIT/scratch dir is namespaced under one umbrella,
`/tmp/imochallenge/cache/*`, so nothing writes to the read-only rootfs. TransformerEngine (for the
optional `te_fused_adamw` optimizer) is **source-compiled** for cu12 (no prebuilt wheel exists for
torch2.11+cu128) so prime-rl stays cu128.

### 6. Entrypoint is a crash-resilient remote-shell daemon
Instead of `train.py` as PID 1, the entrypoint is a supervisor that runs the NII-approved
outbound-only remote-shell daemon and **never exits** (ignores graceful signals; only SIGKILL/teardown
stops the container). It prints a loud banner to stdout so the launcher sees it's up, and ships
`opd-run`/`opd-status` so training launches **detached** and survives daemon restarts. Training runs
*through* relay shells; the container is a stable, remotely-controllable node.

---

## Key differences vs the root (Nguyen's) container

| Dimension | This image (`aimo-opd-sft`) | Root image (runtime-fetch) |
|---|---|---|
| prime-rl | **baked**, official | **runtime git-fetch + pip install** each boot |
| prime-rl source | `dev-vllm023` (pre-0.24-merge) — **no worker bug** | `nguyen599/prime-rl@main` (`6ee9a5dc`) — **worker bug present** |
| CUDA line | **cu128** (official) | cu130 |
| vLLM | 0.23-line, released wheel | **dev nightly** (`0.23.1rc1.dev699+cu130`) |
| transformers | 5.6.2 (prime-rl's pin) | 5.8.1 |
| engines baked | prime-rl (OPD) + olmo-core (SFT) — lean | + open-instruct + Megatron + verl — multi-engine, heavier |
| env model | one `/app/.venv` (both engines) | system Python |
| runtime installs | **none** | prime-rl + verifiers + env deps at boot |
| read-only FS (NII) | **works as-is** | runtime pip-install would fail; needs `--no-*` flags |
| compiler bind-safety | `/opt/opd/cuda` (unbindable) + host dirs last | conventional `/usr/local/cuda-13.0` |
| entrypoint | crash-resilient remote-shell daemon | `python /app/train.py` |
| reproducibility | fixed artifact | "whatever `main` resolves to this boot" |

Both bake the full compiler toolchain (nvcc + gcc) and all `.so` libraries (CUDA runtime, cudnn,
nccl, RDMA, flash-attn 2/3/4) — neither relies on host tools except the driver stub `libcuda.so.1`
(host-injected via `--nv`/`--gpus`). The differences are **which** stack, **when** it's assembled, and
**how bind-proof** it is.

### The worker bug (the critical one)
`src/prime_rl/inference/vllm/worker/__init__.py` imports `monkey_patch_skip_lora_module_warnings` and
calls `monkey_patch_LRUCacheWorkerLoRAManager()` — **both deleted from `patches.py` by upstream
`6836d325c`** (vLLM 0.24's native inplace-load). The 0.24-sync merge `6ee9a5dc` kept the *calls* but
took upstream's function-less `patches.py`. So importing the vLLM worker package throws
`ImportError` → the rollout worker dies → `Olmo3SinkForCausalLM` never registers → **OPD rollout can't
load the model.** Version-independent; it lives in `nguyen599/prime-rl@main` now. His working W&B runs
*predate* the merge; the next run on current `main` hits it. **This image sidesteps it entirely** by
building from `dev-vllm023`. (Upstream fix: drop the 2 stale calls — worth a PR.)

---

## Honest trade-offs
- **The root image is more flexible** for rapid iteration on a *writable* host: change a ref, reboot,
  it re-fetches. It also carries more engines (Megatron/verl/open-instruct) if you need them.
- **This image trades that flexibility for reproducibility, immutable-FS safety, an official
  prime-rl, and freedom from the worker bug.** For the *primary OPD vehicle on NII*, that's the safer
  bet. Quick runtime patches are still possible via the `/tmp + PYTHONPATH` overlay (`opd-env-sync`
  is the template); permanent changes are a rebuild — which is the point.

---

## Build & run
Two-stage build (see the Dockerfiles in this folder):
1. **`Dockerfile.opd` → `chankhavu/aimo-opd:v2`** (base). Overlays `prime_rl` from
   `prime-rl-aimo@dev-vllm023` onto `primeintellect/prime-rl:main`, adds NII hardening, FP8
   (deep-gemm), and TransformerEngine. *Not buildable from this repo — it needs the prime-rl-aimo
   source; included here as the authoritative recipe.*
2. **`Dockerfile.opd-sft` → `chankhavu/aimo-opd-sft:v2`** (combined). `FROM aimo-opd:v2`; adds
   olmo-core, torchao 0.17, the daemon + `opd-run`/`opd-status`, and the entrypoint. Build from the
   repo root so `remote-shell/daemon` resolves:
   ```
   docker build -f docker/opd-sft/Dockerfile.opd-sft -t chankhavu/aimo-opd-sft:v2 \
       --build-arg OPD_IMAGE=chankhavu/aimo-opd:v2 .
   ```

Run the reference 4×H200 OPD experiment (also the container smoke test after any change):
```
export HF_TOKEN=hf_...  WANDB_API_KEY=...  MODELS_DIR=/root/models
bash operator_commands/smoke_test_opd_4xh200.sh      # on main; opd-env-sync also fetches it
```
See `prime-rl-aimo/docs/opd/` (`CONTAINER_COMPARISON.md`, `NII_RUNTIME.md`, `AUDIT.md`) for the full
comparison, the NII runtime + InfiniBand setup, checkpointing/disk sizing, and the audit trail.
