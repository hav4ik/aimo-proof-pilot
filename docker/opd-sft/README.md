# chankhavu OPD container — baked, prime-rl-official (alternative to the root runtime-fetch image)

hav4ik's "bake-everything" container: an **immutable** image with the **official prime-rl cu128 stack**
(no cu130 / dev-nightly deviation) plus olmo-core (SFT), TransformerEngine, deep-gemm FP8, and the
remote-shell daemon. Published as `chankhavu/aimo-opd-sft:v2` on Docker Hub.

## Why baked (vs the runtime-fetch image at the repo root)
- **Reproducible + immutable** — nothing is installed at runtime, so no "unforeseen runtime error"
  surface, and it works on NII's read-only filesystem (only `/tmp` + `$HOME` writable).
- **prime-rl stays bit-for-bit official** — cu128, torch 2.11, vLLM 0.23, transformers 5.6.2 — built
  from `hav4ik/prime-rl-aimo@dev-vllm023`, the **pre-0.24-merge** state, so it does NOT carry the worker
  `ImportError` the 0.24 sync (`6ee9a5dc`) introduced. (See the issue list + comparison in
  `prime-rl-aimo/docs/opd/CONTAINER_COMPARISON.md`.)

## Two-stage build
1. **Base — `Dockerfile.opd` → `chankhavu/aimo-opd:v2`.** Overlays the fork's `prime_rl` source onto
   `primeintellect/prime-rl:main`, adds NII bind-safe compiler hardening, FP8 (deep-gemm), and
   TransformerEngine (cu12 source build). **Build context = the `hav4ik/prime-rl-aimo` repo checked out
   at `dev-vllm023`** (it COPYs `src/prime_rl`). This is *not* buildable from this repo — it needs the
   prime-rl-aimo source. Included here as the authoritative recipe.
2. **Combined — `Dockerfile.opd-sft` → `chankhavu/aimo-opd-sft:v2`.** `FROM aimo-opd:v2`; adds olmo-core
   into the same venv (rich pinned to 15), torchao 0.17, the remote-shell daemon (+ `opd-run` /
   `opd-status`), and the crash-resilient entrypoint. **Build from the repo root so `remote-shell/daemon`
   resolves:**
   ```
   docker build -f docker/opd-sft/Dockerfile.opd-sft -t chankhavu/aimo-opd-sft:v2 \
       --build-arg OPD_IMAGE=chankhavu/aimo-opd:v2 .
   ```

## Run
```
docker pull chankhavu/aimo-opd-sft:v2
# entrypoint boots the crash-resilient remote-shell daemon; training runs through its relay shells.
# See prime-rl-aimo/docs/opd/NII_RUNTIME.md for the InfiniBand env vars, opd-env-sync, and the
# operator-command adaptation (--no-fetch-update, model/data staging, checkpointing/disk).
```
