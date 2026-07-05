#!/usr/bin/env bash
# Build the OPD .sif from the LOCAL docker image (chankhavu/aimo-proof-pilot:v4).
#
# apptainer/SingularityCE 1.5.1 bundles mksquashfs 4.7.5, which SIGSEGVs ("Bug in orderer") on large
# rootfs — and our rootfs is ~35 GB. So we skip the doomed direct build and go straight to the manual
# route: build a --sandbox (no squashfs) -> compress it with the STABLE system mksquashfs -> assemble
# the SIF by hand. The runscript/labels/env live inside the rootfs (/.singularity.d), so a squashfs-only
# SIF still runs `singularity run`.
#
# Usage:  bash build_opd_sif.sh [OUT.sif] [DOCKER_TAG] [DEF]
# Defaults: OUT=aimo-opd-v4.sif  TAG=chankhavu/aimo-proof-pilot:v4  DEF=aimo-opd-v4.def
# Needs ~60 GB free in $OPD_SIF_WORKDIR (default /tmp): ~35 GB sandbox + ~25 GB squashfs.
set -euo pipefail

OUT="${1:-aimo-opd-v4.sif}"
TAG="${2:-chankhavu/aimo-proof-pilot:v4}"
DEF="${3:-aimo-opd-v4.def}"
WORK="${OPD_SIF_WORKDIR:-/tmp}"
SB="$(mktemp -d "$WORK/opd_sb.XXXX")"
SQUASH="$(mktemp -u "$WORK/opd_rootfs.XXXX.squashfs")"

cleanup() { rm -rf "$SB" "$SQUASH"; }
trap cleanup EXIT

[ -f "$DEF" ] || { echo "ERROR: $DEF not found (run from where the .def is)"; exit 1; }
docker image inspect "$TAG" >/dev/null 2>&1 || { echo "ERROR: docker image $TAG not found locally"; exit 1; }

# Guard: opd_secrets.env must exist and be well-formed BEFORE we build — else we'd bake a
# credential-less / broken .sif that fails on NII with no way to fix it live. The DEF's %files copies
# this (cwd-relative), so run this script from the dir that holds the .def AND opd_secrets.env.
SECRETS="opd_secrets.env"
[ -f "$SECRETS" ] || { echo "ERROR: $SECRETS not found — create it (one per line, plain KEY=VALUE):"; \
    echo "         HF_TOKEN=...  WANDB_API_KEY=...  GITHUB_TOKEN=..."; exit 1; }
if grep -qE '(^|[[:space:]])export |=[[:space:]]*["'\'']' "$SECRETS"; then
    echo "ERROR: $SECRETS must be plain KEY=VALUE — no 'export', no quotes (the %environment loop parses it raw)"; exit 1
fi
for k in HF_TOKEN WANDB_API_KEY GITHUB_TOKEN; do
    v="$(sed -n "s/^${k}=//p" "$SECRETS" | head -1)"
    [ -n "$v" ] || { echo "ERROR: $SECRETS is missing or has an empty ${k}"; exit 1; }
done
echo ">> secrets file OK: $SECRETS (HF_TOKEN, WANDB_API_KEY, GITHUB_TOKEN present)"

echo ">> Step 1/3: build sandbox from $DEF (extracts $TAG; ~35 GB, several min)"
rm -rf "$SB"
singularity build --fakeroot --sandbox "$SB" "$DEF"

echo ">> Step 2/3: compress with system mksquashfs ($(/usr/bin/mksquashfs -version 2>/dev/null | head -1))"
/usr/bin/mksquashfs "$SB" "$SQUASH" -noappend -all-root -comp gzip

echo ">> Step 3/3: assemble SIF (primary system partition, squashfs, amd64)"
rm -f "$OUT"
singularity sif new "$OUT"
singularity sif add "$OUT" "$SQUASH" --datatype 4 --parttype 2 --partfs 1 --partarch 2

echo ">> Done."
singularity sif list "$OUT"
ls -lh "$OUT"
echo ">> Verify:  singularity run --nv --containall --bind /tmp $OUT   (or: exec … python /app/smoke_test_opd.py)"
