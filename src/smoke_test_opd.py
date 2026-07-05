#!/usr/bin/env python3
"""OPD container preflight — environment/system checks only (no dependency imports).

Run at startup to confirm the box is configured for an OPD run. It does NOT import the training
stack; it checks the environment the run depends on:

  * distributed env vars set     (WORLD_SIZE, GLOBAL_RANK, MASTER_ADDR, MASTER_PORT)
  * host driver supports CUDA 13.0  (cu130)
  * /tmp and $HOME are writable
  * creds present               (WANDB_API_KEY, GITHUB_TOKEN, HF_TOKEN)
  * enough free disk

Prints a PASS/WARN/FAIL report; exit code is non-zero iff a CRITICAL check fails.
"""
from __future__ import annotations
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_results: list[tuple[str, str, str]] = []

MIN_FREE_GB = float(os.environ.get("OPD_MIN_FREE_GB", "1000"))   # /tmp should hold models + checkpoints + caches; want ~1 TB
DIST_VARS = ["WORLD_SIZE", "GLOBAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
CRED_VARS = ["WANDB_API_KEY", "GITHUB_TOKEN", "HF_TOKEN"]
DISK_DIRS = ["/tmp", os.path.expanduser("~")]


def record(name: str, status: str, msg: str = "") -> None:
    _results.append((name, status, msg))
    icon = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[status]
    print(f"  [{icon}] {name:28s} {status:4s} {msg}")


def guard(name: str, fn, critical: bool = True) -> None:
    try:
        status, msg = fn()
    except Exception as exc:  # noqa: BLE001
        status, msg = (FAIL if critical else WARN), f"{type(exc).__name__}: {exc}"
    record(name, status, msg)


def c_dist_env() -> tuple[str, str]:
    missing = [v for v in DIST_VARS if not os.environ.get(v)]
    if missing:
        return FAIL, f"missing: {', '.join(missing)}"
    return PASS, " ".join(f"{v}={os.environ[v]}" for v in DIST_VARS)


def c_cu130() -> tuple[str, str]:
    p = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
    if p.returncode != 0:
        return FAIL, "nvidia-smi failed (no GPU / driver / --gpus,--nv missing?)"
    m = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", p.stdout)
    if not m:
        return WARN, "could not parse 'CUDA Version' from nvidia-smi"
    major, minor = int(m.group(1)), int(m.group(2))
    ok = (major, minor) >= (13, 0)
    return (PASS if ok else FAIL), f"driver supports CUDA {major}.{minor} (need >= 13.0 for cu130)"


def c_writable() -> tuple[str, str]:
    notes = []
    for label, d in (("/tmp", Path("/tmp")), ("$HOME", Path(os.path.expanduser("~")))):
        probe = d / ".opd_smoke_write_probe"
        probe.write_text("ok")
        probe.unlink()
        notes.append(f"{label} ok")
    return PASS, "; ".join(notes)


def c_creds() -> tuple[str, str]:
    missing = [v for v in CRED_VARS if not os.environ.get(v)]
    if missing:
        return FAIL, f"not set: {', '.join(missing)}"
    return PASS, "set: " + ", ".join(CRED_VARS)


def c_disk() -> tuple[str, str]:
    worst, where = None, None
    for d in DISK_DIRS:
        try:
            free = shutil.disk_usage(d).free / 1024**3
        except FileNotFoundError:
            continue
        if worst is None or free < worst:
            worst, where = free, d
    if worst is None:
        return WARN, "no disk dirs found to check"
    return (PASS if worst >= MIN_FREE_GB else WARN), f"min free {worst:.0f} GB at {where} (need ~{MIN_FREE_GB:.0f})"


def main() -> int:
    print("=" * 70)
    print("OPD container preflight smoke test")
    print("=" * 70)
    print("\nChecks:")
    checks = [
        ("distributed env vars", c_dist_env, True),
        ("cu130 driver support", c_cu130, True),
        ("/tmp + $HOME writable", c_writable, True),
        ("creds set (wandb/gh/hf)", c_creds, True),
        ("free disk", c_disk, False),
    ]
    for name, fn, critical in checks:
        guard(name, fn, critical=critical)

    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    n_warn = sum(1 for _, s, _ in _results if s == WARN)
    print("\n" + "=" * 70)
    print(f"RESULT: {len(_results)} checks | {n_fail} FAIL | {n_warn} WARN | "
          f"{len(_results) - n_fail - n_warn} PASS")
    print("=" * 70)
    if n_fail:
        print("CRITICAL failures — box is NOT ready for an OPD run.")
        return 1
    print("Ready, with warnings." if n_warn else "All green — box ready for an OPD run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
