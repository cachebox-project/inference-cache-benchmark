"""Write a dataset.txt metadata file into a benchmark result directory.

This captures everything reproducible about the dataset used for a run, so a
historical comparison can be regenerated from the recorded knobs alone (see
CAC-159).

Captured per run:
  * dataset path (as recorded in scenario.yaml)
  * SHA-256, line count, byte count
  * first 200 chars of the first prompt (sanity check the file is the right
    kind of data)
  * generator config — embedded from <dataset>.meta.yaml when present
    (written by gen_*.py)

Invoked from run_tuning_bench.sh; also invoked by tools/backfill_dataset_meta.sh
for existing result dirs.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first_line(path: str) -> str:
    with open(path, "rb") as f:
        # Read enough to find a newline even for very long single-line prompts.
        chunk = f.read(1 << 16)
    nl = chunk.find(b"\n")
    raw = chunk if nl == -1 else chunk[:nl]
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return repr(raw)


def line_count(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            n += buf.count(b"\n")
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="Absolute path to the dataset file.")
    p.add_argument("--dataset-rel", default="", help="Dataset path as written in scenario.yaml (for the metadata record).")
    p.add_argument("--scenario", default="", help="Scenario name (label only, for the metadata record).")
    p.add_argument("--outdir", required=True, help="Result directory; dataset.txt is written here.")
    p.add_argument("--note", default="", help="Optional free-form note (used by the backfill tool to flag uncertainty).")
    args = p.parse_args(argv)

    if not os.path.isfile(args.dataset):
        sys.stderr.write(f"write_dataset_meta: dataset not found: {args.dataset}\n")
        return 1
    os.makedirs(args.outdir, exist_ok=True)

    digest = sha256_of(args.dataset)
    size = os.path.getsize(args.dataset)
    n_lines = line_count(args.dataset)
    head = first_line(args.dataset)[:200]

    meta_sidecar = args.dataset + ".meta.yaml"
    embedded_meta = ""
    if os.path.isfile(meta_sidecar):
        with open(meta_sidecar) as f:
            embedded_meta = f.read().rstrip()

    out_path = os.path.join(args.outdir, "dataset.txt")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rel = args.dataset_rel or args.dataset
    with open(out_path, "w") as f:
        f.write("# Dataset metadata — written by lib/write_dataset_meta.py (CAC-159).\n")
        f.write(f"# Captured at: {now}\n")
        if args.note:
            f.write(f"# NOTE: {args.note}\n")
        f.write("\n")
        if args.scenario:
            f.write(f"scenario: {args.scenario}\n")
        f.write(f"dataset_path: {rel}\n")
        f.write(f"sha256: {digest}\n")
        f.write(f"size_bytes: {size}\n")
        f.write(f"line_count: {n_lines}\n")
        f.write("first_prompt_head_200: |\n")
        # Block-scalar — indent each line two spaces; the head should have no
        # internal newlines but defend against it anyway.
        for line in head.splitlines() or [head]:
            f.write(f"  {line}\n")
        f.write("\n")
        if embedded_meta:
            f.write("generator_config:\n")
            for line in embedded_meta.splitlines():
                f.write(f"  {line}\n")
        else:
            f.write("generator_config: null   # no <dataset>.meta.yaml sidecar; cannot recover knobs\n")
        f.write("\n# Scenario YAML overrides (genai_bench:, etc.) are captured in scenario.yaml in this dir.\n")

    sys.stderr.write(f"wrote {out_path} (sha256={digest[:12]}…)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
