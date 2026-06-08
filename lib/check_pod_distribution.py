"""
check_pod_distribution.py — verify a benchmark actually spread traffic across replicas.

Why this exists: `kubectl port-forward svc/<name>` does NOT load-balance. The
apiserver pins each port-forward connection to a single endpoint. Across
multiple PF sessions, one replica can accumulate zero traffic while siblings
get millions of requests — looking like a kube-proxy bug but actually a PF
gotcha (CAC-163). This check turns that silent failure into a loud one.

Modes:
  snapshot   Scrape vllm:prefix_cache_queries_total from each replica's HTTP
             URL (taken from LOOKUP_PROXY_REPLICAS in the env) and write a
             JSON {replica_id: queries_total} to --out.
  diff       Read a "before" snapshot, scrape current values, print a per-
             replica delta table, and exit nonzero if any replica has Δ=0
             while at least one sibling has Δ>0 (the PF-pinning signature).

Both modes silently no-op when LOOKUP_PROXY_REPLICAS is unset (e.g. baseline
mode against a single-replica target). The point of the check is to catch
misconfiguration in *multi-replica* benchmarks.

Usage:
  python3 check_pod_distribution.py snapshot --out before.json
  # ... run the benchmark ...
  python3 check_pod_distribution.py diff --before before.json --out report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import requests

QUERIES_RE = re.compile(
    r'^vllm:prefix_cache_queries_total\{[^}]*\}\s+([-0-9eE.+]+)\s*$',
    re.MULTILINE,
)


def parse_replicas_env() -> List[Tuple[str, str]]:
    """Parse LOOKUP_PROXY_REPLICAS into [(replica_id, http_url), ...]."""
    raw = os.environ.get("LOOKUP_PROXY_REPLICAS", "").strip()
    if not raw:
        return []
    out: List[Tuple[str, str]] = []
    for spec in raw.split(","):
        parts = spec.split("|")
        if len(parts) < 3:
            sys.stderr.write(f"[dist-check] skipping malformed replica spec: {spec!r}\n")
            continue
        rid, _zmq, http_url = parts[0], parts[1], parts[2]
        out.append((rid.strip(), http_url.strip()))
    return out


def scrape_queries(http_url: str) -> float | None:
    """Return prefix_cache_queries_total for the engine, or None on error."""
    try:
        r = requests.get(http_url.rstrip("/") + "/metrics", timeout=5)
        r.raise_for_status()
    except Exception as e:
        sys.stderr.write(f"[dist-check] scrape error from {http_url}: {e}\n")
        return None
    m = QUERIES_RE.search(r.text)
    if not m:
        sys.stderr.write(f"[dist-check] no prefix_cache_queries_total at {http_url}\n")
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def snapshot(replicas: List[Tuple[str, str]]) -> Dict[str, float | None]:
    return {rid: scrape_queries(url) for rid, url in replicas}


def cmd_snapshot(args: argparse.Namespace) -> int:
    replicas = parse_replicas_env()
    if not replicas:
        sys.stderr.write("[dist-check] LOOKUP_PROXY_REPLICAS unset; skipping snapshot\n")
        with open(args.out, "w") as f:
            json.dump({"_skipped": True}, f)
        return 0
    snap = snapshot(replicas)
    with open(args.out, "w") as f:
        json.dump(snap, f, indent=2, sort_keys=True)
    sys.stderr.write(f"[dist-check] wrote snapshot ({len(snap)} replicas) → {args.out}\n")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    replicas = parse_replicas_env()
    if not replicas:
        sys.stderr.write("[dist-check] LOOKUP_PROXY_REPLICAS unset; skipping diff\n")
        return 0
    try:
        with open(args.before) as f:
            before = json.load(f)
    except FileNotFoundError:
        sys.stderr.write(f"[dist-check] before-snapshot missing: {args.before}\n")
        return 0
    if before.get("_skipped"):
        sys.stderr.write("[dist-check] before-snapshot marked skipped; nothing to diff\n")
        return 0
    after = snapshot(replicas)

    rows = []
    deltas: Dict[str, float] = {}
    for rid, _url in replicas:
        b = before.get(rid)
        a = after.get(rid)
        if b is None or a is None:
            rows.append((rid, b, a, None))
            continue
        d = a - b
        deltas[rid] = d
        rows.append((rid, b, a, d))

    # ---- Print human-readable table ----
    print()
    print("=== per-pod prefix_cache_queries_total Δ ===")
    print(f"{'replica':<12} {'before':>16} {'after':>16} {'Δ':>16}")
    for rid, b, a, d in rows:
        bs = f"{b:,.0f}" if isinstance(b, (int, float)) else "—"
        as_ = f"{a:,.0f}" if isinstance(a, (int, float)) else "—"
        ds = f"{d:,.0f}" if isinstance(d, (int, float)) else "—"
        print(f"{rid:<12} {bs:>16} {as_:>16} {ds:>16}")

    # ---- The actual check ----
    nonzero = [rid for rid, d in deltas.items() if d > 0]
    zero    = [rid for rid, d in deltas.items() if d == 0]
    verdict = "ok"
    detail  = ""
    rc = 0

    if not deltas:
        verdict = "skipped"
        detail = "no usable scrapes"
    elif not nonzero:
        verdict = "idle"
        detail = "no replica received traffic — benchmark may not have run"
    elif zero and nonzero:
        verdict = "imbalanced"
        detail = (
            f"replicas {zero} received zero traffic while {nonzero} did. "
            "Likely cause: client path doesn't load-balance "
            "(e.g. `kubectl port-forward svc/...` pins to one pod). "
            "Use per-pod PFs through lookup_proxy.py — see benchmarks/README.md "
            "Prerequisites callout."
        )
        rc = 2

    print()
    print(f"verdict: {verdict}")
    if detail:
        print(f"detail:  {detail}")
    print()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "verdict": verdict,
                    "detail": detail,
                    "before": before,
                    "after": after,
                    "delta": deltas,
                },
                f,
                indent=2,
                sort_keys=True,
            )

    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("snapshot", help="Write per-replica queries_total to JSON")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_snapshot)

    dp = sub.add_parser("diff", help="Compare current vs. before, warn on imbalance")
    dp.add_argument("--before", required=True)
    dp.add_argument("--out", default="", help="Optional JSON report path")
    dp.set_defaults(func=cmd_diff)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
