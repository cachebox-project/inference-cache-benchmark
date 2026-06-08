"""
collect_vllm_metrics.py — scrape per-pod vLLM /metrics for the run window.

Phase 3 added a need for per-pod throughput + cache-hit-rate visibility that
the genai-bench Excel doesn't expose. The companion to collect_ic_metrics.py:
one row per scrape, one column per (metric, pod, label-combo), so the report
can compute deltas across the run window and break out per-pod behaviour.

What's scraped
--------------
Each ``--endpoint id=url`` is treated as one pod. Default Phase 3 targets:

  baseline mode:  baseline=http://localhost:38005/metrics
  cache-plane modes (ic-smoke 3-pod): r0=...:38010 r1=...:38011 r2=...:38012

Metric set (gauges + counters of interest, matching the scenario YAML
``vllm_metrics.capture`` block):

  - vllm:prefix_cache_queries_total       # T1 lookups
  - vllm:prefix_cache_hits_total          # T1 hits
  - vllm:external_prefix_cache_queries_total  # T2 (LMCache) lookups
  - vllm:external_prefix_cache_hits_total     # T2 hits
  - vllm:request_success_total
  - vllm:prompt_tokens_total
  - vllm:generation_tokens_total

CSV format
----------
  ts, <metric>{labels…,pod="<id>"}, …

The benchmark harness uses correlate.py to compute per-run deltas and the
throughput section of the report.

Usage
-----
  python3 collect_vllm_metrics.py \\
    --endpoint baseline=http://localhost:38005/metrics \\
    --endpoint r0=http://localhost:38010/metrics \\
    --endpoint r1=http://localhost:38011/metrics \\
    --endpoint r2=http://localhost:38012/metrics \\
    --interval 30 \\
    --output results/.../vllm-metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import signal
import sys
import time
from typing import Dict, List, Tuple

import requests

METRICS_OF_INTEREST = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:request_success_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)

# vLLM exposes metric names with colons, e.g. "vllm:prompt_tokens_total{...}".
# Prometheus exposition format permits ASCII metric names with letters, digits,
# underscores, and colons. Pattern matches both colon-free and colon-bearing
# metric names so adding non-vLLM metrics later doesn't need a regex change.
LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-0-9eE.+]+)\s*$"
)
LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_endpoint(spec: str) -> Tuple[str, str]:
    """Parse ``id=url`` (or bare ``url`` → id derived from host:port)."""
    if "=" in spec:
        pod_id, url = spec.split("=", 1)
        return pod_id.strip(), url.strip()
    # No id → fall back to "host:port" so the column header is at least unique
    m = re.match(r"https?://([^/]+)", spec)
    pod_id = m.group(1) if m else spec
    return pod_id, spec


def scrape_once(pod_id: str, endpoint: str) -> Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float]:
    """Return a dict keyed by (metric_name, sorted-label-tuple-with-pod) → value."""
    out: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
    try:
        r = requests.get(endpoint, timeout=5)
        r.raise_for_status()
    except Exception as e:
        sys.stderr.write(f"[vllm-scrape pod={pod_id}] error: {e}\n")
        return out
    for line in r.text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        if name not in METRICS_OF_INTEREST:
            continue
        labels = list(LABEL_RE.findall(m.group("labels") or ""))
        labels.append(("pod", pod_id))
        labels_t = tuple(sorted(labels))
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        out[(name, labels_t)] = value
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--endpoint",
        action="append",
        default=[],
        required=True,
        metavar="ID=URL",
        help="vLLM metrics endpoint as ID=URL. Repeatable, one per pod.",
    )
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    endpoints: List[Tuple[str, str]] = [parse_endpoint(e) for e in args.endpoint]

    stop = {"flag": False}
    def _stop(_sig, _frm):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    fieldnames = ["ts"]
    rows: List[dict] = []
    with open(args.output, "w", newline="") as f:
        writer: csv.DictWriter | None = None
        while not stop["flag"]:
            ts = time.time()
            row = {"ts": f"{ts:.3f}"}
            added_field = False
            for pod_id, url in endpoints:
                sample = scrape_once(pod_id, url)
                for (name, labels), value in sorted(sample.items()):
                    col = name + "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"
                    row[col] = value
                    if col not in fieldnames:
                        fieldnames.append(col)
                        added_field = True
            rows.append(row)
            # New label combos can appear mid-run when vLLM first emits a label
            # value (e.g. finish_reason="stop" only shows up after the first
            # successful generation). Rewrite the small CSV when that happens
            # so the header stays in sync with every row.
            if writer is None or added_field:
                f.seek(0)
                f.truncate()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            else:
                writer.writerow(row)
            assert writer is not None
            f.flush()
            for _ in range(args.interval):
                if stop["flag"]:
                    break
                time.sleep(1)

    sys.stderr.write(f"[vllm-scrape] wrote {args.output}\n")


if __name__ == "__main__":
    main()
