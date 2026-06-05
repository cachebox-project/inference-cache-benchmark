"""
collect_ic_metrics.py — scrape inference-cache-server /metrics over time.

Runs in the background during a benchmark. Writes a CSV (one row per scrape)
that correlate.py merges with genai-bench's per-request data.

Captured metrics (gauges + counters of interest):
  - inferencecache_lookup_route_calls_total{reason_code=...}
  - inferencecache_lookup_route_latency_seconds_{sum,count,bucket}
  - inferencecache_index_entries{...}
  - inferencecache_index_evictions_total{reason=...}
  - inferencecache_policy_auth_total{result=...}
  - inferencecache_snapshot_auth_total{result=...}
  - inferencecache_server_up

Adding a metric: just append to METRICS_OF_INTEREST below; the scraper picks it
up automatically. The scraper preserves all labels (one column per label
combination).

Usage:
  python3 collect_ic_metrics.py \
    --endpoint http://localhost:38001/metrics \
    --interval 10 \
    --output /path/to/ic-metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import signal
import sys
import time
from typing import Dict, Tuple

import requests

METRICS_OF_INTEREST = (
    "inferencecache_lookup_route_calls_total",
    "inferencecache_lookup_route_latency_seconds_sum",
    "inferencecache_lookup_route_latency_seconds_count",
    "inferencecache_index_entries",
    "inferencecache_index_evictions_total",
    "inferencecache_policy_auth_total",
    "inferencecache_snapshot_auth_total",
    "inferencecache_server_up",
    "inferencecache_server_grpc_tls_enabled",
)

# Match Prometheus exposition lines:  name{label="value",...} <number>
LINE_RE = re.compile(r"^(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-0-9eE.+]+)\s*$")
LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def scrape_once(endpoint: str) -> Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float]:
    """Return a dict keyed by (metric_name, sorted-label-tuple) → value."""
    out: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
    try:
        r = requests.get(endpoint, timeout=5)
        r.raise_for_status()
    except Exception as e:
        sys.stderr.write(f"[ic-scrape] error: {e}\n")
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
        labels = tuple(sorted(LABEL_RE.findall(m.group("labels") or "")))
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        out[(name, labels)] = value
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:38001/metrics")
    ap.add_argument("--interval", type=int, default=10)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # Allow clean teardown via SIGTERM (run_tuning_bench.sh kills us at end)
    stop = {"flag": False}
    def _stop(_sig, _frm):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    first = True
    fieldnames = ["ts"]
    with open(args.output, "w", newline="") as f:
        writer: csv.DictWriter | None = None
        while not stop["flag"]:
            ts = time.time()
            sample = scrape_once(args.endpoint)
            row = {"ts": f"{ts:.3f}"}
            for (name, labels), value in sorted(sample.items()):
                col = name
                if labels:
                    col += "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"
                row[col] = value
                if col not in fieldnames:
                    fieldnames.append(col)
            # On first iteration write the header; on subsequent, add new
            # columns lazily (Prometheus may surface new label combos as the
            # workload runs — e.g. a new reason_code value).
            if first:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                first = False
            else:
                # If new columns appeared, rewrite the file with the expanded header.
                if writer is not None and writer.fieldnames != fieldnames:
                    f.flush()
                    # Cheap path: just write what we have; correlate.py can
                    # tolerate sparse columns. Avoid the rewrite cost mid-run.
                    pass
            assert writer is not None
            writer.writerow(row)
            f.flush()
            # Sleep with early-exit support
            for _ in range(args.interval):
                if stop["flag"]:
                    break
                time.sleep(1)

    sys.stderr.write(f"[ic-scrape] wrote {args.output}\n")


if __name__ == "__main__":
    main()
