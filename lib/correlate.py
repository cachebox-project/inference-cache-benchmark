"""
correlate.py — merge genai-bench output + IC metrics → markdown report.

Two modes:
  --rundir <dir>        : single-run report (used by `run_tuning_bench.sh run`)
  --compare --labels A B [C] : multi-run comparison report

Outputs markdown to stdout.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


# ----------------------------- loaders ------------------------------------


def load_scenario_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_ic_metrics_csv(path: str) -> List[dict]:
    """Return list of scrape rows with float values."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            out = {}
            for k, v in r.items():
                if v in (None, ""):
                    continue
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
            rows.append(out)
    return rows


def load_genai_bench_metrics(genai_bench_dir: str) -> Optional[dict]:
    """
    Find the genai-bench experiment summary JSON and load it.

    genai-bench typically writes:
      <experiment-folder>/<run-name>/<scenario>/<concurrency>/metrics.json
    or an aggregate summary at the top level. We grep for any *.json file
    containing aggregated TTFT and pick the most recent.
    """
    if not os.path.isdir(genai_bench_dir):
        return None
    # Cheapest signal: a top-level JSON
    candidates = sorted(glob.glob(os.path.join(genai_bench_dir, "**", "*.json"), recursive=True),
                        key=os.path.getmtime, reverse=True)
    for path in candidates:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        # Heuristic — accept anything that looks like a metrics dump.
        if isinstance(data, dict) and any("ttft" in str(k).lower() for k in (data.keys() if isinstance(data, dict) else [])):
            data["_source"] = path
            return data
    # Fallback: nothing found
    return None


# ----------------------------- analysis -----------------------------------


def compute_hit_rate(ic_rows: List[dict]) -> Optional[float]:
    """Compute PREFIX_MATCH / total LookupRoute calls from first→last sample."""
    if not ic_rows:
        return None
    first, last = ic_rows[0], ic_rows[-1]
    match = 0.0
    total = 0.0
    for col, v in last.items():
        if not col.startswith("inferencecache_lookup_route_calls_total"):
            continue
        if not isinstance(v, float):
            continue
        before = first.get(col, 0.0) if isinstance(first.get(col), float) else 0.0
        delta = v - before
        total += delta
        if 'reason_code="PREFIX_MATCH"' in col:
            match += delta
    return (100.0 * match / total) if total > 0 else None


def compute_eviction_delta(ic_rows: List[dict]) -> Optional[float]:
    if not ic_rows:
        return None
    first, last = ic_rows[0], ic_rows[-1]
    total = 0.0
    for col, v in last.items():
        if not col.startswith("inferencecache_index_evictions_total"):
            continue
        if not isinstance(v, float):
            continue
        before = first.get(col, 0.0) if isinstance(first.get(col), float) else 0.0
        total += (v - before)
    return total


def compute_index_peak(ic_rows: List[dict]) -> Optional[float]:
    peak = 0.0
    for row in ic_rows:
        for col, v in row.items():
            if col.startswith("inferencecache_index_entries") and isinstance(v, float):
                peak = max(peak, v)
    return peak if peak > 0 else None


def compute_ttft_percentiles_from_genai_bench(gb: Optional[dict]) -> Dict[str, float]:
    """Best-effort: pull p50/p95/p99 TTFT from whatever genai-bench wrote.

    genai-bench's exact schema varies by version — we walk the dict looking for
    `ttft.p50` / `ttft.p99` style fields. If absent, returns an empty dict
    and the report says "see genai-bench/ for details."
    """
    if not gb:
        return {}
    out = {}
    def walk(d, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(d, (int, float)) and "ttft" in path.lower():
            # Pick out *.ttft.{p50,p95,p99,mean}
            tail = path.lower().split(".")[-1]
            if tail in {"p50", "p95", "p99", "mean", "median"} and path not in out:
                out[path] = float(d)
    walk(gb)
    return out


# ----------------------------- single-run report --------------------------


def emit_single_run_report(scenario_path: str, label: str, mode: str, rundir: str) -> str:
    scenario = load_scenario_yaml(scenario_path)
    ic = load_ic_metrics_csv(os.path.join(rundir, "ic-metrics.csv"))
    gb = load_genai_bench_metrics(os.path.join(rundir, "genai-bench"))
    ttft = compute_ttft_percentiles_from_genai_bench(gb)
    hit_rate = compute_hit_rate(ic)
    evictions = compute_eviction_delta(ic)
    index_peak = compute_index_peak(ic)

    out = []
    out.append(f"# Benchmark report — `{label}` ({mode})")
    out.append("")
    out.append(f"- **Scenario:** {scenario.get('name','?')} — {scenario.get('description','').strip().splitlines()[0] if scenario.get('description') else ''}")
    out.append(f"- **Mode:** `{mode}`")
    out.append(f"- **When:** {datetime.utcnow().isoformat()}Z")
    out.append(f"- **Run dir:** `{rundir}`")
    out.append("")

    out.append("## Headline metrics")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    if ttft:
        for k in sorted(ttft):
            out.append(f"| genai-bench {k} | {ttft[k]:.2f} ms |")
    else:
        out.append("| genai-bench TTFT | _not parsed — see `genai-bench/` for raw_ |")
    if hit_rate is not None:
        out.append(f"| inference-cache hit rate (PREFIX_MATCH / total) | **{hit_rate:.1f}%** |")
    if evictions is not None:
        out.append(f"| Evictions during run | {evictions:.0f} |")
    if index_peak is not None:
        out.append(f"| Index entries peak | {index_peak:.0f} |")
    out.append("")

    # Acceptance gate
    accept = scenario.get("acceptance") or {}
    if accept:
        out.append("## Acceptance gate")
        out.append("")
        out.append("| Criterion | Target | Measured | Verdict |")
        out.append("|---|---|---|---|")
        # TTFT p50 max
        if "ttft_p50_max_ms" in accept:
            t = accept["ttft_p50_max_ms"]
            # Find ANY parsed p50; pick the smallest (best concurrency)
            p50s = [v for k, v in ttft.items() if k.endswith("p50") or k.endswith("median")]
            measured = min(p50s) if p50s else None
            verdict = "—" if measured is None else ("✅ PASS" if measured <= t else "❌ FAIL")
            out.append(f"| TTFT p50 max | ≤ {t} ms | {measured if measured else 'n/a'} | {verdict} |")
        if "lookup_hit_rate_pct_min" in accept and hit_rate is not None:
            t = accept["lookup_hit_rate_pct_min"]
            verdict = "✅ PASS" if hit_rate >= t else "❌ FAIL"
            out.append(f"| Lookup hit rate min | ≥ {t}% | {hit_rate:.1f}% | {verdict} |")
        out.append("")

    out.append("## CRD snapshot at run time")
    snap = os.path.join(rundir, "crd-snapshot.yaml")
    out.append("")
    if os.path.exists(snap) and os.path.getsize(snap) > 0:
        out.append(f"See `{snap}`. Use this to diff vs. future runs.")
    else:
        out.append("_(no CRD snapshot captured)_")
    out.append("")

    out.append("## Raw artifacts")
    out.append("")
    out.append(f"- `{rundir}/scenario.yaml` — exact scenario used")
    out.append(f"- `{rundir}/genai-bench/` — genai-bench experiment dir (Excel via `genai-bench excel`)")
    out.append(f"- `{rundir}/ic-metrics.csv` — inference-cache /metrics scraped every {scenario.get('ic_metrics',{}).get('scrape_interval_s', 10)}s")
    out.append(f"- `{rundir}/crd-snapshot.yaml` — CRDs at run time")
    if mode == "lookup":
        out.append(f"- `{rundir}/lookup_proxy.log` — proxy log (LookupRoute call traces)")
    out.append("")
    return "\n".join(out)


# ----------------------------- comparison report --------------------------


def find_latest_rundir(results_dir: str, label: str) -> Optional[str]:
    matches = sorted(glob.glob(os.path.join(results_dir, f"{label}-*")), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def emit_comparison_report(results_dir: str, labels: List[str]) -> str:
    rundirs = []
    for lab in labels:
        d = find_latest_rundir(results_dir, lab)
        if not d:
            print(f"WARN: no run dir found for label '{lab}'", file=sys.stderr)
            continue
        rundirs.append((lab, d))

    out = []
    out.append(f"# Comparison: " + " vs. ".join(f"`{lab}`" for lab, _ in rundirs))
    out.append("")
    out.append(f"- When: {datetime.utcnow().isoformat()}Z")
    out.append("")

    out.append("## Headline metrics — side by side")
    out.append("")
    headers = ["Metric"] + [lab for lab, _ in rundirs] + (["Δ (last − first)"] if len(rundirs) >= 2 else [])
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "---|" * len(headers))

    rows: Dict[str, List[Optional[float]]] = defaultdict(lambda: [None] * len(rundirs))
    for i, (lab, d) in enumerate(rundirs):
        ic = load_ic_metrics_csv(os.path.join(d, "ic-metrics.csv"))
        gb = load_genai_bench_metrics(os.path.join(d, "genai-bench"))
        ttft = compute_ttft_percentiles_from_genai_bench(gb)
        for k, v in ttft.items():
            rows[f"genai-bench {k}"][i] = v
        hr = compute_hit_rate(ic)
        if hr is not None:
            rows["hit rate %"][i] = hr
        ev = compute_eviction_delta(ic)
        if ev is not None:
            rows["evictions"][i] = ev
        ip = compute_index_peak(ic)
        if ip is not None:
            rows["index entries peak"][i] = ip

    def fmt(v: Optional[float]) -> str:
        return "—" if v is None else f"{v:.1f}"

    for metric, values in sorted(rows.items()):
        cells = [metric] + [fmt(v) for v in values]
        if len(values) >= 2 and values[0] is not None and values[-1] is not None:
            delta = values[-1] - values[0]
            pct = (delta / values[0] * 100) if values[0] else 0
            cells.append(f"{delta:+.1f} ({pct:+.1f}%)")
        out.append("| " + " | ".join(cells) + " |")
    out.append("")

    # CRD diff between A and B (first two only — multi-way diff is too noisy)
    if len(rundirs) >= 2:
        out.append("## CRD diff (first vs. last)")
        out.append("")
        a = os.path.join(rundirs[0][1], "crd-snapshot.yaml")
        b = os.path.join(rundirs[-1][1], "crd-snapshot.yaml")
        if os.path.exists(a) and os.path.exists(b):
            import subprocess
            try:
                diff = subprocess.run(["diff", "-u", a, b], capture_output=True, text=True, timeout=10)
                out.append("```diff")
                out.append(diff.stdout[:5000] if diff.stdout else "(no differences)")
                out.append("```")
            except Exception as e:
                out.append(f"_(diff failed: {e})_")
        else:
            out.append("_(CRD snapshots missing — can't diff)_")
        out.append("")

    out.append("## Verdict — caller's responsibility")
    out.append("")
    out.append("This harness reports numbers; deciding whether a delta beats the noise floor (~5%) is up to you. Run 3× per label if you need confidence on small wins.")
    return "\n".join(out)


# ----------------------------- main ----------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario")
    ap.add_argument("--label")
    ap.add_argument("--mode")
    ap.add_argument("--rundir")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--results-dir")
    ap.add_argument("--labels", nargs="+")
    args = ap.parse_args()

    if args.compare:
        if not args.results_dir or not args.labels:
            sys.exit("--compare needs --results-dir and --labels")
        print(emit_comparison_report(args.results_dir, args.labels))
    else:
        if not all([args.scenario, args.label, args.mode, args.rundir]):
            sys.exit("single-run mode needs --scenario --label --mode --rundir")
        print(emit_single_run_report(args.scenario, args.label, args.mode, args.rundir))


if __name__ == "__main__":
    main()
