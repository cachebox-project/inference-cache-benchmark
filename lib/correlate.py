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
import re
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
                # DictReader uses key None for overflow fields when a row has
                # more columns than the header. That can happen with older
                # ic-metrics.csv files when a Prometheus label appears mid-run.
                if k is None:
                    continue
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


# ------------------- full latency metrics (TTFT/TPOT/E2E/output) ----------
#
# genai-bench writes one JSON per (traffic_scenario, concurrency) run, named
# like  <scenario>_<model>_concurrency_<N>_time_<T>s.json , with shape:
#   {"aggregated_metrics": {... StatField per metric ...},
#    "individual_request_metrics": [...]}
# Each StatField carries: min/max/mean/stddev/sum/p25/p50/p75/p90/p95/p99.
# We surface the four request-level latencies the comparison cares about,
# broken out per concurrency level.

# (display name, genai-bench field name)
GB_METRICS = [
    ("TTFT", "ttft"),
    ("TPOT", "tpot"),
    ("E2E latency", "e2e_latency"),
    ("Output latency", "output_latency"),
]
GB_STATS = ["mean", "p50", "p90", "p95", "p99", "min", "max"]

# Throughput fields written by genai-bench per (scenario, concurrency) run.
# Field names vary across genai-bench versions; we try a few synonyms and take
# the first hit. Scalar OR StatField (in which case we use .mean).
GB_THROUGHPUT_FIELDS = [
    ("input tok/s", [
        "input_throughput_tokens_per_s",
        "input_throughput",
        "mean_input_throughput_tokens_per_second",
    ]),
    ("output tok/s", [
        "output_throughput_tokens_per_s",
        "output_throughput",
        "mean_output_throughput_tokens_per_second",
    ]),
    ("requests/s", [
        "requests_per_second",
        "request_throughput",
        "requests_throughput_per_s",
        "mean_request_throughput_per_second",
    ]),
]
# Per-request token counts come from StatFields named num_*_tokens
# (we render the .mean as the "avg" column).
GB_AVG_TOKEN_FIELDS = [
    ("avg input tok", "num_input_tokens"),
    ("avg output tok", "num_output_tokens"),
]
# genai-bench reports these latencies in SECONDS. We render ms. If a future
# genai-bench version emits ms, flip this to False (sanity-check run #1 against
# the genai-bench Excel — a 27B-INT4 TTFT should be hundreds of ms, not <1).
GB_LATENCY_IS_SECONDS = True
_CONC_RE = re.compile(r"_(?:concurrency|batch_size)_(\d+)_time_\d+s\.json$")


def _find_statfield(node, metric: str):
    """Recursively locate the StatField dict for `metric` anywhere under an
    aggregated_metrics tree (robust to schema nesting across versions). A
    StatField is recognised as a dict carrying at least a mean or p99 key."""
    if isinstance(node, dict):
        v = node.get(metric)
        if isinstance(v, dict) and ("mean" in v or "p99" in v):
            return v
        for vv in node.values():
            r = _find_statfield(vv, metric)
            if r is not None:
                return r
    elif isinstance(node, list):
        for vv in node:
            r = _find_statfield(vv, metric)
            if r is not None:
                return r
    return None


def load_gb_runs(genai_bench_dir: str) -> Dict[int, Dict[str, dict]]:
    """Return {concurrency: {field: {stat: float}}} from genai-bench per-run JSON."""
    runs: Dict[int, Dict[str, dict]] = {}
    if not os.path.isdir(genai_bench_dir):
        return runs
    for path in glob.glob(os.path.join(genai_bench_dir, "**", "*.json"), recursive=True):
        m = _CONC_RE.search(os.path.basename(path))
        if not m:
            continue
        conc = int(m.group(1))
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        agg = data.get("aggregated_metrics", data)
        fields: Dict[str, dict] = {}
        for _disp, key in GB_METRICS:
            sf = _find_statfield(agg, key)
            if sf:
                fields[key] = {s: sf.get(s) for s in GB_STATS + ["p50"]}
        if fields:
            runs[conc] = fields
    return runs


def _find_scalar(node, keys: List[str]):
    """First-hit walker for scalar OR StatField values under one of `keys`.

    For StatField (a dict with mean/p99), returns the `.mean` value. Returns
    None if none of the candidate keys are reached anywhere in the tree.
    """
    if isinstance(node, dict):
        for k in keys:
            if k in node:
                v = node[k]
                if isinstance(v, dict):
                    if "mean" in v and isinstance(v["mean"], (int, float)):
                        return float(v["mean"])
                elif isinstance(v, (int, float)):
                    return float(v)
        for vv in node.values():
            r = _find_scalar(vv, keys)
            if r is not None:
                return r
    elif isinstance(node, list):
        for vv in node:
            r = _find_scalar(vv, keys)
            if r is not None:
                return r
    return None


def load_gb_throughput(genai_bench_dir: str) -> Dict[int, Dict[str, Optional[float]]]:
    """Return {concurrency: {field_label: value}} for the throughput section.

    Walks each per-concurrency JSON for (a) throughput scalars surfaced by
    genai-bench and (b) per-request token counts under num_input_tokens /
    num_output_tokens. Missing fields stay None — the renderer prints "—".
    """
    out: Dict[int, Dict[str, Optional[float]]] = {}
    if not os.path.isdir(genai_bench_dir):
        return out
    for path in glob.glob(os.path.join(genai_bench_dir, "**", "*.json"), recursive=True):
        m = _CONC_RE.search(os.path.basename(path))
        if not m:
            continue
        conc = int(m.group(1))
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        agg = data.get("aggregated_metrics", data)
        row: Dict[str, Optional[float]] = {}
        for label, candidates in GB_THROUGHPUT_FIELDS:
            row[label] = _find_scalar(agg, candidates)
        for label, key in GB_AVG_TOKEN_FIELDS:
            sf = _find_statfield(agg, key)
            mean = sf.get("mean") if isinstance(sf, dict) else None
            row[label] = float(mean) if isinstance(mean, (int, float)) else None
        if any(v is not None for v in row.values()):
            out[conc] = row
    return out


def render_throughput_table(genai_bench_dir: str) -> List[str]:
    """Single-run throughput section. One row per concurrency."""
    runs = load_gb_throughput(genai_bench_dir)
    out = ["## Throughput (per concurrency)", ""]
    if not runs:
        out.append("_No throughput fields parsed — see `genai-bench/` for raw Excel._")
        out.append("")
        return out
    cols = [label for label, _ in GB_THROUGHPUT_FIELDS] + [label for label, _ in GB_AVG_TOKEN_FIELDS]
    out.append("| concurrency | " + " | ".join(cols) + " |")
    out.append("|---|" + "---|" * len(cols))
    for conc in sorted(runs):
        cells = [f"c={conc}"]
        for col in cols:
            v = runs[conc].get(col)
            cells.append("—" if v is None else f"{v:,.1f}")
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return out


def render_throughput_comparison(rundirs: List[Tuple[str, str]]) -> List[str]:
    """Compare: per-concurrency throughput across labels, with Δ column."""
    per_label = [(lab, load_gb_throughput(os.path.join(d, "genai-bench"))) for lab, d in rundirs]
    all_conc = sorted({c for _lab, runs in per_label for c in runs})
    out = ["## Throughput — side by side (per concurrency)", ""]
    if not all_conc:
        out.append("_No throughput fields parsed in any run dir._")
        out.append("")
        return out
    cols = [label for label, _ in GB_THROUGHPUT_FIELDS] + [label for label, _ in GB_AVG_TOKEN_FIELDS]
    multi = len(per_label) >= 2
    for conc in all_conc:
        out.append(f"### Concurrency {conc}")
        out.append("")
        hdr = ["Metric"] + [lab for lab, _ in per_label] + (["Δ (last−first)"] if multi else [])
        out.append("| " + " | ".join(hdr) + " |")
        out.append("|" + "---|" * len(hdr))
        for col in cols:
            vals = [runs.get(conc, {}).get(col) for _lab, runs in per_label]
            cells = [col] + ["—" if v is None else f"{v:,.1f}" for v in vals]
            if multi and vals[0] is not None and vals[-1] is not None:
                d = vals[-1] - vals[0]
                pct = (d / vals[0] * 100) if vals[0] else 0
                cells.append(f"{d:+,.1f} ({pct:+.1f}%)")
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return out


# Run-window aggregate throughput from vllm-metrics.csv ----------------------
#
# When genai-bench's per-concurrency throughput scalars are missing (older
# versions, partial JSONs, etc.), the vllm-metrics scraper still gives us a
# cluster-aggregate floor: sum of (prompt_tokens_total Δ) and
# (generation_tokens_total Δ) across pods over the wall-clock run window.

def _vllm_metric_total_delta(
    rows: List[dict], metric_prefix: str
) -> Optional[float]:
    """Sum (last − first) across all columns whose key starts with `metric_prefix{`.

    Returns None if no matching columns or insufficient samples.
    """
    if len(rows) < 2:
        return None
    first, last = rows[0], rows[-1]
    matched = False
    total = 0.0
    for col, v in last.items():
        if not col.startswith(metric_prefix + "{"):
            continue
        if not isinstance(v, float):
            continue
        before = first.get(col, 0.0) if isinstance(first.get(col), float) else 0.0
        total += (v - before)
        matched = True
    return total if matched else None


def _vllm_run_window_seconds(rows: List[dict]) -> Optional[float]:
    """Wall-clock seconds between the first and last scrape ts in vllm-metrics.csv."""
    if len(rows) < 2:
        return None
    try:
        start = float(rows[0].get("ts"))
        end = float(rows[-1].get("ts"))
    except (TypeError, ValueError):
        return None
    return end - start if end > start else None


def render_vllm_aggregate_throughput(rundir: str) -> List[str]:
    """Single-run vllm-metrics.csv aggregate (across all pods, full run window)."""
    csv_path = os.path.join(rundir, "vllm-metrics.csv")
    rows = load_ic_metrics_csv(csv_path)
    out = ["## vLLM aggregate throughput (run window, all pods)", ""]
    if not rows:
        out.append("_No vllm-metrics.csv — vLLM scraper not enabled for this run._")
        out.append("")
        return out
    duration = _vllm_run_window_seconds(rows)
    input_delta = _vllm_metric_total_delta(rows, "vllm:prompt_tokens_total")
    output_delta = _vllm_metric_total_delta(rows, "vllm:generation_tokens_total")
    req_delta = _vllm_metric_total_delta(rows, "vllm:request_success_total")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| Run window | {duration:.1f} s |" if duration else "| Run window | — |")
    if duration and input_delta is not None:
        out.append(f"| Input tokens/s (sum-of-pods) | {input_delta / duration:,.1f} |")
    if duration and output_delta is not None:
        out.append(f"| Output tokens/s (sum-of-pods) | {output_delta / duration:,.1f} |")
    if duration and req_delta is not None:
        out.append(f"| Requests/s (sum-of-pods) | {req_delta / duration:,.2f} |")
    # Per-pod T1/T2 hit rates
    t1_q = _vllm_metric_total_delta(rows, "vllm:prefix_cache_queries_total")
    t1_h = _vllm_metric_total_delta(rows, "vllm:prefix_cache_hits_total")
    t2_q = _vllm_metric_total_delta(rows, "vllm:external_prefix_cache_queries_total")
    t2_h = _vllm_metric_total_delta(rows, "vllm:external_prefix_cache_hits_total")
    if t1_q and t1_q > 0 and t1_h is not None:
        out.append(f"| T1 (vLLM local prefix-cache) hit rate | {100.0 * t1_h / t1_q:.1f}% |")
    if t2_q and t2_q > 0 and t2_h is not None:
        out.append(f"| T2 (external/LMCache) hit rate | {100.0 * t2_h / t2_q:.1f}% |")
    out.append("")
    return out


def _fmt_lat(v) -> str:
    if v is None:
        return "—"
    return f"{v * 1000:.1f}" if GB_LATENCY_IS_SECONDS else f"{v:.1f}"


def render_latency_tables(genai_bench_dir: str) -> List[str]:
    """Single-run: one table per concurrency, rows=metric, cols=stats (ms)."""
    runs = load_gb_runs(genai_bench_dir)
    out = ["## Latency metrics — genai-bench (ms)", ""]
    if not runs:
        out.append("_No genai-bench per-run JSON found — see `genai-bench/` for the raw Excel._")
        out.append("")
        return out
    for conc in sorted(runs):
        out.append(f"### Concurrency {conc}")
        out.append("")
        out.append("| Metric | " + " | ".join(s.upper() for s in GB_STATS) + " |")
        out.append("|---|" + "---|" * len(GB_STATS))
        for disp, key in GB_METRICS:
            sf = runs[conc].get(key)
            if not sf:
                continue
            out.append("| " + " | ".join([disp] + [_fmt_lat(sf.get(s)) for s in GB_STATS]) + " |")
        out.append("")
    return out


def render_latency_comparison(rundirs: List[Tuple[str, str]]) -> List[str]:
    """Compare: per concurrency, rows="<metric> <stat>", cols=labels + Δ (ms)."""
    per_label = [(lab, load_gb_runs(os.path.join(d, "genai-bench"))) for lab, d in rundirs]
    all_conc = sorted({c for _lab, runs in per_label for c in runs})
    out = ["## Latency metrics — side by side (ms)", ""]
    if not all_conc:
        out.append("_No genai-bench per-run JSON found in any run dir._")
        out.append("")
        return out
    multi = len(per_label) >= 2
    for conc in all_conc:
        out.append(f"### Concurrency {conc}")
        out.append("")
        hdr = ["Metric / stat"] + [lab for lab, _ in per_label] + (["Δ (last−first)"] if multi else [])
        out.append("| " + " | ".join(hdr) + " |")
        out.append("|" + "---|" * len(hdr))
        for disp, key in GB_METRICS:
            for stat in GB_STATS:
                vals = [runs.get(conc, {}).get(key, {}).get(stat) for _lab, runs in per_label]
                cells = [f"{disp} {stat.upper()}"] + [_fmt_lat(v) for v in vals]
                if multi and vals[0] is not None and vals[-1] is not None:
                    scale = 1000 if GB_LATENCY_IS_SECONDS else 1
                    d = (vals[-1] - vals[0]) * scale
                    pct = ((vals[-1] - vals[0]) / vals[0] * 100) if vals[0] else 0.0
                    cells.append(f"{d:+.1f} ({pct:+.1f}%)")
                out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return out


# ----------------------------- single-run report --------------------------


def emit_single_run_report(scenario_path: str, label: str, mode: str, rundir: str) -> str:
    scenario = load_scenario_yaml(scenario_path)
    ic = load_ic_metrics_csv(os.path.join(rundir, "ic-metrics.csv"))
    gb = load_genai_bench_metrics(os.path.join(rundir, "genai-bench"))
    ttft = compute_ttft_percentiles_from_genai_bench(gb)
    gb_runs = load_gb_runs(os.path.join(rundir, "genai-bench"))
    # Robust TTFT p50 (ms) across concurrency levels — drives headline + gate
    # when the legacy top-level-key heuristic above can't parse the JSON shape.
    _scale = 1000 if GB_LATENCY_IS_SECONDS else 1
    ttft_p50_ms = sorted(
        r["ttft"]["p50"] * _scale
        for r in gb_runs.values()
        if r.get("ttft", {}).get("p50") is not None
    )
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
    elif ttft_p50_ms:
        out.append(f"| TTFT p50 (best concurrency) | {ttft_p50_ms[0]:.1f} ms |")
        out.append("| _full TTFT/TPOT/E2E/output breakdown_ | _see Latency metrics below_ |")
    else:
        out.append("| genai-bench TTFT | _not parsed — see `genai-bench/` for raw_ |")
    if hit_rate is not None:
        out.append(f"| inference-cache hit rate (PREFIX_MATCH / total) | **{hit_rate:.1f}%** |")
    if evictions is not None:
        out.append(f"| Evictions during run | {evictions:.0f} |")
    if index_peak is not None:
        out.append(f"| Index entries peak | {index_peak:.0f} |")
    out.append("")

    # Full latency breakdown (TTFT / TPOT / E2E / output) × stats, per concurrency
    out.extend(render_latency_tables(os.path.join(rundir, "genai-bench")))

    # Throughput — per-concurrency from genai-bench JSON, plus an aggregate
    # row-totals view from vllm-metrics.csv when available (Phase 3).
    out.extend(render_throughput_table(os.path.join(rundir, "genai-bench")))
    out.extend(render_vllm_aggregate_throughput(rundir))

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
            measured = min(p50s) if p50s else (ttft_p50_ms[0] if ttft_p50_ms else None)
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
    if os.path.exists(os.path.join(rundir, "vllm-metrics.csv")):
        out.append(f"- `{rundir}/vllm-metrics.csv` — per-pod vLLM /metrics (Phase 3)")
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

    # Full latency breakdown (TTFT / TPOT / E2E / output) × stats, per concurrency
    out.extend(render_latency_comparison(rundirs))

    # Throughput side-by-side
    out.extend(render_throughput_comparison(rundirs))

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
