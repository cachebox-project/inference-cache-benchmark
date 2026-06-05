#!/usr/bin/env bash
# run_tuning_bench.sh — inference-cache tuning harness, built on genai-bench.
#
# Subcommands:
#   run --scenario <name> --label <label> --mode <baseline|no-hint|lookup>
#   compare <label1> <label2> [<label3>]
#   list-scenarios
#   clean [--keep-last N]
#
# Configuration via env vars (or defaults below):
#   IC_SERVER_METRICS   — inference-cache server /metrics URL  (default: http://localhost:38001/metrics)
#   IC_SERVER_GRPC      — inference-cache server gRPC endpoint (default: localhost:38002)
#   VLLM_ENGINE_URL     — cache-enabled vLLM HTTP URL          (default: http://localhost:38000)
#   VLLM_BASELINE_URL   — vanilla-vLLM (no cache plane) URL    (default: http://localhost:38005)
#   LOOKUP_PROXY_PORT   — proxy listen port for `lookup` mode  (default: 18100)
#   KUBECONFIG          — kubeconfig path for the CRD snapshot (default: from environment)
#
# See README.md for the full design.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SCENARIOS_DIR="$ROOT/scenarios"
RESULTS_DIR="$ROOT/results"
LIB_DIR="$ROOT/lib"
PROTO_DIR="${INFERENCE_CACHE_PROTO_DIR:-$ROOT/proto}"

: "${IC_SERVER_METRICS:=http://localhost:38001/metrics}"
: "${IC_SERVER_GRPC:=localhost:38002}"
: "${VLLM_ENGINE_URL:=http://localhost:38000}"
: "${VLLM_BASELINE_URL:=http://localhost:38005}"
: "${LOOKUP_PROXY_PORT:=18100}"
: "${WORKLOAD_NAMESPACE:=default}"

# -------- helpers --------
color_g() { printf '\033[32m%s\033[0m\n' "$*"; }
color_y() { printf '\033[33m%s\033[0m\n' "$*"; }
color_r() { printf '\033[31m%s\033[0m\n' "$*"; }
die()     { color_r "$*"; exit 1; }

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \?//'
  exit "${1:-0}"
}

cmd_list_scenarios() {
  echo "Available scenarios in $SCENARIOS_DIR:"
  for f in "$SCENARIOS_DIR"/*.yaml; do
    name=$(basename "$f" .yaml)
    desc=$(yq '.description // ""' "$f" | head -1)
    printf "  %-25s %s\n" "$name" "${desc:0:80}"
  done
}

cmd_clean() {
  local keep_last=10
  if [[ "${1:-}" == "--keep-last" ]]; then keep_last="$2"; fi
  echo "Cleaning old result dirs (keeping last $keep_last)…"
  cd "$RESULTS_DIR" || die "no results dir"
  ls -dt */ 2>/dev/null | tail -n +"$((keep_last+1))" | xargs -r rm -rf
}

cmd_run() {
  local scenario="" label="" mode="lookup"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --scenario) scenario="$2"; shift 2;;
      --label)    label="$2";    shift 2;;
      --mode)     mode="$2";     shift 2;;
      *) die "unknown arg: $1";;
    esac
  done
  [[ -z "$scenario" || -z "$label" ]] && die "usage: run --scenario <name> --label <label> [--mode lookup|no-hint|baseline]"

  local cfg="$SCENARIOS_DIR/$scenario.yaml"
  [[ -f "$cfg" ]] || die "scenario not found: $cfg"

  local ts="$(printf '%(%Y%m%d-%H%M%S)T' -1)"
  local outdir="$RESULTS_DIR/${label}-${ts}"
  mkdir -p "$outdir/genai-bench"

  color_g "[1/6] Scenario: $scenario   label: $label   mode: $mode"
  color_g "[1/6] Output:   $outdir"
  cp "$cfg" "$outdir/scenario.yaml"

  # ---- parse scenario YAML ----
  local task model tokenizer max_req max_time
  task=$(yq '.genai_bench.task' "$cfg")
  model=$(yq '.genai_bench.model' "$cfg")
  tokenizer=$(yq '.genai_bench.tokenizer // .genai_bench.model' "$cfg")
  max_req=$(yq '.genai_bench.max_requests_per_run' "$cfg")
  max_time=$(yq '.genai_bench.max_time_per_run' "$cfg")
  local prefix_len; prefix_len=$(yq '.genai_bench.prefix_len // ""' "$cfg")
  local scenarios; scenarios=$(yq '.genai_bench.traffic_scenarios[]' "$cfg")
  local concurrencies; concurrencies=$(yq '.genai_bench.num_concurrency[]' "$cfg")
  local scrape_interval; scrape_interval=$(yq '.ic_metrics.scrape_interval_s // 10' "$cfg")

  # ---- snapshot CRDs (for the diff in `compare`) ----
  color_g "[2/6] Snapshotting CRDs from namespace $WORKLOAD_NAMESPACE"
  kubectl -n "$WORKLOAD_NAMESPACE" get cachepolicy,cachebackend,cacheindex -o yaml \
    > "$outdir/crd-snapshot.yaml" 2>/dev/null || color_y "  (couldn't snapshot CRDs — skipping)"

  # ---- pick the target URL for this mode ----
  local target_url
  case "$mode" in
    baseline) target_url="$VLLM_BASELINE_URL" ;;
    no-hint)  target_url="$VLLM_ENGINE_URL"   ;;
    lookup)
      color_g "[3/6] Starting LookupRoute proxy on :$LOOKUP_PROXY_PORT"
      PYTHONPATH="$PROTO_DIR" python3 "$LIB_DIR/lookup_proxy.py" \
        --listen "0.0.0.0:$LOOKUP_PROXY_PORT" \
        --ic-server "$IC_SERVER_GRPC" \
        --upstream "$VLLM_ENGINE_URL" \
        --log "$outdir/lookup_proxy.log" &
      PROXY_PID=$!
      sleep 2
      kill -0 "$PROXY_PID" 2>/dev/null || die "lookup proxy failed to start; see $outdir/lookup_proxy.log"
      target_url="http://localhost:$LOOKUP_PROXY_PORT"
      ;;
    *) die "unknown mode: $mode (use baseline | no-hint | lookup)";;
  esac

  # ---- start ic-metrics scraper in the background ----
  color_g "[4/6] Starting IC metrics scraper (every ${scrape_interval}s)"
  python3 "$LIB_DIR/collect_ic_metrics.py" \
    --endpoint "$IC_SERVER_METRICS" \
    --interval "$scrape_interval" \
    --output "$outdir/ic-metrics.csv" &
  SCRAPER_PID=$!

  # ---- run genai-bench ----
  color_g "[5/6] Running genai-bench"
  local gb_args=(
    "benchmark"
    "--api-backend" "openai"
    "--api-base"    "$target_url"
    "--api-key"     "${API_KEY:-dummy}"
    "--api-model-name" "$model"
    "--model-tokenizer" "$tokenizer"
    "--task" "$task"
    "--max-requests-per-run" "$max_req"
    "--max-time-per-run"     "$max_time"
    "--server-engine" "vLLM"
    "--experiment-folder-name" "${outdir}/genai-bench"
  )
  for s in $scenarios; do gb_args+=("--traffic-scenario" "$s"); done
  for c in $concurrencies; do gb_args+=("--num-concurrency" "$c"); done
  [[ -n "$prefix_len" ]] && gb_args+=("--prefix-len" "$prefix_len")

  if ! genai-bench "${gb_args[@]}" 2>&1 | tee "$outdir/genai-bench.log"; then
    color_r "genai-bench failed; logs at $outdir/genai-bench.log"
    [[ -n "${PROXY_PID:-}" ]] && kill "$PROXY_PID" 2>/dev/null
    kill "$SCRAPER_PID" 2>/dev/null
    exit 1
  fi

  [[ -n "${PROXY_PID:-}" ]] && kill "$PROXY_PID" 2>/dev/null
  sleep 2
  kill "$SCRAPER_PID" 2>/dev/null

  color_g "[6/6] Building report"
  python3 "$LIB_DIR/correlate.py" \
    --scenario "$cfg" \
    --label "$label" \
    --mode "$mode" \
    --rundir "$outdir" \
    > "$outdir/report.md"

  color_g "Done.  Report: $outdir/report.md"
  echo
  head -40 "$outdir/report.md"
}

cmd_compare() {
  [[ $# -lt 2 ]] && die "usage: compare <label1> <label2> [<label3>]"
  local out="$RESULTS_DIR/compare-$(IFS=-; echo "$*")-$(printf '%(%Y%m%d-%H%M%S)T' -1).md"
  python3 "$LIB_DIR/correlate.py" --compare --results-dir "$RESULTS_DIR" --labels "$@" > "$out"
  color_g "Comparison: $out"
  echo
  head -40 "$out"
}

case "${1:-}" in
  run)             shift; cmd_run "$@" ;;
  compare)         shift; cmd_compare "$@" ;;
  list-scenarios)  cmd_list_scenarios ;;
  clean)           shift; cmd_clean "$@" ;;
  ""|-h|--help)    usage 0 ;;
  *)               die "unknown subcommand: $1 (run | compare | list-scenarios | clean)" ;;
esac
