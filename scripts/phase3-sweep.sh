#!/usr/bin/env bash
# phase3-sweep.sh — orchestrate the Phase 3 benchmark portfolio.
#
# Runs 3 scenarios × 3 modes × 4 iters = 36 benchmark invocations. Between
# "cold" iters it restarts the relevant deployments and waits for Ready so
# each cold run starts from an empty cache. Between warm iters it just kills
# stale processes and re-runs the same scenario to capture warmup → steady-
# state behaviour.
#
# Usage:
#   scripts/phase3-sweep.sh [--scenarios "a b c"] [--modes "baseline lookup"]
#                           [--iters "cold warm-1"] [--results-dir DIR]
#                           [--skip-cold-reset] [--dry-run]
#                           [--compare-only]
#
# Defaults run the full matrix. Flags exist so you can rerun a subset after a
# failed batch without rerunning everything from scratch.
#
# Env vars consumed (passed through to run_tuning_bench.sh):
#   LOOKUP_PROXY_REPLICAS, LOOKUP_PROXY_TOKENIZER, IC_SERVER_*,
#   VLLM_BASELINE_URL, VLLM_METRICS_ENDPOINTS, …
#
# Env vars specific to the orchestrator:
#   IC_NAMESPACE              namespace for ic-smoke deployments (default: ic-smoke)
#   BASELINE_NAMESPACE        namespace for gpu-baseline (default: gpu-baseline)
#   BASELINE_DEPLOYMENT       name of vanilla-vLLM deployment (default: vllm-baseline)
#   LMCACHE_DEPLOYMENT        name of LMCache server deployment (default: lm-smoke)
#   VLLM_ENGINE_DEPLOYMENT    name of cache-enabled vLLM deployment (default: vllm-engine)
#   ROLLOUT_TIMEOUT           kubectl rollout status timeout (default: 10m)
#   STALE_PROC_PATTERNS       extra pgrep patterns to clean between iters
#                              (default: dumb_gateway_client.py|lookup_proxy_legacy.py|
#                                        collect_ic_metrics.py|collect_vllm_metrics.py|genai-bench)

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_BENCH="$ROOT/run_tuning_bench.sh"

# -------- defaults --------
DEFAULT_SCENARIOS="rag-multi-context cache-stress-extreme perfect-storm-rag"
DEFAULT_MODES="baseline no-hint lookup"
DEFAULT_ITERS="cold warm-1 warm-2 warm-3"

: "${IC_NAMESPACE:=ic-smoke}"
: "${BASELINE_NAMESPACE:=gpu-baseline}"
: "${BASELINE_DEPLOYMENT:=vllm-baseline}"
: "${LMCACHE_DEPLOYMENT:=lm-smoke}"
: "${VLLM_ENGINE_DEPLOYMENT:=vllm-engine}"
: "${ROLLOUT_TIMEOUT:=10m}"
: "${STALE_PROC_PATTERNS:=dumb_gateway_client.py|lookup_proxy_legacy.py|collect_ic_metrics.py|collect_vllm_metrics.py|genai-bench}"

SCENARIOS="$DEFAULT_SCENARIOS"
MODES="$DEFAULT_MODES"
ITERS="$DEFAULT_ITERS"
DRY_RUN=0
SKIP_COLD_RESET=0
COMPARE_ONLY=0
RESULTS_DIR="$ROOT/results"

# -------- pretty-print --------
color_g() { printf '\033[32m%s\033[0m\n' "$*"; }
color_y() { printf '\033[33m%s\033[0m\n' "$*"; }
color_r() { printf '\033[31m%s\033[0m\n' "$*"; }
die()     { color_r "$*"; exit 1; }

run() {
  # Echo + run; respects --dry-run.
  echo "  $*"
  if [[ $DRY_RUN -eq 0 ]]; then
    "$@"
  fi
}

# -------- arg parsing --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenarios)         SCENARIOS="$2"; shift 2 ;;
    --modes)             MODES="$2";     shift 2 ;;
    --iters)             ITERS="$2";     shift 2 ;;
    --results-dir)       RESULTS_DIR="$2"; shift 2 ;;
    --dry-run)           DRY_RUN=1; shift ;;
    --skip-cold-reset)   SKIP_COLD_RESET=1; shift ;;
    --compare-only)      COMPARE_ONLY=1; shift ;;
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) die "unknown arg: $1" ;;
  esac
done

# -------- cold reset --------
# baseline → restart vllm-baseline; cache-plane modes → restart LMCache + vllm-engine.
# In either case we wait for the relevant deployments to become Ready before
# proceeding. PF refresher (CAC-162) should restore port-forwards as pods come
# back; we log if any are still dead.
cold_reset() {
  local mode="$1"
  if [[ $SKIP_COLD_RESET -eq 1 ]]; then
    color_y "  (--skip-cold-reset → not restarting deployments)"
    return 0
  fi
  case "$mode" in
    baseline)
      color_g "  cold-reset baseline → rollout restart $BASELINE_NAMESPACE/$BASELINE_DEPLOYMENT"
      run kubectl -n "$BASELINE_NAMESPACE" rollout restart "deployment/$BASELINE_DEPLOYMENT"
      run kubectl -n "$BASELINE_NAMESPACE" rollout status "deployment/$BASELINE_DEPLOYMENT" --timeout="$ROLLOUT_TIMEOUT"
      ;;
    no-hint|lookup)
      color_g "  cold-reset $mode → rollout restart LMCache + vllm-engine in $IC_NAMESPACE"
      # LMCache first so vllm-engine reconnects to a fresh server.
      run kubectl -n "$IC_NAMESPACE" rollout restart "deployment/$LMCACHE_DEPLOYMENT"
      run kubectl -n "$IC_NAMESPACE" rollout status "deployment/$LMCACHE_DEPLOYMENT" --timeout="$ROLLOUT_TIMEOUT"
      run kubectl -n "$IC_NAMESPACE" rollout restart "deployment/$VLLM_ENGINE_DEPLOYMENT"
      run kubectl -n "$IC_NAMESPACE" rollout status "deployment/$VLLM_ENGINE_DEPLOYMENT" --timeout="$ROLLOUT_TIMEOUT"
      ;;
    *)
      color_y "  cold-reset: unknown mode '$mode' — skipping"
      return 0
      ;;
  esac
  # Give the port-forward refresher (CAC-162) a window to reattach to the
  # restored pods before we kick off the bench.
  run sleep 15
  verify_port_forwards "$mode"
}

# verify_port_forwards: best-effort smoke check that PFs are reachable. Logs
# loudly if anything looks dead. Never aborts — the bench harness will fail
# fast on its own if it really can't reach the engine.
verify_port_forwards() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  (verify_port_forwards skipped in dry-run)"
    return 0
  fi
  local mode="$1"
  case "$mode" in
    baseline)
      if ! curl -fsS -o /dev/null --connect-timeout 3 "${VLLM_BASELINE_URL:-http://localhost:38005}/v1/models"; then
        color_y "  WARNING: baseline vLLM PF appears dead (${VLLM_BASELINE_URL:-http://localhost:38005})"
      fi
      ;;
    no-hint|lookup)
      if [[ -n "${LOOKUP_PROXY_REPLICAS:-}" ]]; then
        IFS=',' read -ra _reps <<< "$LOOKUP_PROXY_REPLICAS"
        for r in "${_reps[@]}"; do
          local rid http_url
          rid=$(awk -F'|' '{print $1}' <<< "$r")
          http_url=$(awk -F'|' '{print $3}' <<< "$r")
          if [[ -n "$http_url" ]] && \
             ! curl -fsS -o /dev/null --connect-timeout 3 "${http_url%/}/v1/models"; then
            color_y "  WARNING: replica $rid PF appears dead ($http_url)"
          fi
        done
      fi
      ;;
  esac
}

# kill_stale: SIGTERM any harness-spawned children left over from a previous
# iteration. The runner kills them too, but a Ctrl-C in the middle leaves
# things lying around.
kill_stale() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  pkill -f \"$STALE_PROC_PATTERNS\"  # (dry-run)"
    return 0
  fi
  if pkill -f "$STALE_PROC_PATTERNS" 2>/dev/null; then
    color_y "  killed stale processes matching: $STALE_PROC_PATTERNS"
    sleep 2
  fi
}

# -------- run one bench --------
run_one() {
  local scenario="$1" mode="$2" iter="$3"
  local label="phase3-${scenario}-${mode}-${iter}"
  color_g "==== ${scenario} / ${mode} / ${iter}  (label=${label})"
  if [[ "$iter" == "cold" ]]; then
    cold_reset "$mode"
  fi
  kill_stale
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  $RUN_BENCH run --scenario $scenario --label $label --mode $mode  # (dry-run)"
    return 0
  fi
  if ! "$RUN_BENCH" run --scenario "$scenario" --label "$label" --mode "$mode"; then
    color_r "  bench failed for label=$label — continuing with next iter"
    return 1
  fi
}

# -------- main matrix --------
if [[ $COMPARE_ONLY -eq 0 ]]; then
  color_g "Phase 3 sweep starting"
  color_g "  scenarios: $SCENARIOS"
  color_g "  modes:     $MODES"
  color_g "  iters:     $ITERS"
  color_g "  dry-run:   $DRY_RUN     skip-cold-reset: $SKIP_COLD_RESET"
  echo

  rc=0
  for scenario in $SCENARIOS; do
    for mode in $MODES; do
      for iter in $ITERS; do
        if ! run_one "$scenario" "$mode" "$iter"; then
          rc=1
        fi
      done
    done
  done

  if [[ $rc -ne 0 ]]; then
    color_r "One or more runs failed. Check $RESULTS_DIR/phase3-*-*-*/ for partial artifacts."
  fi
fi

# -------- comparison reports --------
# For each scenario, build a baseline/no-hint/lookup three-way comparison
# off the cold iter (the headline numbers). Warm-* iters are still on disk
# for tail-analysis but don't drive the headline gate.
color_g
color_g "Building per-scenario comparison reports"
for scenario in $SCENARIOS; do
  labels=()
  for mode in $MODES; do
    # `compare` resolves "<label>" to the latest matching results dir, so a
    # bare prefix without a timestamp is fine. We pick the "cold" iter as
    # the headline; warm iters live alongside for tail-analysis.
    labels+=("phase3-${scenario}-${mode}-cold")
  done
  echo "  $RUN_BENCH compare ${labels[*]}"
  if [[ $DRY_RUN -eq 0 ]]; then
    "$RUN_BENCH" compare "${labels[@]}" \
      || color_y "  compare failed for $scenario (some labels probably missing)"
  fi
done

color_g "Phase 3 sweep done."
