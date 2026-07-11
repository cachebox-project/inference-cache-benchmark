#!/usr/bin/env bash
# phase3-sweep.sh — orchestrate the Phase 3 benchmark portfolio.
#
# Runs 3 scenarios × 3 modes × 4 iters = 36 benchmark invocations. Before each
# scenario/mode group it restarts the relevant deployment(s), forces configured
# port-forwards to reconnect, and runs a tiny warm-start before the first
# measured iteration. Warm iterations of the same mode intentionally reuse the
# state from the previous iteration to capture warmup → steady-state behaviour.
#
# Usage:
#   scripts/phase3-sweep.sh [--scenarios "a b c"] [--modes "baseline lookup"]
#                           [--iters "cold warm-1"] [--results-dir DIR]
#                           [--skip-mode-reset] [--dry-run]
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
#   PF_REFRESHER_CONFIG       config for oci-session-refresher.sh (default:
#                             $REFRESHER_CONFIG or ~/.oci-session-refresher.conf)
#   PF_RESET_AFTER_ROLLOUT    force-restart configured PFs after rollouts (default: 1)
#   PF_SETTLE_SECONDS         seconds to wait after PF refresh before warmup (default: 15)
#   MODE_WARMUP_REQUESTS      tiny warm-start request count after mode reset (default: 5)
#   MODE_WARMUP_CONCURRENCY   concurrency for the warm-start (default: 1)
#   MODE_WARMUP_MAX_TIME      max seconds for the warm-start (default: 120)
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
: "${PF_REFRESHER_CONFIG:=${REFRESHER_CONFIG:-$HOME/.oci-session-refresher.conf}}"
: "${PF_RESET_AFTER_ROLLOUT:=1}"
: "${PF_SETTLE_SECONDS:=15}"
: "${MODE_WARMUP_REQUESTS:=5}"
: "${MODE_WARMUP_CONCURRENCY:=1}"
: "${MODE_WARMUP_MAX_TIME:=120}"
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
    --skip-cold-reset|--skip-mode-reset) SKIP_COLD_RESET=1; shift ;;
    --compare-only)      COMPARE_ONLY=1; shift ;;
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) die "unknown arg: $1" ;;
  esac
done

# -------- mode-boundary reset --------
# baseline → restart vllm-baseline; cache-plane modes → restart LMCache + vllm-engine.
# In either case we wait for the relevant deployments to become Ready, force the
# configured port-forwards to reconnect, then warm-start before measurement.
mode_reset() {
  local mode="$1"
  if [[ $SKIP_COLD_RESET -eq 1 ]]; then
    color_y "  (--skip-mode-reset → not restarting deployments or port-forwards)"
    return 0
  fi
  case "$mode" in
    baseline)
      color_g "  mode-reset baseline → rollout restart $BASELINE_NAMESPACE/$BASELINE_DEPLOYMENT"
      run kubectl -n "$BASELINE_NAMESPACE" rollout restart "deployment/$BASELINE_DEPLOYMENT"
      run kubectl -n "$BASELINE_NAMESPACE" rollout status "deployment/$BASELINE_DEPLOYMENT" --timeout="$ROLLOUT_TIMEOUT"
      ;;
    no-hint|lookup)
      color_g "  mode-reset $mode → rollout restart LMCache + vllm-engine in $IC_NAMESPACE"
      # LMCache first so vllm-engine reconnects to a fresh server.
      run kubectl -n "$IC_NAMESPACE" rollout restart "deployment/$LMCACHE_DEPLOYMENT"
      run kubectl -n "$IC_NAMESPACE" rollout status "deployment/$LMCACHE_DEPLOYMENT" --timeout="$ROLLOUT_TIMEOUT"
      run kubectl -n "$IC_NAMESPACE" rollout restart "deployment/$VLLM_ENGINE_DEPLOYMENT"
      run kubectl -n "$IC_NAMESPACE" rollout status "deployment/$VLLM_ENGINE_DEPLOYMENT" --timeout="$ROLLOUT_TIMEOUT"
      ;;
    *)
      color_y "  mode-reset: unknown mode '$mode' — skipping"
      return 0
      ;;
  esac
  refresh_port_forwards "$mode"
}

refresh_port_forwards() {
  local mode="$1"
  if [[ "$PF_RESET_AFTER_ROLLOUT" != "1" ]]; then
    color_y "  PF_RESET_AFTER_ROLLOUT=$PF_RESET_AFTER_ROLLOUT → not forcing PF restart"
    verify_port_forwards "$mode"
    return 0
  fi
  if [[ ! -f "$PF_REFRESHER_CONFIG" ]]; then
    color_y "  no PF_REFRESHER_CONFIG at $PF_REFRESHER_CONFIG — cannot force PF restart"
    verify_port_forwards "$mode"
    return 0
  fi
  color_g "  force-refreshing port-forwards via $PF_REFRESHER_CONFIG"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  FORCE_RESTART_PFS=1 $ROOT/scripts/oci-session-refresher.sh --once $PF_REFRESHER_CONFIG  # (dry-run)"
  else
    FORCE_RESTART_PFS=1 "$ROOT/scripts/oci-session-refresher.sh" --once "$PF_REFRESHER_CONFIG" \
      || color_y "  WARNING: forced PF refresh failed"
  fi
  run sleep "$PF_SETTLE_SECONDS"
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
  local scenario="$1" mode="$2" iter="$3" warmup_requests="${4:-0}"
  local label="phase3-${scenario}-${mode}-${iter}"
  color_g "==== ${scenario} / ${mode} / ${iter}  (label=${label})"
  kill_stale
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  BENCH_WARMUP_REQUESTS=$warmup_requests BENCH_WARMUP_CONCURRENCY=$MODE_WARMUP_CONCURRENCY BENCH_WARMUP_MAX_TIME=$MODE_WARMUP_MAX_TIME $RUN_BENCH run --scenario $scenario --label $label --mode $mode  # (dry-run)"
    return 0
  fi
  if ! BENCH_WARMUP_REQUESTS="$warmup_requests" \
       BENCH_WARMUP_CONCURRENCY="$MODE_WARMUP_CONCURRENCY" \
       BENCH_WARMUP_MAX_TIME="$MODE_WARMUP_MAX_TIME" \
       "$RUN_BENCH" run --scenario "$scenario" --label "$label" --mode "$mode"; then
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
  color_g "  dry-run:   $DRY_RUN     skip-mode-reset: $SKIP_COLD_RESET"
  echo

  rc=0
  for scenario in $SCENARIOS; do
    for mode in $MODES; do
      mode_reset "$mode"
      warmup_requests=0
      if [[ $SKIP_COLD_RESET -ne 1 ]]; then
        warmup_requests="$MODE_WARMUP_REQUESTS"
      fi
      for iter in $ITERS; do
        if ! run_one "$scenario" "$mode" "$iter" "$warmup_requests"; then
          rc=1
        fi
        warmup_requests=0
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
