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
#   IC_SERVER_METRICS       — server /metrics URL                (default: http://localhost:38001/metrics)
#   IC_SERVER_GRPC          — server gRPC endpoint               (default: localhost:38002)
#   VLLM_ENGINE_URL         — cache-enabled vLLM HTTP URL        (default: http://localhost:38000)
#   VLLM_BASELINE_URL       — vanilla-vLLM (no cache plane) URL  (default: http://localhost:38005)
#   LOOKUP_PROXY_PORT       — client listen port (lookup/no-hint mode) (default: 18100)
#   LOOKUP_PROXY_TOKENIZER  — HF tokenizer id (lookup mode)             (default: unsloth/Meta-Llama-3.1-8B-Instruct)
#   LOOKUP_PROXY_REPLICAS   — per-replica config (lookup/no-hint mode)  — see README; required for both modes
#                             Format per replica: id|<zmq>|http_url[|router]
#                             (the zmq + router fields are parsed for legacy
#                              backward compat with the chain-walking proxy and
#                              ignored by the dumb client — only id + http_url
#                              are used)
#   USE_LEGACY_PROXY        — set to 1 to invoke the deprecated lookup_proxy_legacy.py
#                             instead of the new dumb_gateway_client.py (default: 0).
#                             Kept for one release so operators can A/B compare
#                             the chain-walking proxy with the production model.
#   KUBECONFIG              — kubeconfig path for CRD snapshot   (default: from environment)
#   CLUSTER_STATE_TARGETS   — pods to snapshot for cluster-state.yaml (CAC-161)
#                             Comma-separated <namespace>:<name-prefix> pairs.
#                             (default: ic-smoke:vllm-engine,ic-smoke:lm-smoke,gpu-baseline:vllm-baseline)
#   CLUSTER_STATE_EVENTS_NS — namespaces to pull pod events from for cluster-state.yaml
#                             (default: unique namespaces in CLUSTER_STATE_TARGETS)
#   VLLM_METRICS_ENDPOINTS  — per-pod vLLM /metrics URLs for the Phase 3 scraper
#                             Comma-separated <id>=<url> pairs. (default: derived
#                             per mode — baseline=VLLM_BASELINE_URL/metrics for
#                             baseline mode, the LOOKUP_PROXY_REPLICAS http urls
#                             for no-hint/lookup modes.) Set explicitly to override.
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
: "${LOOKUP_PROXY_TOKENIZER:=unsloth/Meta-Llama-3.1-8B-Instruct}"
: "${USE_LEGACY_PROXY:=0}"
: "${WORKLOAD_NAMESPACE:=default}"
# LOOKUP_PROXY_REPLICAS — comma-separated, one entry per replica.
# Within a replica, use "|" as the field separator (colons are ambiguous in URLs).
# Format per replica: <id>|<zmq_endpoint>|<http_url>
# Example:
#   "r0|tcp://localhost:15001|http://localhost:38010,r1|tcp://localhost:15002|http://localhost:38011"
: "${LOOKUP_PROXY_REPLICAS:=}"
: "${CLUSTER_STATE_TARGETS:=ic-smoke:vllm-engine,ic-smoke:lm-smoke,gpu-baseline:vllm-baseline}"
: "${CLUSTER_STATE_EVENTS_NS:=}"
: "${VLLM_METRICS_ENDPOINTS:=}"
: "${VLLM_METRICS_INTERVAL:=30}"

# -------- helpers --------
color_g() { printf '\033[32m%s\033[0m\n' "$*"; }
color_y() { printf '\033[33m%s\033[0m\n' "$*"; }
color_r() { printf '\033[31m%s\033[0m\n' "$*"; }
die()     { color_r "$*"; exit 1; }

usage() {
  sed -n '2,24p' "$0" | sed 's/^# \?//'
  exit "${1:-0}"
}

# Parse $CLUSTER_STATE_TARGETS (comma-separated NS:PREFIX entries) into a
# repeated --target argv array. Writes result to the caller-named array.
build_cluster_state_targets() {
  local _outvar="$1" _val="$2"
  local -a _args=()
  IFS=',' read -ra _entries <<< "$_val"
  for e in "${_entries[@]}"; do
    e="${e#"${e%%[![:space:]]*}"}"  # ltrim
    e="${e%"${e##*[![:space:]]}"}"  # rtrim
    [[ -z "$e" ]] && continue
    _args+=("--target" "$e")
  done
  # shellcheck disable=SC2034
  eval "$_outvar=(\"\${_args[@]}\")"
}

# Compute the unique set of namespaces in CLUSTER_STATE_TARGETS (used when
# CLUSTER_STATE_EVENTS_NS is unset). Writes to the caller-named array. Avoids
# `declare -A` so it works under macOS's bash 3.2.
build_cluster_state_event_ns() {
  local _outvar="$1" _val="$2"
  local -a _args=()
  local _seen=" "  # space-bracketed list, used for substring containment
  IFS=',' read -ra _entries <<< "$_val"
  for e in "${_entries[@]}"; do
    local ns="${e%%:*}"
    ns="${ns#"${ns%%[![:space:]]*}"}"
    ns="${ns%"${ns##*[![:space:]]}"}"
    [[ -z "$ns" ]] && continue
    case "$_seen" in
      *" $ns "*) ;;
      *) _args+=("--events-namespace" "$ns"); _seen+="$ns " ;;
    esac
  done
  # shellcheck disable=SC2034
  eval "$_outvar=(\"\${_args[@]}\")"
}

cmd_list_scenarios() {
  echo "Available scenarios in $SCENARIOS_DIR:"
  for f in "$SCENARIOS_DIR"/*.yaml; do
    name=$(basename "$f" .yaml)
    desc=$(yq -r '.description // ""' "$f" | head -1)
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

  local ts="$(date +%Y%m%d-%H%M%S)"
  local outdir="$RESULTS_DIR/${label}-${ts}"
  mkdir -p "$outdir/genai-bench"

  color_g "[1/7] Scenario: $scenario   label: $label   mode: $mode"
  color_g "[1/7] Output:   $outdir"
  cp "$cfg" "$outdir/scenario.yaml"

  # ---- parse scenario YAML ----
  local task model tokenizer max_req max_time
  task=$(yq -r '.genai_bench.task' "$cfg")
  model=$(yq -r '.genai_bench.model' "$cfg")
  tokenizer=$(yq -r '.genai_bench.tokenizer // .genai_bench.model' "$cfg")
  max_req=$(yq -r '.genai_bench.max_requests_per_run' "$cfg")
  max_time=$(yq -r '.genai_bench.max_time_per_run' "$cfg")
  local prefix_len; prefix_len=$(yq -r '.genai_bench.prefix_len // ""' "$cfg")
  local scenarios; scenarios=$(yq -r '.genai_bench.traffic_scenarios[]?' "$cfg")
  local concurrencies; concurrencies=$(yq -r '.genai_bench.num_concurrency[]' "$cfg")
  local scrape_interval; scrape_interval=$(yq -r '.ic_metrics.scrape_interval_s // 10' "$cfg")
  # Dataset mode (mutually exclusive with traffic_scenarios + prefix_len).
  # Paths are resolved relative to the harness root.
  local dataset_path_rel; dataset_path_rel=$(yq -r '.genai_bench.dataset_path // ""' "$cfg")
  local dataset_path="$dataset_path_rel"
  if [[ -n "$dataset_path" && "$dataset_path" != /* ]]; then
    dataset_path="$ROOT/$dataset_path"
  fi
  if [[ -n "$dataset_path" && ! -f "$dataset_path" ]]; then
    die "scenario $scenario references dataset_path=$dataset_path but the file is missing. \
Generate it first (see the scenario's description for the generator command)."
  fi

  # Per-run dataset metadata (CAC-159) — write before the bench starts so even
  # a failed run leaves a record of which dataset was active.
  if [[ -n "$dataset_path" ]]; then
    python3 "$LIB_DIR/write_dataset_meta.py" \
      --dataset "$dataset_path" \
      --dataset-rel "$dataset_path_rel" \
      --scenario "$scenario" \
      --outdir "$outdir" \
      || color_y "  (dataset metadata writer failed — continuing)"
  fi

  # ---- snapshot CRDs (for the diff in `compare`) ----
  color_g "[2/7] Snapshotting CRDs from namespace $WORKLOAD_NAMESPACE"
  kubectl -n "$WORKLOAD_NAMESPACE" get cachepolicy,cachebackend,cacheindex -o yaml \
    > "$outdir/crd-snapshot.yaml" 2>/dev/null || color_y "  (couldn't snapshot CRDs — skipping)"

  # ---- cluster-state: pod snapshot at run start (CAC-161) ----
  # Captures pod names, restart counts, ages, and UIDs for the configured
  # targets so post-mortem reviewers can tell whether a pod was deleted+
  # recreated or container-restarted mid-run. Best-effort — never fatal.
  color_g "[3/7] Snapshotting cluster state (start): $CLUSTER_STATE_TARGETS"
  local -a CLUSTER_STATE_TARGET_ARGS=()
  build_cluster_state_targets CLUSTER_STATE_TARGET_ARGS "$CLUSTER_STATE_TARGETS"
  local run_start_epoch; run_start_epoch=$(date +%s)
  python3 "$LIB_DIR/cluster_state.py" snapshot \
    "${CLUSTER_STATE_TARGET_ARGS[@]}" \
    --output "$outdir/.cluster-state-start.json" \
    || color_y "  (cluster-state start snapshot failed — continuing)"

  # ---- pick the target URL for this mode ----
  # no-hint and lookup both route through the dumb gateway client so
  # genai-bench's traffic spreads across all configured replicas. The mode
  # selects --routing-mode:
  #   lookup       → call LookupRoute and honor PREFIX_MATCH; round-robin
  #                  fallback on NO_HINT / TIMEOUT / TENANT_HOT (CAC-154).
  #   no-hint      → --routing-mode=round-robin, skipping the RPC entirely
  #                  (CAC-153 — cache plane up, routing layer disabled).
  # baseline points genai-bench at vanilla vLLM directly — no client at all.
  local target_url
  case "$mode" in
    baseline) target_url="$VLLM_BASELINE_URL" ;;
    no-hint|lookup)
      # USE_LEGACY_PROXY=1 lets operators A/B compare the deprecated
      # chain-walking proxy with the production dumb-gateway model during
      # the one-release deprecation window (CAC-152).
      local client_script="$LIB_DIR/dumb_gateway_client.py"
      local client_label="dumb_gateway_client"
      if [[ "$USE_LEGACY_PROXY" == "1" ]]; then
        client_script="$LIB_DIR/lookup_proxy_legacy.py"
        client_label="lookup_proxy_legacy"
        color_y "  USE_LEGACY_PROXY=1 — running the deprecated chain-walking proxy"
      fi
      color_g "[4/7] Starting $client_label on :$LOOKUP_PROXY_PORT (mode=$mode)"

      # Build --replica args from LOOKUP_PROXY_REPLICAS (id|zmq|http_url[|router]).
      # The dumb client only uses the id + http_url; ZMQ + router fields are
      # parsed for legacy backward compat and ignored. The legacy proxy
      # needs all four fields, so we pass the spec through verbatim either way.
      local -a replica_args=()
      if [[ -n "$LOOKUP_PROXY_REPLICAS" ]]; then
        IFS=',' read -ra _reps <<< "$LOOKUP_PROXY_REPLICAS"
        for r in "${_reps[@]}"; do
          replica_args+=("--replica" "$r")
        done
      else
        color_y "  WARNING: LOOKUP_PROXY_REPLICAS not set — no replica URLs"
        color_y "  for the client to round-robin across. Both modes will fail"
        color_y "  to start. See README §Setting up lookup mode."
      fi

      # LOOKUP_PROXY_EXTRA_ARGS: free-form extra args appended to the client
      # invocation. Useful for --hash-scheme, --block-size, or any future flag.
      local -a extra_args=()
      if [[ -n "${LOOKUP_PROXY_EXTRA_ARGS:-}" ]]; then
        # shellcheck disable=SC2206
        extra_args=(${LOOKUP_PROXY_EXTRA_ARGS})
      fi

      local routing_mode="lookup"
      [[ "$mode" == "no-hint" ]] && routing_mode="round-robin"

      if [[ "$USE_LEGACY_PROXY" == "1" ]]; then
        # Legacy path: --no-lookup-route + --replicas + --tokenizer.
        local -a fallback_urls=()
        if [[ -n "$LOOKUP_PROXY_REPLICAS" ]]; then
          IFS=',' read -ra _reps <<< "$LOOKUP_PROXY_REPLICAS"
          for r in "${_reps[@]}"; do
            local _http_url; _http_url=$(awk -F'|' '{print $3}' <<< "$r")
            [[ -n "$_http_url" ]] && fallback_urls+=("$_http_url")
          done
        fi
        local replicas_csv
        if (( ${#fallback_urls[@]} > 0 )); then
          replicas_csv=$(IFS=','; echo "${fallback_urls[*]}")
        else
          replicas_csv="$VLLM_ENGINE_URL"
        fi
        local -a legacy_extra=()
        [[ "$mode" == "no-hint" ]] && legacy_extra+=("--no-lookup-route")
        PYTHONPATH="$LIB_DIR:$PROTO_DIR" python3 "$client_script" \
          --listen "0.0.0.0:$LOOKUP_PROXY_PORT" \
          --ic-server "$IC_SERVER_GRPC" \
          --replicas "$replicas_csv" \
          --tokenizer "$LOOKUP_PROXY_TOKENIZER" \
          --tenant "${LOOKUP_PROXY_TENANT:-$WORKLOAD_NAMESPACE}" \
          "${replica_args[@]}" \
          "${legacy_extra[@]}" \
          "${extra_args[@]}" \
          --log "$outdir/${client_label}.log" &
      else
        # New dumb-gateway path: --routing-mode + --replica id=url (or legacy
        # id|zmq|http_url piped form — the parser accepts both).
        PYTHONPATH="$LIB_DIR:$PROTO_DIR" python3 "$client_script" \
          --listen "0.0.0.0:$LOOKUP_PROXY_PORT" \
          --ic-server "$IC_SERVER_GRPC" \
          --tokenizer "$LOOKUP_PROXY_TOKENIZER" \
          --tenant "${LOOKUP_PROXY_TENANT:-$WORKLOAD_NAMESPACE}" \
          --routing-mode "$routing_mode" \
          "${replica_args[@]}" \
          "${extra_args[@]}" \
          --log "$outdir/${client_label}.log" &
      fi
      PROXY_PID=$!
      sleep 3  # tokenizer load takes a moment
      kill -0 "$PROXY_PID" 2>/dev/null || die "$client_label failed to start; see $outdir/${client_label}.log"
      target_url="http://localhost:$LOOKUP_PROXY_PORT"
      ;;
    *) die "unknown mode: $mode (use baseline | no-hint | lookup)";;
  esac

  # ---- start ic-metrics scraper in the background ----
  color_g "[5/7] Starting IC metrics scraper (every ${scrape_interval}s)"
  python3 "$LIB_DIR/collect_ic_metrics.py" \
    --endpoint "$IC_SERVER_METRICS" \
    --interval "$scrape_interval" \
    --output "$outdir/ic-metrics.csv" &
  SCRAPER_PID=$!

  # ---- start per-pod vLLM /metrics scraper in the background (Phase 3) ----
  # Endpoint set defaults from mode + LOOKUP_PROXY_REPLICAS; override entirely
  # via VLLM_METRICS_ENDPOINTS (comma-separated id=url, repeated). The scraper
  # is a no-op if no endpoints can be derived.
  local -a vllm_endpoint_args=()
  if [[ -n "$VLLM_METRICS_ENDPOINTS" ]]; then
    IFS=',' read -ra _veps <<< "$VLLM_METRICS_ENDPOINTS"
    for ep in "${_veps[@]}"; do
      ep="${ep#"${ep%%[![:space:]]*}"}"
      ep="${ep%"${ep##*[![:space:]]}"}"
      [[ -z "$ep" ]] && continue
      vllm_endpoint_args+=("--endpoint" "$ep")
    done
  elif [[ "$mode" == "baseline" ]]; then
    vllm_endpoint_args+=("--endpoint" "baseline=${VLLM_BASELINE_URL%/}/metrics")
  elif [[ -n "$LOOKUP_PROXY_REPLICAS" ]]; then
    IFS=',' read -ra _reps <<< "$LOOKUP_PROXY_REPLICAS"
    for r in "${_reps[@]}"; do
      local _rid _http_url
      _rid=$(awk -F'|' '{print $1}' <<< "$r")
      _http_url=$(awk -F'|' '{print $3}' <<< "$r")
      if [[ -n "$_rid" && -n "$_http_url" ]]; then
        vllm_endpoint_args+=("--endpoint" "${_rid}=${_http_url%/}/metrics")
      fi
    done
  fi
  if (( ${#vllm_endpoint_args[@]} > 0 )); then
    color_g "      vLLM metrics scraper: ${#vllm_endpoint_args[@]} endpoint(s), every ${VLLM_METRICS_INTERVAL}s"
    python3 "$LIB_DIR/collect_vllm_metrics.py" \
      "${vllm_endpoint_args[@]}" \
      --interval "$VLLM_METRICS_INTERVAL" \
      --output "$outdir/vllm-metrics.csv" &
    VLLM_SCRAPER_PID=$!
  else
    color_y "  (no vLLM metrics endpoints derivable — skipping vllm-metrics.csv)"
  fi

  # ---- pre-run per-pod distribution snapshot (CAC-163) ----
  # No-op when LOOKUP_PROXY_REPLICAS is unset. When set, scrapes each pod's
  # vllm:prefix_cache_queries_total; the post-run diff fails loud if any pod
  # received zero traffic while siblings got some — the signature of a
  # `kubectl port-forward svc/...` PF-pinning misconfiguration.
  python3 "$LIB_DIR/check_pod_distribution.py" snapshot \
    --out "$outdir/dist-before.json" \
    || color_y "  (dist-check snapshot failed — continuing)"

  # ---- run genai-bench ----
  color_g "[6/7] Running genai-bench"
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
  if [[ -n "$dataset_path" ]]; then
    gb_args+=("--dataset-path" "$dataset_path")
  else
    for s in $scenarios; do gb_args+=("--traffic-scenario" "$s"); done
    [[ -n "$prefix_len" ]] && gb_args+=("--prefix-len" "$prefix_len")
  fi
  for c in $concurrencies; do gb_args+=("--num-concurrency" "$c"); done

  local bench_rc=0
  if ! genai-bench "${gb_args[@]}" 2>&1 | tee "$outdir/genai-bench.log"; then
    color_r "genai-bench failed; logs at $outdir/genai-bench.log"
    bench_rc=1
  fi

  [[ -n "${PROXY_PID:-}" ]] && kill "$PROXY_PID" 2>/dev/null
  sleep 2
  kill "$SCRAPER_PID" 2>/dev/null
  [[ -n "${VLLM_SCRAPER_PID:-}" ]] && kill "$VLLM_SCRAPER_PID" 2>/dev/null

  # ---- post-run distribution check (CAC-163) ----
  # Exit-code semantics: 0 = ok/skipped/idle, 2 = imbalanced. We capture the
  # rc but don't abort the run — the report.md still has value, and we want
  # the warning printed prominently at the end of the run.
  set +e
  python3 "$LIB_DIR/check_pod_distribution.py" diff \
    --before "$outdir/dist-before.json" \
    --out    "$outdir/dist-report.json"
  DIST_RC=$?
  set -e 2>/dev/null || true

  # ---- cluster-state: pod snapshot at run end + finalize (CAC-161) ----
  # Runs even on a failed bench — that's the case we most want pod state for.
  color_g "[7/7] Snapshotting cluster state (end) + finalizing"
  python3 "$LIB_DIR/cluster_state.py" snapshot \
    "${CLUSTER_STATE_TARGET_ARGS[@]}" \
    --output "$outdir/.cluster-state-end.json" \
    || color_y "  (cluster-state end snapshot failed — continuing)"
  local -a CLUSTER_STATE_EVENT_ARGS=()
  if [[ -n "$CLUSTER_STATE_EVENTS_NS" ]]; then
    IFS=',' read -ra _ens <<< "$CLUSTER_STATE_EVENTS_NS"
    for ns in "${_ens[@]}"; do CLUSTER_STATE_EVENT_ARGS+=("--events-namespace" "$ns"); done
  else
    build_cluster_state_event_ns CLUSTER_STATE_EVENT_ARGS "$CLUSTER_STATE_TARGETS"
  fi
  python3 "$LIB_DIR/cluster_state.py" finalize \
    --start "$outdir/.cluster-state-start.json" \
    --end "$outdir/.cluster-state-end.json" \
    --run-start-epoch "$run_start_epoch" \
    "${CLUSTER_STATE_EVENT_ARGS[@]}" \
    --output "$outdir/cluster-state.yaml" \
    || color_y "  (cluster-state finalize failed — continuing)"

  if [[ "$bench_rc" -ne 0 ]]; then
    exit "$bench_rc"
  fi

  color_g "Building report"
  python3 "$LIB_DIR/correlate.py" \
    --scenario "$cfg" \
    --label "$label" \
    --mode "$mode" \
    --rundir "$outdir" \
    > "$outdir/report.md"

  color_g "Done.  Report: $outdir/report.md"
  if [[ "${DIST_RC:-0}" == "2" ]]; then
    color_r "WARNING: per-pod distribution check flagged imbalance — see $outdir/dist-report.json"
    color_r "         (likely cause: kubectl port-forward svc/... pins to one pod; see README Prerequisites)"
  fi
  echo
  head -40 "$outdir/report.md"
}

cmd_compare() {
  [[ $# -lt 2 ]] && die "usage: compare <label1> <label2> [<label3>]"
  local out="$RESULTS_DIR/compare-$(IFS=-; echo "$*")-$(date +%Y%m%d-%H%M%S).md"
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
