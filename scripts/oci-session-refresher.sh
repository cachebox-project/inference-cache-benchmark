#!/usr/bin/env bash
# OCI session refresher + kubectl port-forward health monitor.
#
# Every $INTERVAL seconds:
#   1. Refresh the OCI session token (so kubectl can still talk to OKE).
#   2. For each declared port-forward, verify a process is LISTENing on the
#      local port. If not, kill any stale kubectl process for that port and
#      re-spawn the port-forward.
#   3. Log a summary line per tick:
#        "tick OK — PFs alive: N/M"            (steady state)
#        "tick OK — PFs alive: N/M — restored: 38010 15001"   (after restore)
#
# Why this exists: kubectl port-forwards die silently for many reasons
# (server-side timeouts, VPN blips, OKE API restarts). The original refresher
# only kept auth alive, so benchmarks would later fail with connection-refused
# because the PFs underneath had quietly died. See CAC-162.
#
# Usage:
#   oci-session-refresher.sh [config-file]
#
# Config file is sourced as bash. It can set:
#   PROFILE        OCI session profile      (default: BoatOc1)
#   INTERVAL       seconds between ticks    (default: 1800 = 30 min)
#   KUBECONFIG     kubeconfig path          (default: env, else kubectl default)
#   PF_LOG_DIR     where per-port kubectl logs go    (default: /tmp)
#   PF_SPECS       array of "port|kubectl-args" entries (see below)
#   resolve_pf_specs()   optional shell function called each tick to rebuild
#                        PF_SPECS dynamically (e.g., for pod-targeted PFs whose
#                        pod names change after restarts).
#
# A PF_SPECS entry is "<local_port>|<kubectl args>", where the kubectl args
# are everything EXCEPT --kubeconfig and the local-port half of the mapping.
# The remote port appears bare as ":<remote_port>"; the script rewrites it to
# "<local>:<remote>" at exec time. Example:
#
#   PF_SPECS=(
#     "38001|-n inference-cache port-forward svc/inference-cache-server :8080"
#     "38002|-n inference-cache port-forward svc/inference-cache-server :9090"
#     "38005|-n gpu-baseline    port-forward svc/vllm-baseline          :8000"
#   )
#
# A missing or empty config means "session refresh only" — backward-compatible
# with the original /tmp/oci-session-refresher.sh that did just `oci session
# refresh` and nothing else.

set -o pipefail   # not -u: many code paths use optionally-set vars and empty arrays
                  # under bash 3.2 (macOS default), which trip up nounset.

PROFILE="${PROFILE:-BoatOc1}"
INTERVAL="${INTERVAL:-1800}"
PF_LOG_DIR="${PF_LOG_DIR:-/tmp}"
CONFIG="${1:-${REFRESHER_CONFIG:-$HOME/.oci-session-refresher.conf}}"
PF_SPECS=()

ts() { date '+%F %T'; }

if [[ -f "$CONFIG" ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG"
    echo "[$(ts)] sourced config: $CONFIG"
else
    echo "[$(ts)] no config at $CONFIG — running in session-refresh-only mode"
fi

refresh_session() {
    if oci session refresh --profile "$PROFILE" >/dev/null 2>&1; then
        echo "[$(ts)] session refresh OK ($PROFILE)"
    else
        echo "[$(ts)] session refresh FAILED ($PROFILE) — VPN down or session fully expired (manual 'oci session authenticate' needed)"
    fi
}

pf_listening() {
    local port=$1
    [[ -n "$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null)" ]]
}

# Kill any kubectl port-forward whose argv contains "<port>:" — i.e. the
# stale process that owned this local port. We only get here when the port
# is not LISTEN, so any such process is hung/dying and safe to terminate.
kill_stale_kubectl_for_port() {
    local port=$1 pids
    pids=$(ps -eo pid,command 2>/dev/null \
        | awk -v p="$port" '/kubectl/ && /port-forward/ && index($0, " " p ":") { print $1 }')
    if [[ -n "$pids" ]]; then
        echo "[$(ts)] :$port — killing stale kubectl PIDs: $pids"
        # shellcheck disable=SC2086
        kill $pids 2>/dev/null || true
        sleep 1
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
    fi
}

restart_pf() {
    local port=$1; shift
    local args=("$@")
    local kc_args=()
    [[ -n "${KUBECONFIG:-}" ]] && kc_args=(--kubeconfig "$KUBECONFIG")

    # Rewrite the bare ":<remote>" token into "<local>:<remote>".
    local rewritten=() found=0 a
    for a in "${args[@]}"; do
        if [[ "$a" =~ ^:([0-9]+)$ ]]; then
            rewritten+=("${port}:${BASH_REMATCH[1]}")
            found=1
        else
            rewritten+=("$a")
        fi
    done
    if (( ! found )); then
        echo "[$(ts)] :$port — spec missing ':<remote-port>' token; passing args through"
    fi

    local log="$PF_LOG_DIR/pf-${port}.log"
    echo "[$(ts)] :$port — restarting: kubectl ${kc_args[*]} ${rewritten[*]}"
    {
        echo "----- restart $(ts) -----"
        echo "kubectl ${kc_args[*]} ${rewritten[*]}"
    } >>"$log"
    nohup kubectl "${kc_args[@]}" "${rewritten[@]}" >>"$log" 2>&1 &
    disown
}

check_pfs() {
    # Optional dynamic resolution — lets the config rebuild PF_SPECS each tick
    # (e.g., re-discover pod names after a pod restart).
    if declare -f resolve_pf_specs >/dev/null; then
        PF_SPECS=()
        resolve_pf_specs
    fi

    if (( ${#PF_SPECS[@]} == 0 )); then
        echo "[$(ts)] PF monitor: no specs declared — skipping health check"
        return
    fi

    local alive=0 total=0 restored=() spec port rest
    for spec in "${PF_SPECS[@]}"; do
        total=$((total+1))
        port=${spec%%|*}
        rest=${spec#*|}
        # shellcheck disable=SC2206
        local args=($rest)   # word-split is intentional here

        if pf_listening "$port"; then
            alive=$((alive+1))
            continue
        fi

        kill_stale_kubectl_for_port "$port"
        restart_pf "$port" "${args[@]}"

        # Give kubectl up to ~5s to bind the local port before we declare a win.
        local i
        for i in 1 2 3 4 5; do
            sleep 1
            pf_listening "$port" && break
        done
        if pf_listening "$port"; then
            alive=$((alive+1))
            restored+=("$port")
        else
            echo "[$(ts)] :$port — did not come up within 5s (see $PF_LOG_DIR/pf-${port}.log)"
        fi
    done

    if (( ${#restored[@]} )); then
        echo "[$(ts)] tick OK — PFs alive: $alive/$total — restored: ${restored[*]}"
    else
        echo "[$(ts)] tick OK — PFs alive: $alive/$total"
    fi
}

trap 'echo "[$(ts)] caught signal, exiting"; exit 0' INT TERM

echo "[$(ts)] starting; profile=$PROFILE interval=${INTERVAL}s specs=${#PF_SPECS[@]} dynamic=$(declare -f resolve_pf_specs >/dev/null && echo yes || echo no)"
while true; do
    refresh_session
    check_pfs
    sleep "$INTERVAL"
done
