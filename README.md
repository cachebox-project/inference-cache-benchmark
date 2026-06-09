# inference-cache benchmark

A reusable benchmark harness for measuring [inference-cache](https://github.com/cachebox-project/inference-cache) performance and iterating on cache-policy tuning. Built on top of [sgl-project/genai-bench](https://github.com/sgl-project/genai-bench).

## What this is

genai-bench is great at measuring LLM serving performance — TTFT, throughput, latency distributions — and it natively supports prefix-cache-friendly workloads via `--prefix-len`. But it has no visibility into the inference-cache **routing decision** or **server-side state** (hit rate, eviction, index utilization).

This harness adds a thin layer that:

- Routes traffic through inference-cache's `LookupRoute` gRPC when measuring `lookup` mode via a minimal client (`lib/dumb_gateway_client.py`, CAC-152) that mirrors what a production gateway does — tokenize, hash, ask, route. genai-bench alone can't call gRPC services.
- Scrapes the inference-cache server's `/metrics` in parallel with the run
- Merges genai-bench's per-request data with the server-side state into a single comparison report
- Supports the **tuning loop**: apply a CRD change → re-run → compare two labeled runs

## When to use this

| Use case | Tool |
|---|---|
| Day-to-day tuning of `CachePolicy` / `CacheBackend` / `PromptTemplate` CRs | **this harness** |
| Comparing two CRD configurations side-by-side | **this harness** |
| Reproducing a customer's reported TTFT issue | **this harness** |
| Multi-model sweep with consistent methodology | **this harness** |
| Pure LLM throughput / latency measurement (no cache plane involved) | [genai-bench](https://github.com/sgl-project/genai-bench) directly |

## Layout

```
.
├── README.md                       # this file
├── run_tuning_bench.sh             # entry point — run / compare / list-scenarios / clean
├── lib/
│   ├── dumb_gateway_client.py      # production-shaped HTTP gateway (default; CAC-152)
│   ├── lookup_proxy_legacy.py      # deprecated chain-walking proxy (kept one release for A/B)
│   ├── collect_ic_metrics.py       # scrapes inference-cache /metrics over time
│   └── correlate.py                # merges genai-bench + ic-metrics → markdown report
├── scenarios/
│   ├── rag-headline.yaml           # RAG-style workload with long shared prefix
│   ├── tuning-loop.yaml            # short, cheap iteration scenario
│   └── chat-multi-turn.yaml        # chat workload — shorter prefix, more concurrency
└── scripts/
    ├── oci-session-refresher.sh    # keeps OCI session + kubectl PFs alive
    └── oci-session-refresher.conf.example
```

### The dumb-gateway principle

The benchmark client (`lib/dumb_gateway_client.py`) is intentionally minimal:
tokenize the prompt, hash blocks content-addressably, ask the server, route.
**The server makes routing decisions; the gateway follows.** No chain table,
no ZMQ subscriptions, no per-replica state — exactly what a production gateway
would do.

This is the "we decide routing; the gateway follows" principle (see the main
inference-cache project's design docs). The deprecated `lookup_proxy_legacy.py`
re-implemented chain-walking inside the harness; that drift hid server-side
ranking bugs (CAC-150) and skewed Phase 2 results. The new client makes the
benchmark a faithful proxy for what production will do.

## Prerequisites

```bash
# 1. Python deps
make install
# (or: pip install genai-bench grpcio grpcio-tools aiohttp pandas pyyaml requests)
```

Plus on the runner host: `kubectl`, `yq`, `jq`, `make`.

You'll also need:

1. **A Kubernetes cluster with inference-cache installed** and a running vLLM workload wired into a `CacheBackend`. See the [main repo's getting-started guide](https://github.com/cachebox-project/inference-cache) for setup.
2. **Generated proto stubs** for inference-cache's gRPC API. From this directory, with the main [inference-cache](https://github.com/cachebox-project/inference-cache) repo checked out as a sibling:
   ```bash
   make proto
   # → regenerates proto/ from ../inference-cache/proto/
   ```
   Override the source path if it lives elsewhere:
   ```bash
   make proto INFERENCE_CACHE_REPO=/abs/path/to/inference-cache
   ```
3. **Port-forwards.** The harness needs reachability to the inference-cache server and to vLLM. **Read the gotcha below before choosing PF targets.**

   For the inference-cache server (single replica — `svc/...` is fine):
   ```bash
   kubectl -n inference-cache port-forward svc/inference-cache-server 38001:8080 &  # /metrics
   kubectl -n inference-cache port-forward svc/inference-cache-server 38002:9090 &  # gRPC
   ```

   For vLLM, **always use per-pod port-forwards** when the deployment has more than one replica — see the [`lookup` mode setup](#setting-up-lookup-mode) below. That covers `--mode lookup` and `--mode no-hint` (the client round-robins fallback traffic across all pods).

   For `--mode baseline` against a **single-replica** vanilla vLLM, `svc/...` is acceptable:
   ```bash
   kubectl -n <baseline-ns> port-forward svc/<vanilla-vllm-svc> 38005:8000 &
   ```

   Override the defaults via env vars (`IC_SERVER_METRICS`, `IC_SERVER_GRPC`, `VLLM_ENGINE_URL`, `VLLM_BASELINE_URL`) — see `run_tuning_bench.sh` header.

   Verify everything is reachable before a real run:
   ```bash
   make check-paths
   ```

> ⚠️ **`kubectl port-forward svc/<name>` does NOT load-balance.** The apiserver
> picks one endpoint at PF-establish time and pins every connection on that
> port to that endpoint until the PF dies. Across multiple PF sessions over
> 36 h on the `ic-smoke` deployment (CAC-163), one pod accumulated 0 requests
> while siblings served millions — the kube-proxy/iptables layer was fine; the
> imbalance came entirely from PF pinning. **Any measurement that depends on
> traffic spreading across replicas (cache distribution, per-pod hit rate,
> replica-balance studies) must use per-pod PFs through `dumb_gateway_client.py`'s
> explicit round-robin** (`LOOKUP_PROXY_REPLICAS`, set up in the [lookup-mode
> section](#setting-up-lookup-mode) below). Use `svc/...` PF only for
> single-replica targets, or for local connectivity smoke tests where you
> don't care which pod answers.
>
> The harness includes an automatic post-run check (`lib/check_pod_distribution.py`)
> that scrapes per-pod `vllm:prefix_cache_queries_total` before and after the
> run and warns loudly if any pod received zero traffic while siblings got
> some — pointing at exactly this misconfiguration.

### Keeping port-forwards alive across long sessions

`kubectl port-forward` dies silently for many reasons — server-side connection timeouts, OCI VPN blips, OKE API restarts. On OCI specifically, the OCI session token also expires on its own cadence, and once it does every `kubectl` (including any subsequent `port-forward`) starts failing with an auth error. A run that takes 4 hours in real time will routinely hit one of these.

`scripts/oci-session-refresher.sh` is a long-running watchdog that handles both:

- refreshes the OCI session token (`oci session refresh --profile $PROFILE`) every `$INTERVAL` seconds,
- checks each declared local port with `lsof -ti tcp:$port -sTCP:LISTEN`, and
- for any port that's no longer LISTEN, kills the stale `kubectl` process holding that port (if any) and re-runs `kubectl port-forward` for it.

Each tick logs one summary line — `tick OK — PFs alive: 12/12` in steady state, or `tick OK — PFs alive: 12/12 — restored: 38010 15001` after a restore. Per-port kubectl output goes to `$PF_LOG_DIR/pf-<port>.log` so you can dig into why a particular PF won't stay up.

```bash
# 1. Copy the example config and edit PROFILE / KUBECONFIG / PF_SPECS for your cluster:
cp scripts/oci-session-refresher.conf.example ~/.oci-session-refresher.conf
$EDITOR ~/.oci-session-refresher.conf

# 2. Launch detached. Output goes to wherever you redirect; logs are per-tick.
nohup scripts/oci-session-refresher.sh > /tmp/oci-session-refresher.log 2>&1 &
disown

# 3. Tail the log to watch tick state:
tail -f /tmp/oci-session-refresher.log
```

For pod-targeted PFs whose pod names can change after a restart (e.g. the per-replica `:5557` ZMQ ports used by `lookup` mode), the config can define a `resolve_pf_specs` shell function that's called each tick to rebuild `PF_SPECS` from `kubectl get pod` output. See the commented-out block at the bottom of the example config for the pattern.

To validate restore behavior end-to-end without waiting on a real PF to die: launch the refresher with `INTERVAL=10`, find a kubectl PF process with `pgrep -fl 'kubectl.*port-forward'`, `kill -9` it, and watch the next tick log a `— restored: <port>` entry.

## Quick start

### One-off run

```bash
./run_tuning_bench.sh run \
  --scenario rag-headline \
  --label "current" \
  --mode lookup
```

Produces `results/current-<timestamp>/`:
- `genai-bench/` — raw genai-bench experiment dir (Excel, plots, per-request JSON)
- `ic-metrics.csv` — inference-cache metrics scraped every Ns
- `report.md` — correlated comparison vs. acceptance criteria
- `crd-snapshot.yaml` — CRDs at run time (so future `compare` runs can diff)
- `scenario.yaml` — the scenario YAML used (verbatim copy)
- `dataset.txt` — dataset fingerprint (path, SHA-256, line count, first-prompt
  head, generator config) — only written when the scenario uses `dataset_path`
- `cluster-state.yaml` — pod + node + event snapshot for the run window (see below)

### cluster-state.yaml — why it's there

When a pod restarts mid-run (lm-smoke OOM, vllm-engine rolling restart, kubelet
eviction, etc.), the bench result on its own gives no signal about *why* things
went sideways — by the time you go looking, the cluster state that explained it
is gone. `cluster-state.yaml` is the smallest artifact that lets you do
post-mortem days later without a live cluster.

Per run, the harness captures:

- **`pods_start` / `pods_end`** — name, namespace, pod_uid, node, age, and
  `restart_count` for the configured targets at the run boundaries. Two
  comparisons matter: `pod_uid` changes mean the pod was deleted+recreated;
  `restart_count` deltas mean a container restarted in place.
- **`pod_changes`** — a denormalised summary of those deltas so the headline
  ("lm-smoke-yyy was restarted") is visible without diffing the two lists.
- **`events`** — pod-level events (`OOMKilled`, `BackOff`, `FailedScheduling`,
  …) from the configured namespaces that fall inside the run window. Pulled
  via `kubectl get events --field-selector involvedObject.namespace=<ns>`.
- **`nodes`** — capacity, allocatable, and pressure conditions for every node
  hosting one of the captured pods.

Which pods to capture is controlled by `CLUSTER_STATE_TARGETS`, a
comma-separated list of `<namespace>:<name-prefix>` pairs. The default targets
the Phase 2 setup:

```
CLUSTER_STATE_TARGETS="ic-smoke:vllm-engine,ic-smoke:lm-smoke,gpu-baseline:vllm-baseline"
```

Override it for a different workload layout — e.g. a Qwen sweep:

```bash
export CLUSTER_STATE_TARGETS="ic-smoke:qwen-vllm,gpu-baseline:qwen-baseline"
./run_tuning_bench.sh run --scenario rag-headline --label qwen-trial --mode lookup
```

Events are pulled from the unique namespaces in `CLUSTER_STATE_TARGETS`
unless you set `CLUSTER_STATE_EVENTS_NS` explicitly (also comma-separated).

All kubectl calls are best-effort: any failure (no cluster context, missing
namespace, RBAC denial) is logged in yellow and the run continues. The merged
yaml will contain an `error:` field instead of pod data.

### The tuning loop

```bash
# 1. Baseline measurement with current CRDs
./run_tuning_bench.sh run --scenario tuning-loop --label "ttl-60s" --mode lookup

# 2. Change a CRD field
kubectl -n <ns> patch cachepolicy <name> --type=merge -p '{"spec":{"evictionTTL":"5m"}}'

# 3. Wait for steady state, re-measure
sleep 60
./run_tuning_bench.sh run --scenario tuning-loop --label "ttl-5m" --mode lookup

# 4. Compare
./run_tuning_bench.sh compare ttl-60s ttl-5m
# → results/compare-ttl-60s-ttl-5m-<timestamp>.md
```

### Three-mode comparison

```bash
for mode in baseline no-hint lookup; do
  ./run_tuning_bench.sh run --scenario rag-headline --label "v1-$mode" --mode "$mode"
done
./run_tuning_bench.sh compare v1-baseline v1-no-hint v1-lookup
```

### Phase 3 sweep

For the brown-bag headline benchmark, three new scenarios live alongside
`rag-headline` (full details in [`scenarios/README.md`](scenarios/README.md)):

| Scenario | Working set | What it shows |
|---|---|---|
| `rag-multi-context` | 1.1M tokens (200 × 5500) | Moderate WS: baseline thrashes T1, 3-pod spread fits — routing affinity wins |
| `cache-stress-extreme` | 7M tokens (1000 × 7000) | T1+T2 hierarchy under stress; graceful degradation |
| `perfect-storm-rag` | 1.2M tokens (150 × 8000) | Maximum-delta showcase — steady-state hit rate ≈ 99% in lookup mode |

Each scenario specifies four concurrency points (`[4, 8, 16, 32]`); the
`scripts/phase3-sweep.sh` orchestrator runs all three across the three modes
(`baseline` / `no-hint` / `lookup`) and four iterations (`cold`, `warm-1`,
`warm-2`, `warm-3`). Cold iters do a `kubectl rollout restart` between runs;
warm iters reuse the cache state from the previous run to capture warmup →
steady-state transitions.

```bash
# Generate datasets first (one-time):
python scenarios/datasets/gen_rag_multi_context.py
python scenarios/datasets/gen_cache_stress.py \
    --num-prefixes 1000 --questions-per-prefix 3 --words-per-prefix 5500 \
    --output scenarios/datasets/cache_stress_extreme.txt
python scenarios/datasets/gen_perfect_storm_rag.py

# Run the full 36-bench matrix (3 scenarios × 3 modes × 4 iters):
scripts/phase3-sweep.sh

# Or a subset — useful for rerunning after a failure:
scripts/phase3-sweep.sh --scenarios "perfect-storm-rag" --iters "cold warm-1"

# Dry-run (echo commands without executing):
scripts/phase3-sweep.sh --dry-run

# Build comparison reports only (skips re-running):
scripts/phase3-sweep.sh --compare-only
```

Interpreting the output:

- Each iteration writes a `results/phase3-<scenario>-<mode>-<iter>-<ts>/`
  directory with the usual `report.md`, `ic-metrics.csv`,
  `vllm-metrics.csv` (Phase 3), `cluster-state.yaml`, etc.
- The orchestrator builds a three-way comparison per scenario at the end
  (`results/compare-phase3-<scenario>-{baseline,no-hint,lookup}-cold-<ts>.md`),
  using the `cold` iters as the headline numbers.
- The headline is the **delta** between modes within a single scenario, not
  absolute numbers across scenarios.
- Warm iters live alongside the cold runs for tail-analysis; load them
  manually with `./run_tuning_bench.sh compare <label1> <label2>` if you
  want a warmup-vs-steady-state view.

Per-pod vLLM metrics (`vllm-metrics.csv`) are scraped automatically when
`VLLM_METRICS_ENDPOINTS` is set or derivable from `LOOKUP_PROXY_REPLICAS` —
baseline mode auto-derives one endpoint from `VLLM_BASELINE_URL`. Each
`report.md` now includes a "Throughput (per concurrency)" section and a
"vLLM aggregate throughput" section computed from those CSVs.

## Modes

| Mode | What it points genai-bench at | What it tests |
|---|---|---|
| `baseline` | Vanilla vLLM, no cache plane at all (`VLLM_BASELINE_URL`) | Reference TTFT — single pod, no cache plane |
| `no-hint` | `dumb_gateway_client.py --routing-mode=round-robin` in front of cache-enabled vLLM | Cache plane up, routing layer disabled — every request round-robins across all configured `LOOKUP_PROXY_REPLICAS`. Isolates the LMCache wrapper's contribution from any LookupRoute-driven routing benefit. (CAC-153) |
| `lookup` | `dumb_gateway_client.py --routing-mode=lookup` in front of cache-enabled vLLM | Full system — the client calls `LookupRoute` and routes to the hinted replica on `PREFIX_MATCH`, otherwise round-robins across `LOOKUP_PROXY_REPLICAS` (CAC-154) |

`no-hint` requires `LOOKUP_PROXY_REPLICAS` to be set so the client has multiple upstreams to round-robin across — with the env var unset both modes fail to start (the warning prints in that case). See **Setting up `lookup` mode** below for how to populate `LOOKUP_PROXY_REPLICAS`. The same env var serves both modes; `no-hint` only uses the HTTP-URL field of each entry (the ZMQ field is parsed but ignored, and the tokenizer / `--ic-server` are not loaded).

The client is the bit that mediates between genai-bench's "single endpoint URL" assumption and inference-cache's "list of replicas + a routing hint" model. Without the client, genai-bench can't exercise the routing-decision path *and* can't spread traffic across replicas — which is why `no-hint` also routes through the client (just with the LookupRoute RPC short-circuited).

### How `lookup` mode actually works (the dumb-gateway model)

The new client (`lib/dumb_gateway_client.py`) follows the production gateway
contract exactly — no chain table, no ZMQ, no per-replica state:

```
incoming HTTP request
    ↓
tokenize prompt (HF AutoTokenizer; Llama-3.1 chat template for chat-completions)
    ↓
chunk into block-size-token blocks (default 16) and emit content-addressed
sha256-truncated block hashes (parent_hash || token_bytes → 8-byte digest)
    ↓
send LookupRoute(tenant, model, block_hashes, block_token_counts)
    ↓
PREFIX_MATCH → route to replica_scores[0].replica_id's HTTP URL
NO_HINT / TIMEOUT / TENANT_HOT / unknown replica id → round-robin across
                                                       --replica pool (CAC-154)
```

Two key consequences:

1. **Hashes are content-addressed and reproducible across processes.** Same
   prompt on any pod (or in the client) yields the same chain. No need to
   subscribe to engine events to reconstruct the index — the server matches
   the client-computed chain directly against its `(tenant, model, hash_scheme)`
   partition. This is the whole reason the chain-walking proxy went away.

2. **The server makes the routing decision.** The client doesn't pick "the
   longest match" or "the highest-scoring replica" — it forwards the server's
   pick verbatim. CAC-149's chat-template matched-tokens floor and CAC-151's
   distinguishing-power scoring both live server-side and now drive routing
   directly, without the harness filtering them through chain-walk logic.

**`--replica` repeated**: the replica pool — `id=url` (preferred) or legacy
`id|zmq|http_url[|router]` (zmq + router fields parsed for backward compat
with `LOOKUP_PROXY_REPLICAS` and ignored). Per-process round-robin pointer:
30 fallback picks across 3 URLs produces an exact 10/10/10 split (CAC-154).

**`--routing-mode={lookup,round-robin}`**: clean lever for both bench modes
from one binary (CAC-153). `lookup` is the production path; `round-robin`
skips the RPC entirely — tokenizer not loaded, gRPC channel not opened.

**Cold-start behavior**: until the server's index has seen this `(tenant,
model, hash_scheme)` partition populated with matching chains, requests get
`NO_HINT` and are routed round-robin. With CAC-149's Llama chat-template
floor active, the framing tokens alone produce `PREFIX_MATCH` responses even
on a cold index, so warmup looks smoother than under the old chain-walking
proxy. This mirrors how the real production gateway will behave.

### Setting up `lookup` mode

You need a port-forward to the HTTP serve port of each replica:

| Per-replica need | What it's for |
|---|---|
| **HTTP serve port** (`:8000` on the pod) | Where the client forwards requests (PREFIX_MATCH hint or round-robin fallback) |

No ZMQ port-forward is needed — the dumb client doesn't subscribe to engine
events. The `tcp://...` field in `LOOKUP_PROXY_REPLICAS` is parsed for
backward compat and ignored. (If you also run `USE_LEGACY_PROXY=1` to A/B
against the deprecated chain-walking proxy, the ZMQ PFs *are* required for
that mode — see the **Legacy proxy comparison** section.)

For a 3-replica deployment in namespace `ic-smoke`:

```bash
# List the pod names
PODS=($(kubectl -n ic-smoke get pod -l app=vllm-engine -o jsonpath='{.items[*].metadata.name}'))

# HTTP port-forwards (38010, 38011, 38012)
kubectl -n ic-smoke port-forward pod/${PODS[0]} 38010:8000 &
kubectl -n ic-smoke port-forward pod/${PODS[1]} 38011:8000 &
kubectl -n ic-smoke port-forward pod/${PODS[2]} 38012:8000 &

# Tell the harness about them. The ZMQ field (`tcp://...`) is kept in the
# env var format for backward compat with the legacy proxy; the dumb client
# parses and ignores it.
export LOOKUP_PROXY_REPLICAS="r0|tcp://unused|http://localhost:38010,r1|tcp://unused|http://localhost:38011,r2|tcp://unused|http://localhost:38012"

# Pick the tokenizer matching what the served model expects (default:
# unsloth/Meta-Llama-3.1-8B-Instruct). MUST match the engine's tokenizer
# or the block hashes won't agree with what the engine commits to its
# index — see lib/dumb_gateway_client.py docstring.
export LOOKUP_PROXY_TOKENIZER="unsloth/Meta-Llama-3.1-8B-Instruct"
```

Then run normally:

```bash
./run_tuning_bench.sh run --scenario rag-headline --label test-dgw --mode lookup
```

Per-request routing decisions are logged to `<results_dir>/dumb_gateway_client.log`. Each response carries `X-Cache-Lookup-Reason` and `X-Cache-Route-Reason` headers for visibility.

Each `route_decision` log line includes a `match_quality` field bucketed from the chain length the client computed:

| Bucket | Chain blocks | Tokens (approx.) | Meaning |
|---|---|---|---|
| `trivial` | 1 | ~16 | chat-template framing only — server's CAC-149 floor returns PREFIX_MATCH here regardless of replica state; routing benefit is minimal |
| `weak` | 2 – 7 | 32 – 112 | small shared prefix |
| `strong` | 8+ | 128+ | meaningful prefix reuse — the routing benefit lands here |

Reading the log distribution by bucket tells you what fraction of `PREFIX_MATCH` responses actually drove routing benefit. Pairs with the server-side `inferencecache_lookup_route_floor_match_total` counter from CAC-149.

For runtime stats (per-mode counters, per-replica round-robin counts):

```bash
curl -s http://localhost:18100/proxy/metrics | jq
```

### Legacy proxy comparison

The previous chain-walking proxy lives on as `lib/lookup_proxy_legacy.py` for
one release to let operators A/B compare its routing decisions against the
dumb client's. To run a bench with it instead of the new client:

```bash
USE_LEGACY_PROXY=1 ./run_tuning_bench.sh run \
  --scenario rag-headline --label legacy-comparison --mode lookup
```

The legacy mode requires ZMQ port-forwards (`:5557` per pod) in addition to
the HTTP ones — see git history for the original setup. **It will be removed
in the next release.** New work should rely on the dumb client; the legacy
proxy is retained only for cross-checks during the transition.

### Limitations

- **Cold start**: until the server's index has matching chains for this
  `(tenant, model, hash_scheme)` partition, requests get NO_HINT and round-robin.
  CAC-149's chat-template floor returns PREFIX_MATCH on framing tokens alone,
  so warmup is smoother than under the legacy proxy.
- **One LookupRoute per request**: 500ms default client-side timeout. Failing
  open on timeout falls back to round-robin (no retry). Tune via
  `--lookup-timeout` if you need a different budget.
- **Hash-scheme agreement**: the engine's kvevent-subscriber must populate the
  index under the same `--hash-scheme` tag the client sends (default `vllm`)
  AND use the same content-addressed block-hashing algorithm. If they disagree,
  the server returns NO_HINT for non-trivial prefixes and falls through to the
  CAC-149 chat-template floor — you'll see PREFIX_MATCH for framing only.

## Reproducing a historical run

Every run that uses a `dataset_path` scenario gets a `dataset.txt` in its
result dir capturing exactly which dataset bytes were used. To re-run a
historical comparison from scratch:

1. **Read `results/<label>-<ts>/dataset.txt`.** It lists the dataset's
   SHA-256, line count, and the `generator_config:` block (`num_prefixes`,
   `questions_per_prefix`, `words_per_prefix`, RNG seeds).
2. **Regenerate the dataset** with those exact knobs:
   ```bash
   python scenarios/datasets/gen_cache_stress.py \
     --num-prefixes 600 \
     --questions-per-prefix 5 \
     --words-per-prefix 5000
   ```
   Generators are seeded, so the output is byte-identical across runs with
   the same args. Confirm by re-hashing:
   ```bash
   shasum -a 256 scenarios/datasets/cache_stress.txt
   # → must match the sha256: field in dataset.txt
   ```
3. **Re-run the harness** with the same scenario and the rest of the harness
   config (CRDs, replicas, etc.) restored to what `crd-snapshot.yaml` records.

If a result dir is missing `dataset.txt` it predates CAC-159; use
`tools/backfill_dataset_meta.sh` to write the fingerprint based on the
**current** dataset (the script flags this with a `NOTE:` line since the
on-disk dataset may differ from what the run actually used).

### Generator knobs (cache_stress)

| CLI flag | Default | Meaning |
|---|---|---|
| `--num-prefixes` | 600 | Distinct prefix blocks |
| `--questions-per-prefix` | 5 | Questions appended per prefix (capped at 10) |
| `--words-per-prefix` | 5000 | ≈ 1.4× tokens for Llama tokenizer |
| `--prefix-seed` | `cache_stress_prefix` | Per-block RNG seed prefix |
| `--shuffle-seed` | `cache_stress_shuffle` | Final shuffle seed |
| `--output` | `scenarios/datasets/cache_stress.txt` | Output path |

The generator also writes `<output>.meta.yaml` alongside the dataset; the
harness embeds this in each run's `dataset.txt` so the knobs travel with the
result.

Historical sizings used in Phase 2 (recorded in CAC-159 for context):
- `600 × 5 × 5000` — current default, CAC-139 forced-eviction sizing (~100 MB, 3.3M prefix tokens)
- `200 × 5 × 5000` — reverted 5-iter rerun sizing (~34 MB, 1.4M prefix tokens)
- `50 × 10 × 1100` — original phase-1 sizing (~3 MB, 75K prefix tokens)

## Scenario YAML schema

```yaml
name: rag-headline
description: |
  RAG-style workload with a 1500-token shared prefix and a ~500-token unique
  suffix. The launch-claim measurement.

genai_bench:
  task: text-to-text
  api_backend: openai
  model: vllm-model
  tokenizer: meta-llama/Llama-3.1-8B-Instruct
  traffic_scenarios:
    - "D(2000,200)"            # input,output tokens (deterministic)
  prefix_len: 1500             # 75% prefix sharing
  num_concurrency: [1, 4]
  max_requests_per_run: 200
  max_time_per_run: 5          # minutes

ic_metrics:
  scrape_interval_s: 10
  capture:
    - inferencecache_lookup_route_calls_total
    - inferencecache_lookup_route_latency_seconds
    - inferencecache_index_entries
    - inferencecache_index_evictions_total

acceptance:
  ttft_p50_max_ms: 80
  lookup_hit_rate_pct_min: 70
  cold_path_overhead_max_ms: 5
```

Copy `rag-headline.yaml` as the template for new scenarios. See [`docs.sglang.io/genai-bench/user-guide/scenario-definition`](https://docs.sglang.io/genai-bench/user-guide/scenario-definition) for the full set of `traffic_scenarios` syntax.

## What the report contains

**Single-run `report.md`:**

1. Header: scenario, label, timestamp, CRD snapshot reference
2. TTFT distribution: p50/p95/p99 (from genai-bench)
3. Throughput: tokens/sec, requests/sec
4. inference-cache view: hit rate, eviction rate, server-side lookup latency p99, index entries
5. Acceptance gate: each criterion PASS/FAIL with the measured value
6. Sanity checks: errors, index population

**`compare-<A>-<B>.md` for comparisons:**

1. Side-by-side table with delta column
2. CRD diff (`diff -u`) between the two runs
3. Honest "deciding if delta beats noise floor is your call" verdict

## Anti-patterns

- **Multi-knob changes between runs.** If you `kubectl apply` two CRD changes at once, the comparison can't tell you which won. The harness can't enforce this — discipline lives in your head.
- **Comparing single-run results.** Run each label 3× and take the median if anything close to noise floor. genai-bench's `--max-time-per-run` makes this cheap.
- **Ignoring the acceptance gate.** If acceptance fails, the report says so. Don't paper over it.
- **Cherry-picking concurrency levels.** Report all sweep points; never just the one that looks good.

## What this harness does NOT do

- **Tune CRDs for you.** That's future work (controller-side tuning recommendations).
- **Generate per-customer workloads.** Use real customer datasets via the genai-bench `--dataset-path` option; configure in the YAML.
- **Run in CI.** This is a manual tuning instrument. Use lower-level test scripts for regression gates.
- **Test failure modes.** That's a separate operational-validation effort.

## Contributing

PRs welcome. New scenarios should follow the YAML schema above and include a short `description` block explaining when to use them. The proxy's hash-scheme is configurable via `--hash-scheme` — match it to your engine's kvevent-subscriber configuration.

## License

Apache License 2.0. See [LICENSE](./LICENSE).
