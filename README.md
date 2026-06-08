# inference-cache benchmark

A reusable benchmark harness for measuring [inference-cache](https://github.com/cachebox-project/inference-cache) performance and iterating on cache-policy tuning. Built on top of [sgl-project/genai-bench](https://github.com/sgl-project/genai-bench).

## What this is

genai-bench is great at measuring LLM serving performance — TTFT, throughput, latency distributions — and it natively supports prefix-cache-friendly workloads via `--prefix-len`. But it has no visibility into the inference-cache **routing decision** or **server-side state** (hit rate, eviction, index utilization).

This harness adds a thin layer that:

- Routes traffic through inference-cache's `LookupRoute` gRPC when measuring `lookup` mode (genai-bench alone can't call gRPC services)
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
├── README.md                 # this file
├── run_tuning_bench.sh       # entry point — run / compare / list-scenarios / clean
├── lib/
│   ├── lookup_proxy.py       # LookupRoute-aware HTTP proxy in front of vLLM
│   ├── collect_ic_metrics.py # scrapes inference-cache /metrics over time
│   └── correlate.py          # merges genai-bench + ic-metrics → markdown report
└── scenarios/
    ├── rag-headline.yaml     # RAG-style workload with long shared prefix
    ├── tuning-loop.yaml      # short, cheap iteration scenario
    └── chat-multi-turn.yaml  # chat workload — shorter prefix, more concurrency
```

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
3. **Port-forwards** to the inference-cache server (gRPC + metrics) and the vLLM engine (HTTP). The harness expects:
   ```bash
   # In the namespace where inference-cache is installed:
   kubectl -n inference-cache port-forward svc/inference-cache-server 38001:8080 &  # /metrics
   kubectl -n inference-cache port-forward svc/inference-cache-server 38002:9090 &  # gRPC

   # In the workload namespace:
   kubectl -n <workload-ns> port-forward svc/<vllm-engine-svc> 38000:8000 &
   # For baseline-mode measurements (vanilla vLLM, no cache plane):
   kubectl -n <baseline-ns> port-forward svc/<vanilla-vllm-svc> 38005:8000 &
   ```

   Override the defaults via env vars (`IC_SERVER_METRICS`, `IC_SERVER_GRPC`, `VLLM_ENGINE_URL`, `VLLM_BASELINE_URL`) — see `run_tuning_bench.sh` header.

   Verify everything is reachable before a real run:
   ```bash
   make check-paths
   ```

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

## Modes

| Mode | What it points genai-bench at | What it tests |
|---|---|---|
| `baseline` | Vanilla vLLM, no cache plane at all (`VLLM_BASELINE_URL`) | Reference TTFT — single pod, no cache plane |
| `no-hint` | `lookup_proxy.py --no-lookup-route` in front of cache-enabled vLLM | Cache plane up, routing layer disabled — every request round-robins across all configured `LOOKUP_PROXY_REPLICAS`. Isolates the LMCache wrapper's contribution from any LookupRoute-driven routing benefit. |
| `lookup` | `lookup_proxy.py` in front of cache-enabled vLLM | Full system — the proxy calls `LookupRoute` and routes to the hinted replica on `PREFIX_MATCH`, otherwise round-robins across `LOOKUP_PROXY_REPLICAS` (CAC-154) |

`no-hint` requires `LOOKUP_PROXY_REPLICAS` to be set so the proxy has multiple upstreams to round-robin across — with the env var unset it falls back to a single-element `--replicas` list (`VLLM_ENGINE_URL`) and collapses to a 1-pod measurement (the warning prints in that case). See **Setting up `lookup` mode** below for how to populate `LOOKUP_PROXY_REPLICAS`. The same env var serves both modes; `no-hint` only uses the HTTP-URL field of each entry (the ZMQ field is parsed but ignored, and the tokenizer / `--ic-server` are not loaded).

The proxy is the bit that mediates between genai-bench's "single endpoint URL" assumption and inference-cache's "list of replicas + a routing hint" model. Without the proxy, genai-bench can't exercise the routing-decision path *and* can't spread traffic across replicas — which is why `no-hint` also routes through the proxy (just with the LookupRoute RPC short-circuited and ZMQ subscriptions skipped).

### How `lookup` mode actually works (the B-b architecture)

vLLM's block hashes are computed with `builtins.hash()` by default — process-local and not reproducible across pods. A gateway client that hand-rolls a hash from prompt text will silently produce values that never match anything in the server's index. The proxy avoids this by **observing the engine's emitted hashes** rather than recomputing them.

```
ZMQ subscriber tasks (one per replica)         per-replica chain table
    ↓                                                ↓
listen to BlockStored / BlockRemoved / -→  (parent_hash, token_tuple) → block_hash
AllBlocksCleared on each replica's :5557

incoming HTTP request
    ↓
tokenize prompt (HF AutoTokenizer)
    ↓
chunk into B-token blocks (B is auto-detected from BlockStored.block_size)
    ↓
walk each replica's chain → find longest leading match across all replicas
    ↓
send LookupRoute with that replica's exact block_hashes + token_counts chain
    ↓
PREFIX_MATCH → route to hinted replica's HTTP URL
NO_HINT / TIMEOUT / NO_CHAIN_OBSERVED → round-robin across --replicas pool
```

**`--replicas` (CAC-154)**: the round-robin fallback pool — a list of
upstream URLs, comma-separated or repeated. Models the production
dumb-gateway behavior: when the server returns no usable hint, traffic
spreads evenly across all known replicas instead of concentrating on one
pod. Replaces the old `--default-upstream` (which sent every NO_HINT
response to a single target, hammering r0 in the typical layout). The
proxy keeps a per-process round-robin pointer so picks are deterministic
and a 30-request NO_HINT burst against 3 URLs produces a clean 10/10/10
split.

**Cold-start behavior**: until the proxy has seen events for a prefix, that prefix gets NO_HINT and is routed round-robin. After ~5-10 requests with a shared prefix, the chain table populates and subsequent requests get PREFIX_MATCH. This is exactly how a real gateway integration would behave — and what benchmarks measure during their warmup + steady-state windows.

### Setting up `lookup` mode

You need port-forwards to TWO endpoints per replica:

| Per-replica need | What it's for |
|---|---|
| **ZMQ event port** (`:5557` on the pod) | Subscriber listens; vLLM publishes here |
| **HTTP serve port** (`:8000` on the pod) | Where the proxy forwards requests when it has a hint |

For a 3-replica deployment in namespace `ic-smoke`:

```bash
# List the pod names
PODS=($(kubectl -n ic-smoke get pod -l app=vllm-engine -o jsonpath='{.items[*].metadata.name}'))

# ZMQ port-forwards (15001, 15002, 15003)
kubectl -n ic-smoke port-forward pod/${PODS[0]} 15001:5557 &
kubectl -n ic-smoke port-forward pod/${PODS[1]} 15002:5557 &
kubectl -n ic-smoke port-forward pod/${PODS[2]} 15003:5557 &

# HTTP port-forwards (38010, 38011, 38012)
kubectl -n ic-smoke port-forward pod/${PODS[0]} 38010:8000 &
kubectl -n ic-smoke port-forward pod/${PODS[1]} 38011:8000 &
kubectl -n ic-smoke port-forward pod/${PODS[2]} 38012:8000 &

# Tell the harness about them
export LOOKUP_PROXY_REPLICAS="r0|tcp://localhost:15001|http://localhost:38010,r1|tcp://localhost:15002|http://localhost:38011,r2|tcp://localhost:15003|http://localhost:38012"

# Pick the tokenizer matching what the served model expects
export LOOKUP_PROXY_TOKENIZER="hf-internal-testing/llama-tokenizer"
```

Then run normally:

```bash
./run_tuning_bench.sh run --scenario rag-headline --label test-bb --mode lookup
```

The proxy's per-request routing decisions are logged to `<results_dir>/lookup_proxy.log`. Each response also carries `X-Cache-Lookup-Reason` and `X-Cache-Route-Reason` headers for visibility.

Each `route_decision` log line includes a `match_quality` field bucketed from the chain length the proxy walked:

| Bucket | Chain blocks | Tokens (approx.) | Meaning |
|---|---|---|---|
| `trivial` | 1 | ~16 | chat-template framing only — every replica matches the same prefix; routing here is essentially round-robin |
| `weak` | 2 – 7 | 32 – 112 | small shared prefix — modest hit |
| `strong` | 8+ | 128+ | meaningful prefix reuse — the routing benefit lands here |

Reading the log distribution by bucket tells you what fraction of `PREFIX_MATCH` responses actually drove routing benefit, without waiting for the server-side `PREFIX_MATCH_STRONG/WEAK` differentiation (CAC-149).

For runtime stats (chain table size per replica, hit/miss counters):

```bash
curl -s http://localhost:18100/proxy/metrics | jq
```

A Prometheus text-format mirror lives at `/proxy/metrics.prom`; scrape that for the per-replica ZMQ event counter (`lookup_proxy_zmq_events_received_total{replica="r0"}`) — the load-bearing signal for silent-SUB outages. A replica stuck at 0 while siblings advance means its ZMQ PUB subscription isn't receiving events; long-chain matches for that replica will silently route elsewhere until the SUB recovers.

The proxy also gates startup on this signal: it blocks the HTTP listener from binding until every configured replica has produced at least one event, or `--zmq-startup-timeout` (default 30s) elapses. On timeout it starts anyway and logs the silent replicas loudly; a background loop then re-establishes silent SUB sockets every `--zmq-retry-interval` (default 10s) — but only when at least one sibling replica is flowing (the all-silent state is "cluster idle", not a SUB bug).

### Limitations

- **Cold start is real**: the first few requests with a brand-new prefix will be NO_HINT until events propagate. Benchmarks should treat this as part of the warmup window.
- **One LookupRoute per request**: ~50ms timeout. Failing open on timeout adds tail latency at p99. Tune `LOOKUP_TIMEOUT_S` in `lookup_proxy.py` if you need a different budget.
- **Per-replica chain tables don't share state**: same token sequence on two replicas produces two different hashes (process-local). The proxy walks each replica separately and picks the longest hit; it doesn't try to merge across replicas.
- **Memory**: each replica's chain table is bounded at 100k entries by LRU. At ~16 tokens/block, that's enough for ~1.6M tokens of cumulative prefix history per replica — usually adequate for benchmarks.

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
