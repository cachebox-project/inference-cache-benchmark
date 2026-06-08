# Scenarios

Each YAML in this directory is a benchmark **scenario** — a complete set of
inputs to `run_tuning_bench.sh run`. Scenarios are intentionally
data-shaped so any of them can be re-run unchanged later; the shape is
documented in the top-level README under "Scenario YAML schema".

## Phase 3 portfolio (the brown-bag headline set)

The three scenarios below were designed together so that the working-set sizes
land at meaningfully different points on the cache-capacity curve. Read them
as a portfolio: each one exercises a different part of the system, and the
*deltas* between modes (`baseline` / `no-hint` / `lookup`) are the headline
output — not absolute TTFT numbers.

| Scenario | Working set | Per-pod under 3-pod spread | What it shows |
|---|---|---|---|
| `rag-multi-context` | 1.1M tokens (200 × 5500) | ~370K (fits T1 ≈ 450K) | Routing affinity wins big — baseline thrashes T1, 3-pod spread fits |
| `cache-stress-extreme` | 7M tokens (1000 × 7000) | ~2.3M (5× T1) | T1+T2 hierarchy under stress; lookup benefit lands once CAC-166 ships |
| `perfect-storm-rag` | 1.2M tokens (150 × 8000) | ~400K (just fits T1) | Maximum-delta showcase: baseline thrashes, lookup sticks |

Each scenario specifies four concurrency points (`[4, 8, 16, 32]`) so the
report includes a sweep, not a single point.

### `rag-multi-context.yaml`

Moderate working-set RAG. 200 contexts × 30 questions = 6000 prompts, with
each context ≈ 5500 Llama tokens. The arithmetic puts the per-pod working set
at ~370K tokens under a 3-pod spread — comfortably inside T1 — while the
1.1M total causes a single-pod baseline to evict constantly.

Expected: baseline TTFT elevated and high-variance. `no-hint` recovers via
T2 hits but loses cross-pod routing benefit. `lookup` is fastest because the
routing layer pins each context to a single replica, letting that replica's
T1 stay warm.

Generate dataset:

```bash
python scenarios/datasets/gen_rag_multi_context.py
# → scenarios/datasets/rag_multi_context.txt
```

### `cache-stress-extreme.yaml`

Production-scale stress. 1000 prefixes × 3 questions = 3000 prompts, each
prefix ≈ 7000 tokens. The 7M working set is 5× T1 even when spread across
3 pods, so no mode escapes eviction. The point of this scenario is to show
graceful degradation: hit rates fall but TTFT doesn't cliff-edge, and
`lookup` mode's routing affinity keeps per-pod working sets bounded.

Generate dataset (note: the `cache_stress.txt` default lives next to the
existing `cache-stress.yaml`, so this scenario uses an explicit output path):

```bash
python scenarios/datasets/gen_cache_stress.py \
    --num-prefixes 1000 --questions-per-prefix 3 --words-per-prefix 5500 \
    --output scenarios/datasets/cache_stress_extreme.txt
# → scenarios/datasets/cache_stress_extreme.txt
```

### `perfect-storm-rag.yaml`

Best-case demo. 150 contexts × 100 questions × ~8000 tokens each. Working set
sized to JUST overflow a single-pod T1 (1.2M ≈ 2.6× T1) while comfortably
fitting in the 3-pod spread (~400K per pod). With 100 questions rotating
through each context, repeat probability is high enough that steady-state hit
rate in `lookup` mode approaches 99%.

This is the scenario for the headline number. Use it sparingly — it's the
demo, not the validation.

Generate dataset:

```bash
python scenarios/datasets/gen_perfect_storm_rag.py
# → scenarios/datasets/perfect_storm_rag.txt
```

## Other scenarios

| Scenario | Purpose |
|---|---|
| `rag-headline` | Launch-claim measurement: 75% shared prefix, classic RAG shape |
| `cache-stress` | Pre-Phase-3 cache-stress (600 prefixes × 5 questions); kept for back-compat |
| `chat-multi-turn` | Multi-turn chat workload; tests stickier per-conversation routing |
| `tuning-loop` | Short cheap iteration scenario for CRD tuning |

## Running

Single scenario, single mode:

```bash
./run_tuning_bench.sh run --scenario rag-multi-context --label demo --mode lookup
```

Three-mode comparison for one scenario:

```bash
for mode in baseline no-hint lookup; do
  ./run_tuning_bench.sh run --scenario rag-multi-context --label "demo-$mode" --mode "$mode"
done
./run_tuning_bench.sh compare demo-baseline demo-no-hint demo-lookup
```

Full Phase 3 sweep (3 scenarios × 3 modes × 4 iters):

```bash
scripts/phase3-sweep.sh
```

See the top-level README "Phase 3 sweep" section for details on
cold-reset behaviour, reporting, and rerun-after-failure flags.
