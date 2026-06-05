"""
Generate the cache-stress prompt dataset.

Produces 50 distinct ~1500-token "prefix blocks", each combined with 10
different short questions, for a total of 500 prompts. Output is one prompt
per line in a plain text file (genai-bench `--dataset-path` format).

Why this exists
---------------
The headline RAG scenario uses a single shared prefix that fits comfortably in
vLLM's in-process prefix cache forever. Result: vLLM alone handles every hit,
and inference-cache's offload tier + routing layer have no observable job.

Cache-stress is the opposite: many distinct prefixes (~75-100k tokens of
total working set), more than vLLM's local cache can hold at once. Random
ordering ensures no temporal locality. Behaviors to expect:

  baseline (vLLM-only): vLLM thrashes — newer prefixes evict older ones from
                        its local cache, so repeat hits to evicted prefixes
                        cost full prefill.
  no-hint:              LMCache offload tier can pull evicted blocks back
                        from lmcache-server. Repeats of evicted prefixes
                        recover, but at LMCache-load latency (not GPU-resident
                        latency).
  lookup:               LookupRoute hint routes a repeat-prefix request to
                        the specific replica whose local cache still has it.
                        Best case: replica still has it warm → GPU-resident
                        hit, fastest. If no replica has it locally, falls
                        back to no-hint behaviour.

Usage
-----
    python scenarios/datasets/gen_cache_stress.py
    # writes scenarios/datasets/cache_stress.txt

Reproducible: seeded RNG. Re-running gives byte-identical output.
"""

from __future__ import annotations

import os
import random
import sys

# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------

NUM_PREFIXES = 600
QUESTIONS_PER_PREFIX = 5
WORDS_PER_PREFIX = 5000  # ≈ 5500–7000 Llama tokens (measured ≈5500 in genai-bench logs)
# Total: 600 × 5 = 3000 prompts; ~5500 tokens each.
# Working set: 600 × ~5500 = 3.3M prefix tokens.
# This is 2.5× Llama-3.1-8B's per-replica KV budget at --max-model-len 8192
# (~450K-token cache capacity) when round-robined across 3 replicas
# (3.3M / 3 ≈ 1.1M per replica), and 7× when targeting a single replica —
# eviction is unavoidable in either case, forcing vLLM's local prefix cache
# to spill into the LMCache offload tier and exercising the routing layer.
#
# Bumped from 200→600 prefixes (CAC-139) to settle whether LMCache T2 is
# functional: the original 200-prefix size at ~1.1M tokens / 3 pods stayed
# near T1 capacity and never observably evicted, so LMCache `put` may not
# have fired at all. The 3× working set removes that ambiguity.

# ---------------------------------------------------------------------------
# Topical vocabulary — three pools so prefixes look domain-distinct.
# Distinct vocabularies tokenize differently → less accidental overlap.
# ---------------------------------------------------------------------------

COMMON = (
    "the of and to in is that for it as was with be by on not this are or at "
    "have an but from they which one you all would will can has more when "
    "what who said there been no if her his my your our their we were them "
    "then also only just time year work because each how about other into "
    "after first many such most over its some these way much new very even "
    "between within while where these those without through under above"
).split()

SCIENCE = (
    "theory hypothesis experiment observation measurement data analysis "
    "method evidence result conclusion variable control sample population "
    "statistic correlation causation model framework predict outcome trial "
    "instrument calibration accuracy precision uncertainty error noise "
    "signal baseline reference standard protocol replicate metric quantum "
    "particle wave frequency amplitude energy momentum velocity acceleration "
    "force field boundary surface interior molecule structure bond reaction "
    "catalyst equilibrium gradient flux density viscosity temperature pressure"
).split()

TECH = (
    "system network protocol server client request response cache memory "
    "buffer queue thread process kernel driver module library function "
    "interface endpoint configuration parameter argument value default "
    "timeout retry fallback fail open circuit breaker rate limit throttle "
    "deploy release version build artifact image container pod service "
    "cluster node region zone load balance scale capacity throughput "
    "latency tail percentile histogram metric trace span log event audit "
    "secret token credential authentication authorisation policy rule"
).split()

BUSINESS = (
    "market customer revenue growth strategy plan budget forecast quarter "
    "fiscal report metric kpi roadmap milestone deliverable stakeholder "
    "vendor supplier contract agreement renewal term price discount margin "
    "cost expense operational capital invest return risk hedge exposure "
    "compliance regulation policy procedure governance review audit board "
    "executive director manager team lead role responsibility scope owner "
    "initiative project programme launch ship release announcement campaign"
).split()

POOLS = [SCIENCE, TECH, BUSINESS]

QUESTIONS = [
    "Summarise the main idea in one sentence.",
    "What is the central claim?",
    "Identify two pieces of evidence the text relies on.",
    "What follows logically from the above?",
    "What assumptions are being made?",
    "How would you evaluate this argument?",
    "What is the practical implication?",
    "What questions remain unanswered?",
    "What might a critic of this view say?",
    "What would change if a key assumption were false?",
]


def make_prefix(idx: int) -> str:
    """Generate a deterministic ~1500-token-equivalent prefix block."""
    rng = random.Random(f"cache_stress_prefix:{idx}")
    pool = POOLS[idx % len(POOLS)]
    vocab = COMMON + pool + pool  # bias toward the topical pool
    # Build sentences of 8-18 words for plausible structure.
    words: list[str] = []
    remaining = WORDS_PER_PREFIX
    while remaining > 0:
        n = rng.randint(8, 18)
        sentence = [rng.choice(vocab) for _ in range(min(n, remaining))]
        sentence[0] = sentence[0].capitalize()
        words.extend(sentence)
        if remaining - n > 0:
            # Punctuate the previous sentence's last word.
            words[-1] = words[-1] + rng.choice([".", "."])
        remaining -= n
    text = " ".join(words)
    # Final period if missing.
    if not text.endswith("."):
        text += "."
    # genai-bench dataset-path expects one prompt per line — guard against
    # accidental newlines from the generator.
    return text.replace("\n", " ").replace("\r", " ")


def main() -> int:
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cache_stress.txt"
    )
    prompts: list[str] = []
    for prefix_idx in range(NUM_PREFIXES):
        prefix = make_prefix(prefix_idx)
        for q_idx in range(QUESTIONS_PER_PREFIX):
            q = QUESTIONS[q_idx]
            # Single-line prompt: <prefix block> question: <question>
            prompts.append(f"{prefix} Question: {q}")
    # Shuffle so no temporal locality across the run.
    random.Random("cache_stress_shuffle").shuffle(prompts)
    with open(out_path, "w") as f:
        for p in prompts:
            f.write(p + "\n")
    avg_chars = sum(len(p) for p in prompts) // len(prompts)
    sys.stderr.write(
        f"wrote {len(prompts)} prompts to {out_path}; "
        f"distinct prefixes={NUM_PREFIXES}, avg prompt length={avg_chars} chars "
        f"(~{avg_chars // 4} tokens estimated)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
