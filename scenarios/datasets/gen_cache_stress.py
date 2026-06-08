"""
Generate the cache-stress prompt dataset.

Produces N distinct ~K-token "prefix blocks", each combined with Q different
short questions, for a total of N*Q prompts. Output is one prompt per line in
a plain text file (genai-bench `--dataset-path` format).

Why this exists
---------------
The headline RAG scenario uses a single shared prefix that fits comfortably in
vLLM's in-process prefix cache forever. Result: vLLM alone handles every hit,
and inference-cache's offload tier + routing layer have no observable job.

Cache-stress is the opposite: many distinct prefixes (cumulative working set
well past vLLM's local cache capacity), with random ordering so there's no
temporal locality. Behaviors to expect:

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
    # default knobs — 600 × 5 × 5000 (CAC-139 forced-eviction sizing)
    python scenarios/datasets/gen_cache_stress.py

    # custom knobs (recorded in <output>.meta.yaml for reproducibility)
    python scenarios/datasets/gen_cache_stress.py \\
        --num-prefixes 200 --questions-per-prefix 5 --words-per-prefix 5000

Reproducible: seeded RNG. Re-running with the same args gives byte-identical
output. The sidecar `<output>.meta.yaml` records the exact args used so the
benchmark harness can capture them in per-run metadata (see CAC-159).
"""

from __future__ import annotations

import argparse
import os
import random
import sys

# ---------------------------------------------------------------------------
# Defaults — match the post-CAC-139 sizing that main settled on (600 × 5 × 5000).
#
# Sizing notes for the defaults:
#   Total prompts: 600 × 5 = 3000; ~5500 tokens each.
#   Working set:   600 × ~5500 = 3.3M prefix tokens.
# This is 2.5× Llama-3.1-8B's per-replica KV budget at --max-model-len 8192
# (~450K-token cache capacity) when round-robined across 3 replicas
# (3.3M / 3 ≈ 1.1M per replica), and 7× when targeting a single replica —
# eviction is unavoidable in either case, forcing vLLM's local prefix cache
# to spill into the LMCache offload tier and exercising the routing layer.
#
# History (also recorded in CAC-159 for context):
#   600 × 5 × 5000 → current default; settles CAC-139 (LMCache T2 functional?)
#   200 × 5 × 5000 → reverted 5-iter rerun sizing; T1-bound on 3-pod spread
#   50 × 10 × 1100 → original phase-1 sizing
# ---------------------------------------------------------------------------

DEFAULT_NUM_PREFIXES = 600
DEFAULT_QUESTIONS_PER_PREFIX = 5
DEFAULT_WORDS_PER_PREFIX = 5000  # ≈ 5500–7000 Llama tokens at ~1.4 tokens/word
DEFAULT_PREFIX_SEED = "cache_stress_prefix"
DEFAULT_SHUFFLE_SEED = "cache_stress_shuffle"

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


def make_prefix(idx: int, words_per_prefix: int, seed: str) -> str:
    """Generate a deterministic prefix block of ``words_per_prefix`` words."""
    rng = random.Random(f"{seed}:{idx}")
    pool = POOLS[idx % len(POOLS)]
    vocab = COMMON + pool + pool  # bias toward the topical pool
    # Build sentences of 8-18 words for plausible structure.
    words: list[str] = []
    remaining = words_per_prefix
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
    if not text.endswith("."):
        text += "."
    # genai-bench dataset-path expects one prompt per line — guard against
    # accidental newlines from the generator.
    return text.replace("\n", " ").replace("\r", " ")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the cache-stress prompt dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--num-prefixes", type=int, default=DEFAULT_NUM_PREFIXES,
        help="Number of distinct prefix blocks to generate.",
    )
    p.add_argument(
        "--questions-per-prefix", type=int, default=DEFAULT_QUESTIONS_PER_PREFIX,
        help="Questions appended per prefix (cycled from a fixed pool of 10).",
    )
    p.add_argument(
        "--words-per-prefix", type=int, default=DEFAULT_WORDS_PER_PREFIX,
        help="Words per prefix block (≈ 1.4× tokens for Llama).",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Output file path. Defaults to cache_stress.txt next to this script.",
    )
    p.add_argument(
        "--prefix-seed", type=str, default=DEFAULT_PREFIX_SEED,
        help="Seed prefix for the per-block RNG.",
    )
    p.add_argument(
        "--shuffle-seed", type=str, default=DEFAULT_SHUFFLE_SEED,
        help="Seed for the final prompt-order shuffle.",
    )
    return p.parse_args(argv)


def write_meta_sidecar(meta_path: str, args: argparse.Namespace, num_prompts: int) -> None:
    """Write a YAML sidecar recording the exact args used to produce the dataset.

    The benchmark harness embeds this in per-run dataset.txt metadata so a
    historical run can be reproduced by re-running the generator with the
    recorded args.
    """
    capped = min(args.questions_per_prefix, len(QUESTIONS))
    capped_note = (
        f"  # capped at len(QUESTIONS)={len(QUESTIONS)}\n"
        if args.questions_per_prefix > len(QUESTIONS) else ""
    )
    with open(meta_path, "w") as f:
        f.write("# Auto-generated by gen_cache_stress.py — do not hand-edit.\n")
        f.write("generator: gen_cache_stress.py\n")
        f.write(f"num_prefixes: {args.num_prefixes}\n")
        f.write(f"questions_per_prefix: {capped}\n{capped_note}")
        f.write(f"words_per_prefix: {args.words_per_prefix}\n")
        f.write(f"prefix_seed: {args.prefix_seed!r}\n")
        f.write(f"shuffle_seed: {args.shuffle_seed!r}\n")
        f.write(f"total_prompts: {num_prompts}\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cache_stress.txt"
    )
    # Cap questions to the available pool — using more would IndexError below.
    q_count = min(args.questions_per_prefix, len(QUESTIONS))
    prompts: list[str] = []
    for prefix_idx in range(args.num_prefixes):
        prefix = make_prefix(prefix_idx, args.words_per_prefix, args.prefix_seed)
        for q_idx in range(q_count):
            q = QUESTIONS[q_idx]
            prompts.append(f"{prefix} Question: {q}")
    random.Random(args.shuffle_seed).shuffle(prompts)
    with open(out_path, "w") as f:
        for p in prompts:
            f.write(p + "\n")

    write_meta_sidecar(out_path + ".meta.yaml", args, len(prompts))

    avg_chars = sum(len(p) for p in prompts) // len(prompts)
    sys.stderr.write(
        f"wrote {len(prompts)} prompts to {out_path}; "
        f"distinct prefixes={args.num_prefixes}, avg prompt length={avg_chars} chars "
        f"(~{avg_chars // 4} tokens estimated)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
