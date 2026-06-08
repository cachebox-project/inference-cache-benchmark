"""
Generate the rag-multi-context prompt dataset.

Produces N distinct RAG-style contexts (each ~K words of synthetic topical
text) × Q questions per context = N×Q prompts. Each prompt is a single line:

    <context> Question: <question>

Sizing the working set
----------------------
The Phase 3 scenario targets a *moderate* working set — large enough that a
single-pod baseline thrashes T1, but small enough that the 3-pod spread fits
comfortably in T1 once routing affinity kicks in. The default knobs:

    200 contexts × 30 questions × 4000 words/context
    ≈ 200 × 5500 tokens = 1.1M working-set tokens
    ≈ 6000 prompts total

Per-pod working set under 3-pod spread: 1.1M / 3 ≈ 370K (fits T1 ~450K).
On baseline (single pod, no routing): full 1.1M → 2.5× T1 → eviction storm.

Why distinct question pools
---------------------------
Each "context" is the cacheable prefix. Questions are short and varied so the
benchmark exercises the prefix-cache (cache wins are on the context block),
not the trailing few tokens.

Reproducible: seeded RNG; same args → byte-identical output.

Usage
-----
    # default — matches scenarios/rag-multi-context.yaml
    python scenarios/datasets/gen_rag_multi_context.py

    # custom shape
    python scenarios/datasets/gen_rag_multi_context.py \\
        --num-contexts 150 --questions-per-context 100 --words-per-context 6000 \\
        --output scenarios/datasets/perfect_storm_rag.txt
"""

from __future__ import annotations

import argparse
import os
import random
import sys

# ---------------------------------------------------------------------------
# Topical vocabulary pools — diverse enough that tokenized prefixes don't
# accidentally collide across contexts.
# ---------------------------------------------------------------------------

COMMON = (
    "the of and to in is that for it as was with be by on not this are or at "
    "have an but from they which one you all would will can has more when "
    "what who said there been no if her his my your our their we were them "
    "then also only just time year work because each how about other into "
    "after first many such most over its some these way much new very even "
    "between within while where these those without through under above"
).split()

MEDICINE = (
    "patient diagnosis treatment therapy clinical study trial cohort placebo "
    "outcome efficacy adverse symptom medication dosage prescription protocol "
    "anatomy physiology biopsy lesion tumor benign malignant chronic acute "
    "metabolic vascular neural hormonal genetic mutation expression receptor "
    "antibody antigen immune inflammation pathology etiology prognosis remission "
    "screening prevention vaccine pediatric geriatric oncology cardiology "
    "neurology endocrinology rehabilitation recovery comorbidity epidemiology"
).split()

LAW = (
    "court ruling judgment statute precedent appellate jurisdiction defendant "
    "plaintiff plaintiffs counsel motion brief evidence testimony witness "
    "discovery deposition subpoena warrant indictment charge plea verdict "
    "sentence appeal remand reversal injunction settlement damages contract "
    "tort negligence liability breach remedy equity arbitration mediation "
    "constitution amendment regulation enforcement compliance violation fine "
    "license permit franchise easement title deed lien lease tenancy"
).split()

HISTORY = (
    "empire kingdom dynasty revolution rebellion conflict alliance treaty "
    "settlement migration colony expedition voyage discovery conquest decline "
    "renaissance enlightenment reformation industrial agrarian feudal urban "
    "century decade era epoch period regime monarchy republic democracy "
    "parliament assembly council senate magistrate consul tribune emperor "
    "chronicle inscription manuscript artifact excavation archaeology heritage "
    "trade route caravan port garrison fortress citadel chapel cathedral monument"
).split()

ENGINEERING = (
    "design specification tolerance load stress strain beam column truss "
    "moment torque pressure flow turbulence laminar viscosity friction friction "
    "thermal conductivity insulation expansion contraction tensile compressive "
    "ductile brittle hardness elasticity plasticity fatigue fracture corrosion "
    "alloy composite ceramic polymer concrete steel aluminum titanium copper "
    "circuit voltage current resistance capacitance inductance frequency phase "
    "amplifier filter regulator transformer rectifier sensor actuator controller"
).split()

POOLS = [MEDICINE, LAW, HISTORY, ENGINEERING]

# Short, varied questions. RAG-multi-context wants the question to be the
# trailing-and-uncached part of the prompt; the context is what gets reused.
QUESTIONS = [
    "Summarize the main idea in one sentence.",
    "What is the central claim being made?",
    "Identify two pieces of evidence the text relies on.",
    "What follows logically from the above?",
    "What assumptions are being made?",
    "How would you evaluate this argument?",
    "What is the practical implication?",
    "What questions remain unanswered?",
    "What might a critic of this view say?",
    "What would change if a key assumption were false?",
    "Restate the argument in plain language.",
    "What is the strongest counter-argument?",
    "What evidence would falsify this claim?",
    "Identify a hidden assumption.",
    "Who would benefit from this conclusion being true?",
    "What context would a reader need to evaluate this?",
    "Compare this to the conventional view.",
    "What would a skeptic ask first?",
    "Where is the argument weakest?",
    "Where is the argument strongest?",
    "What is the second-order implication?",
    "What does the text leave out?",
    "How could the analysis be tightened?",
    "What follow-up study would resolve the ambiguity?",
    "Translate one key term into ordinary language.",
    "What metric would test this empirically?",
    "How would a practitioner apply this?",
    "What time horizon does this assume?",
    "What is the smallest change that would invalidate it?",
    "What broader trend does this fit into?",
    "What earlier view does this revise?",
    "What would the opposing camp emphasize?",
    "Is the framing neutral or loaded?",
    "Where does the text rely on authority over evidence?",
    "What would a numerate reader want quantified?",
    "What single example would best illustrate the point?",
    "What is the boldest claim and is it warranted?",
    "What single sentence could be cut without loss?",
    "What two ideas could be merged?",
    "Identify a redundancy in the argument.",
    "What is the topic sentence?",
    "What is the concluding move?",
    "Reorder the argument for clarity.",
    "What additional source would strengthen this?",
    "What test of the claim is easiest to run?",
    "Distill the argument into a tweet.",
    "What is the most-cited concept here?",
    "What value judgment underlies the framing?",
    "Identify two terms that need definition.",
    "What would a different discipline notice first?",
]


def make_context(idx: int, words_per_context: int, seed: str) -> str:
    """Generate a deterministic ~K-word context block."""
    rng = random.Random(f"{seed}:{idx}")
    pool = POOLS[idx % len(POOLS)]
    vocab = COMMON + pool + pool  # bias toward the topical pool
    words: list[str] = []
    remaining = words_per_context
    while remaining > 0:
        n = rng.randint(8, 18)
        sentence = [rng.choice(vocab) for _ in range(min(n, remaining))]
        sentence[0] = sentence[0].capitalize()
        words.extend(sentence)
        if remaining - n > 0:
            words[-1] = words[-1] + "."
        remaining -= n
    text = " ".join(words)
    if not text.endswith("."):
        text += "."
    return text.replace("\n", " ").replace("\r", " ")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description="Generate the rag-multi-context prompt dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--num-contexts", type=int, default=200,
        help="Number of distinct RAG context blocks.",
    )
    p.add_argument(
        "--questions-per-context", type=int, default=30,
        help="Questions appended per context (cycled from the QUESTIONS pool).",
    )
    p.add_argument(
        "--words-per-context", type=int, default=4000,
        help="Words per context block (≈ 1.4× Llama tokens).",
    )
    p.add_argument(
        "--output", type=str,
        default=os.path.join(here, "rag_multi_context.txt"),
        help="Output file path.",
    )
    p.add_argument(
        "--context-seed", type=str, default="rag_multi_context",
        help="Seed prefix for the per-context RNG.",
    )
    p.add_argument(
        "--shuffle-seed", type=str, default="rag_multi_context_shuffle",
        help="Seed for the final prompt-order shuffle.",
    )
    p.add_argument(
        "--no-shuffle", action="store_true",
        help="Emit prompts in context-major order (default: shuffled).",
    )
    return p.parse_args(argv)


def write_meta_sidecar(meta_path: str, args: argparse.Namespace, num_prompts: int) -> None:
    """Sidecar yaml so per-run metadata can record exact generator params."""
    capped = min(args.questions_per_context, len(QUESTIONS))
    capped_note = (
        f"  # capped at len(QUESTIONS)={len(QUESTIONS)}\n"
        if args.questions_per_context > len(QUESTIONS) else ""
    )
    with open(meta_path, "w") as f:
        f.write("# Auto-generated by gen_rag_multi_context.py — do not hand-edit.\n")
        f.write("generator: gen_rag_multi_context.py\n")
        f.write(f"num_contexts: {args.num_contexts}\n")
        f.write(f"questions_per_context: {capped}\n{capped_note}")
        f.write(f"words_per_context: {args.words_per_context}\n")
        f.write(f"context_seed: {args.context_seed!r}\n")
        f.write(f"shuffle_seed: {args.shuffle_seed!r}\n")
        f.write(f"shuffled: {not args.no_shuffle}\n")
        f.write(f"total_prompts: {num_prompts}\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.questions_per_context > len(QUESTIONS):
        sys.stderr.write(
            f"warning: --questions-per-context={args.questions_per_context} "
            f"exceeds the {len(QUESTIONS)}-question pool; cycling with repeats.\n"
        )
    q_count = args.questions_per_context
    prompts: list[str] = []
    for ctx_idx in range(args.num_contexts):
        context = make_context(ctx_idx, args.words_per_context, args.context_seed)
        for q_idx in range(q_count):
            q = QUESTIONS[q_idx % len(QUESTIONS)]
            prompts.append(f"{context} Question: {q}")
    if not args.no_shuffle:
        random.Random(args.shuffle_seed).shuffle(prompts)
    with open(args.output, "w") as f:
        for p in prompts:
            f.write(p + "\n")

    write_meta_sidecar(args.output + ".meta.yaml", args, len(prompts))

    avg_chars = sum(len(p) for p in prompts) // len(prompts) if prompts else 0
    sys.stderr.write(
        f"wrote {len(prompts)} prompts to {args.output}; "
        f"distinct contexts={args.num_contexts}, "
        f"words/context≈{args.words_per_context}, "
        f"avg prompt length={avg_chars} chars (~{avg_chars // 4} tokens estimated)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
