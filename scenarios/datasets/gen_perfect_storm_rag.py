"""
Generate the perfect-storm-rag prompt dataset.

This is the rag-multi-context generator with larger contexts and a higher
per-context question multiplier. Sized to JUST overflow a single-pod T1 cache
while comfortably fitting in the 3-pod spread — the maximum-delta showcase
between baseline and lookup modes.

Default knobs:
    150 contexts × 100 questions × 6000 words/context
    ≈ 150 × 8000 tokens = 1.2M working-set tokens
    ≈ 15000 prompts total

Per-pod working set under 3-pod spread: 1.2M / 3 ≈ 400K (just fits T1 ~450K).
On baseline (single pod): full 1.2M → 2.6× T1 → constant eviction.
Replaying the same 100 questions per context = high reuse, big lookup win.

This module just thinly wraps gen_rag_multi_context so the perfect-storm
defaults live in one obvious place. Use either generator with the matching
scenario YAML; the harness will fail loudly if dataset_path doesn't exist.

Usage
-----
    python scenarios/datasets/gen_perfect_storm_rag.py
    # → writes scenarios/datasets/perfect_storm_rag.txt
"""

from __future__ import annotations

import os
import sys

# Import the rag-multi-context generator. Living in the same dir so a relative
# import works without dragging in a setup.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_rag_multi_context as base  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(here, "perfect_storm_rag.txt")
    # Perfect-storm-specific defaults: larger contexts (~8000 tokens), more
    # question rotations per context (high reuse), narrower context count
    # (150 fits the routing-spread sweet spot).
    defaults = [
        "--num-contexts", "150",
        "--questions-per-context", "100",
        "--words-per-context", "6000",
        "--output", default_output,
        "--context-seed", "perfect_storm_rag",
        "--shuffle-seed", "perfect_storm_rag_shuffle",
    ]
    args = defaults + list(argv or sys.argv[1:])
    return base.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
