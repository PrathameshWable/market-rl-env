"""
Curriculum schedule for Stage 1 GRPO.

Per the plan:
- Steps   0 –  599: easy only
- Steps 600 – 1499: mixed (50% easy, 50% medium)
- Steps 1500+:      full (1/3 easy, 1/3 medium, 1/3 hard)

Tweaking these cutoffs is one of the first places to look if the reward
curve stalls early. Easy gives a clear signal that lets the model learn
the action format and basic directional trading; introducing medium too
soon drowns the gradient in noise.
"""

from __future__ import annotations

import random

EASY_UNTIL = 600
MIXED_UNTIL = 1500


def difficulty_for_step(step: int, rng: random.Random) -> str:
    """Sample a difficulty for the given GRPO step."""
    if step < EASY_UNTIL:
        return "easy"
    if step < MIXED_UNTIL:
        return rng.choice(["easy", "medium"])
    return rng.choice(["easy", "medium", "hard"])
