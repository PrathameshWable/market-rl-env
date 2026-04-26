"""
Centralised feature flags for M7B Stage-1 ablation studies.

Each `AblationConfig` captures everything that varies between runs:
- curriculum on/off (`use_curriculum`)
- auxiliary direction-alignment reward weight (0 = disabled)
- model + training hyperparameters

Three named presets cover the M7B story:
- `BASELINE_REPLAY`: identical to Stage 1, regenerated for fair comparison
- `NO_CURRICULUM`:   shows the curriculum schedule actually helps
- `AUX_DIRECTION`:   adds signal-alignment bonus; targets Probe 3 bias

The same factory is consumed by:
- `MarketEnvironment(aux_direction_weight=cfg.aux_direction_weight)`
- `notebooks/train_ablation_colab.ipynb` (selects a preset by name)
- `tests/test_ablations.py`
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Callable

from training import curriculum

Difficulty = str  # "easy" | "medium" | "hard"


@dataclass(frozen=True)
class AblationConfig:
    """Immutable per-run config. Pass `asdict(cfg)` to W&B for reproducibility."""

    name: str
    description: str
    use_curriculum: bool = True
    aux_direction_weight: float = 0.0

    # Training shape
    sft_steps: int = 150
    grpo_steps: int = 200
    grpo_logging_steps: int = 5
    grpo_save_steps: int = 50
    sft_episodes: int = 500

    # Generation hyperparameters used by the LLM policy
    temperature: float = 0.7
    top_p: float = 0.9

    # Eval defaults (always identical across runs so numbers are comparable)
    eval_seeds: tuple[int, ...] = field(default_factory=lambda: tuple(range(100, 110)))
    eval_difficulty: Difficulty = "medium"
    eval_bot_config: str = "default"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Difficulty schedulers
# ---------------------------------------------------------------------------

def _uniform_medium(_step: int, _rng: random.Random) -> Difficulty:
    """No-curriculum baseline: every step samples medium difficulty.

    Medium (not random) is chosen so the comparison with curriculum-on
    isolates the *schedule*, not the difficulty distribution.
    """
    return "medium"


def make_difficulty_scheduler(cfg: AblationConfig) -> Callable[[int, random.Random], Difficulty]:
    """Return a `(step, rng) -> difficulty` callable for the given config."""
    if cfg.use_curriculum:
        return curriculum.difficulty_for_step
    return _uniform_medium


# ---------------------------------------------------------------------------
# Named presets
# ---------------------------------------------------------------------------

BASELINE_REPLAY = AblationConfig(
    name="baseline_replay",
    description=(
        "Stage 1 settings reproduced exactly for a fair vs.-ablation comparison. "
        "Curriculum on, no auxiliary reward. 150 SFT + 200 GRPO steps."
    ),
    use_curriculum=True,
    aux_direction_weight=0.0,
)

NO_CURRICULUM = AblationConfig(
    name="no_curriculum",
    description=(
        "Curriculum disabled — every GRPO step samples medium difficulty. "
        "Tests whether the easy-medium-hard schedule actually helps."
    ),
    use_curriculum=False,
    aux_direction_weight=0.0,
)

AUX_DIRECTION = AblationConfig(
    name="aux_direction",
    description=(
        "Auxiliary signal-alignment reward enabled (weight=0.10). Targets the "
        "'say above' bias seen in Stage 1 ToM Probe 3 by paying the agent for "
        "trading in the direction of its private signal sum."
    ),
    use_curriculum=True,
    aux_direction_weight=0.10,
)

PRESETS: dict[str, AblationConfig] = {
    cfg.name: cfg for cfg in (BASELINE_REPLAY, NO_CURRICULUM, AUX_DIRECTION)
}


def get_preset(name: str) -> AblationConfig:
    if name not in PRESETS:
        raise ValueError(
            f"unknown ablation preset: {name!r} "
            f"(must be one of {sorted(PRESETS)})"
        )
    return PRESETS[name]
