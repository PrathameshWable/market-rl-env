"""
Stage 1 evaluation harness.

Two modes:

1. CLI baseline mode (no GPU required):
       python -m training.evaluate --policy random --out training/runs/stage1_2026-04-25/eval_random.json
       python -m training.evaluate --policy hold   --out training/runs/stage1_2026-04-25/eval_hold.json

2. Programmatic mode for the trained LLM (called from the Colab notebook):
       from training.evaluate import run_evaluation
       results = run_evaluation(llm_policy, label="trained", seeds=range(100, 110))

The harness runs N episodes (held-out seeds, never seen during training)
and reports four numbers per policy:

- mean_pnl_normalized:  P&L scaled by (true_value × 100); the same scale
                        as the training reward signal. Range roughly
                        [-1, +1]; positive = beat buy-and-hold baseline.
- pnl_std:              std-dev across seeds; high = inconsistent.
- participation_rate:   fraction of turns the policy placed an order
                        (anything other than `hold`).
- parse_success_rate:   only meaningful for LLM policies — fraction of
                        turns where the model's output parsed as valid
                        JSON. RandomBot/hold are scripted so it's 1.0.

The default eval set is **seeds 100–109 at medium difficulty** with the
default 4-bot opponent composition. These are intentionally outside the
seed ranges used by the GRPO prompt collector (0–N) and the SFT teacher
(0–499), so the model has not been trained on them.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from market_env.bots import RandomBot
from market_env.environment import MarketEnvironment
from market_env.models import MarketAction, MarketObservation

from training.rollout import Policy, bot_policy, run_episode


# ---------------------------------------------------------------------------
# Defaults — keep in sync with the blog/README so all numbers are comparable.
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = tuple(range(100, 110))      # 10 held-out scenarios
DEFAULT_DIFFICULTY = "medium"
DEFAULT_EPISODE_LENGTH = 50
DEFAULT_BOT_CONFIG = "default"              # 4 scripted opponents
DEFAULT_TRAINABLE_AGENT = "agent_1"


@dataclass
class EpisodeResult:
    seed: int
    pnl_normalized: float
    participation_rate: float
    parse_success_rate: float
    n_turns: int
    true_value: float


@dataclass
class EvalSummary:
    label: str
    n_episodes: int
    seeds: list[int]
    difficulty: str
    bot_config: str

    mean_pnl_normalized: float
    median_pnl_normalized: float
    pnl_std: float
    pnl_min: float
    pnl_max: float

    participation_rate: float
    parse_success_rate: float

    episodes: list[EpisodeResult]


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _hold_policy(_obs: MarketObservation) -> tuple[MarketAction, bool]:
    return MarketAction(action_type="hold"), True


def run_evaluation(
    policy: Policy,
    *,
    label: str,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    difficulty: str = DEFAULT_DIFFICULTY,
    episode_length: int = DEFAULT_EPISODE_LENGTH,
    bot_config: str = DEFAULT_BOT_CONFIG,
    trainable_agent_id: str = DEFAULT_TRAINABLE_AGENT,
) -> EvalSummary:
    """Run `policy` on the held-out scenario set and return aggregate metrics."""
    env = MarketEnvironment()
    seeds = list(seeds)
    episodes: list[EpisodeResult] = []

    for seed in seeds:
        traj = run_episode(
            env, policy,
            seed=seed,
            difficulty=difficulty,
            episode_length=episode_length,
            bot_config=bot_config,
            trainable_agent_id=trainable_agent_id,
        )
        actions = [t.action.action_type for t in traj.turns]
        active = sum(1 for a in actions if a != "hold")
        parsed = sum(1 for t in traj.turns if t.parse_ok)

        episodes.append(EpisodeResult(
            seed=seed,
            pnl_normalized=float(traj.reward_breakdown.get("pnl_normalized", 0.0)),
            participation_rate=active / max(len(actions), 1),
            parse_success_rate=parsed / max(len(actions), 1),
            n_turns=len(actions),
            true_value=float(traj.true_value or 0.0),
        ))

    pnls = [e.pnl_normalized for e in episodes]
    return EvalSummary(
        label=label,
        n_episodes=len(episodes),
        seeds=seeds,
        difficulty=difficulty,
        bot_config=bot_config,

        mean_pnl_normalized=statistics.mean(pnls),
        median_pnl_normalized=statistics.median(pnls),
        pnl_std=statistics.stdev(pnls) if len(pnls) > 1 else 0.0,
        pnl_min=min(pnls),
        pnl_max=max(pnls),

        participation_rate=statistics.mean(e.participation_rate for e in episodes),
        parse_success_rate=statistics.mean(e.parse_success_rate for e in episodes),

        episodes=episodes,
    )


def save_summary(summary: EvalSummary, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(summary), fh, indent=2)


def print_summary(summary: EvalSummary) -> None:
    print(f"\n=== {summary.label} | {summary.n_episodes} episodes "
          f"({summary.difficulty}, {summary.bot_config}) ===")
    print(f"  mean P&L (normalized) : {summary.mean_pnl_normalized:+.4f}")
    print(f"  std                    : {summary.pnl_std:.4f}")
    print(f"  median                 : {summary.median_pnl_normalized:+.4f}")
    print(f"  range                  : [{summary.pnl_min:+.4f}, {summary.pnl_max:+.4f}]")
    print(f"  participation rate     : {summary.participation_rate:.1%}")
    print(f"  parse success rate     : {summary.parse_success_rate:.1%}")


# ---------------------------------------------------------------------------
# CLI — only supports scripted policies (random, hold). The LLM policy
# requires a GPU + Unsloth and is invoked from the Colab notebook directly.
# ---------------------------------------------------------------------------

POLICY_FACTORIES = {
    "random": lambda seed: bot_policy(RandomBot("agent_1", seed=seed)),
    "hold":   lambda _seed: _hold_policy,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=list(POLICY_FACTORIES), default="random")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--difficulty", default=DEFAULT_DIFFICULTY)
    args = parser.parse_args()

    # The CLI policies are stateful per seed (RandomBot has its own RNG), so
    # we can't share a single policy across all seeds. Run one episode at a
    # time and aggregate.
    env = MarketEnvironment()
    episodes = []
    for seed in args.seeds:
        policy = POLICY_FACTORIES[args.policy](seed)
        traj = run_episode(env, policy, seed=seed, difficulty=args.difficulty)
        actions = [t.action.action_type for t in traj.turns]
        episodes.append(EpisodeResult(
            seed=seed,
            pnl_normalized=float(traj.reward_breakdown.get("pnl_normalized", 0.0)),
            participation_rate=sum(1 for a in actions if a != "hold") / max(len(actions), 1),
            parse_success_rate=1.0,
            n_turns=len(actions),
            true_value=float(traj.true_value or 0.0),
        ))

    pnls = [e.pnl_normalized for e in episodes]
    summary = EvalSummary(
        label=args.policy,
        n_episodes=len(episodes),
        seeds=args.seeds,
        difficulty=args.difficulty,
        bot_config=DEFAULT_BOT_CONFIG,
        mean_pnl_normalized=statistics.mean(pnls),
        median_pnl_normalized=statistics.median(pnls),
        pnl_std=statistics.stdev(pnls) if len(pnls) > 1 else 0.0,
        pnl_min=min(pnls),
        pnl_max=max(pnls),
        participation_rate=statistics.mean(e.participation_rate for e in episodes),
        parse_success_rate=1.0,
        episodes=episodes,
    )

    save_summary(summary, args.out)
    print_summary(summary)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
