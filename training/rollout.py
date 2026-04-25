"""
Episode rollout: thin loop around env.reset/env.step that records every
turn's prompt, the action that was taken, and whether the model output
parsed cleanly.

The same loop is used by:
- generate_sft_data.py — policy = scripted teacher (InformedBot.act)
- the Colab notebook GRPO loop — policy = LLM-driven (formats prompt,
                                  generates, parses)

Keeping the loop in one place guarantees SFT and GRPO see identical
prompt formatting, which matters because GRPO compounds tiny mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from market_env.environment import MarketEnvironment
from market_env.models import MarketAction, MarketObservation

from training.prompts import format_observation, serialize_action

# A policy takes an observation, returns (action, parse_ok). parse_ok is
# always True for scripted bots and reflects parser success for LLMs.
Policy = Callable[[MarketObservation], tuple[MarketAction, bool]]


@dataclass
class Turn:
    """One step of an episode, captured for SFT data or GRPO training."""

    turn_index: int
    prompt: str                # user message (formatted observation)
    action: MarketAction       # the action that was actually applied
    action_text: str           # JSON serialization (matches parse_action format)
    parse_ok: bool             # only meaningful for LLM policies


@dataclass
class Trajectory:
    """A complete episode's record."""

    episode_id: str
    seed: int
    difficulty: str
    turns: list[Turn] = field(default_factory=list)
    final_reward: float = 0.0
    true_value: Optional[float] = None
    reward_breakdown: dict = field(default_factory=dict)

    @property
    def parse_failure_rate(self) -> float:
        if not self.turns:
            return 0.0
        return sum(1 for t in self.turns if not t.parse_ok) / len(self.turns)


def run_episode(
    env: MarketEnvironment,
    policy: Policy,
    *,
    seed: int,
    difficulty: str = "medium",
    episode_length: int = 50,
    bot_config: str = "default",
    trainable_agent_id: str = "agent_1",
) -> Trajectory:
    """Run one full episode, return the captured trajectory."""
    obs = env.reset(
        seed=seed,
        difficulty=difficulty,
        episode_length=episode_length,
        bot_config=bot_config,
        trainable_agent_id=trainable_agent_id,
    )
    traj = Trajectory(episode_id=obs.episode_id, seed=seed, difficulty=difficulty)

    done = False
    reward = 0.0
    info: dict = {}
    while not done:
        action, parse_ok = policy(obs)
        traj.turns.append(
            Turn(
                turn_index=obs.turn,
                prompt=format_observation(obs),
                action=action,
                action_text=serialize_action(action),
                parse_ok=parse_ok,
            )
        )
        obs, reward, done, info = env.step(obs.episode_id, action)

    traj.final_reward = float(reward)
    traj.true_value = obs.true_value
    traj.reward_breakdown = info.get("reward_breakdown", {})
    return traj


# ---------------------------------------------------------------------------
# Convenience adapters: turn a Bot or a callable-without-parse-ok into a Policy
# ---------------------------------------------------------------------------

def bot_policy(bot) -> Policy:
    """Wrap a Bot (whose .act returns just MarketAction) as a Policy."""
    def _policy(obs: MarketObservation) -> tuple[MarketAction, bool]:
        return bot.act(obs), True
    return _policy
