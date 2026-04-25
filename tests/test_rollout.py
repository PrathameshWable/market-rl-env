"""Tests for the rollout helper and the SFT data generator."""
from __future__ import annotations

import json
from pathlib import Path

from market_env.bots import RandomBot
from market_env.environment import MarketEnvironment
from market_env.models import MarketAction
from training.generate_sft_data import generate
from training.rollout import bot_policy, run_episode


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

class TestRollout:
    def test_run_episode_with_hold_policy_terminates(self):
        env = MarketEnvironment()

        def hold_policy(_obs):
            return MarketAction(action_type="hold"), True

        traj = run_episode(env, hold_policy, seed=0, episode_length=5, difficulty="easy")
        assert len(traj.turns) == 5
        assert traj.true_value is not None
        assert isinstance(traj.final_reward, float)
        assert "pnl_normalized" in traj.reward_breakdown

    def test_run_episode_with_random_bot_completes(self):
        env = MarketEnvironment()
        bot = RandomBot("agent_1", seed=0)
        traj = run_episode(
            env, bot_policy(bot), seed=1, episode_length=5, difficulty="easy",
        )
        assert len(traj.turns) == 5
        assert traj.parse_failure_rate == 0.0  # scripted bot never fails to parse

    def test_each_turn_records_prompt_and_action_text(self):
        env = MarketEnvironment()
        bot = RandomBot("agent_1", seed=0)
        traj = run_episode(env, bot_policy(bot), seed=2, episode_length=3)
        for turn in traj.turns:
            assert turn.prompt
            assert turn.action_text.startswith("{")
            assert turn.action_text.endswith("}")


# ---------------------------------------------------------------------------
# SFT data generator
# ---------------------------------------------------------------------------

class TestSFTGenerator:
    def test_generates_jsonl_with_chat_format(self, tmp_path: Path):
        out = tmp_path / "sft_mini.jsonl"
        stats = generate(n_episodes=3, out_path=out, episode_length=5)

        assert out.exists()
        assert stats["n_examples"] == 3 * 5  # 3 episodes, 5 turns each

        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == stats["n_examples"]

        sample = json.loads(lines[0])
        assert "messages" in sample
        roles = [m["role"] for m in sample["messages"]]
        assert roles == ["system", "user", "assistant"]

        # Assistant message must be valid action JSON
        assistant = json.loads(sample["messages"][2]["content"])
        assert assistant["action_type"] in {"buy", "sell", "cancel", "hold"}

    def test_action_distribution_is_not_all_one_thing(self, tmp_path: Path):
        out = tmp_path / "sft_dist.jsonl"
        stats = generate(n_episodes=10, out_path=out, episode_length=20)
        # Teacher should sometimes act, not just hold every turn
        assert stats["hold_ratio"] < 0.99
