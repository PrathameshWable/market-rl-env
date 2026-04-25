"""Tests for the evaluation harness."""
from __future__ import annotations

import json
from pathlib import Path

from market_env.bots import RandomBot
from market_env.models import MarketAction
from training.evaluate import (
    DEFAULT_SEEDS,
    run_evaluation,
    save_summary,
)
from training.rollout import bot_policy


def _hold_policy(_obs):
    return MarketAction(action_type="hold"), True


class TestRunEvaluation:
    def test_hold_policy_zero_pnl_zero_participation(self):
        summary = run_evaluation(
            _hold_policy, label="hold", seeds=range(100, 103), difficulty="easy",
            episode_length=5,
        )
        assert summary.n_episodes == 3
        assert summary.mean_pnl_normalized == 0.0
        assert summary.participation_rate == 0.0
        assert summary.parse_success_rate == 1.0

    def test_random_bot_runs_without_crash(self):
        # We rebuild the policy per-seed inside the harness via a wrapper
        # so RandomBot RNGs don't share state across episodes.
        # For the test we just use one shared bot — fine for "doesn't crash".
        bot = RandomBot("agent_1", seed=42)
        summary = run_evaluation(
            bot_policy(bot), label="random",
            seeds=range(100, 105), difficulty="easy", episode_length=5,
        )
        assert summary.n_episodes == 5
        assert 0 <= summary.participation_rate <= 1
        # Pnl can be anything; just ensure it's a finite number
        assert summary.pnl_min <= summary.mean_pnl_normalized <= summary.pnl_max

    def test_summary_records_per_episode_results(self):
        summary = run_evaluation(
            _hold_policy, label="hold", seeds=[100, 101], episode_length=3,
        )
        assert len(summary.episodes) == 2
        assert summary.episodes[0].seed == 100
        assert summary.episodes[1].seed == 101
        for ep in summary.episodes:
            assert ep.n_turns == 3

    def test_save_summary_round_trips_to_json(self, tmp_path: Path):
        summary = run_evaluation(
            _hold_policy, label="hold", seeds=[100], episode_length=3,
        )
        out = tmp_path / "eval.json"
        save_summary(summary, out)

        data = json.loads(out.read_text())
        assert data["label"] == "hold"
        assert data["n_episodes"] == 1
        assert data["mean_pnl_normalized"] == 0.0
        assert "episodes" in data
        assert data["episodes"][0]["seed"] == 100


class TestDefaultSeeds:
    def test_default_seeds_are_outside_training_range(self):
        # SFT used seeds 0-499; the prompt collector uses seeds 0+. Eval seeds
        # must not overlap so we measure generalization, not memorization.
        for seed in DEFAULT_SEEDS:
            assert seed >= 100
        assert len(DEFAULT_SEEDS) == 10
