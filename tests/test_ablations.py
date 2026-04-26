"""Tests for M7B ablation infrastructure."""
from __future__ import annotations

import random

import pytest

from market_env.environment import MarketEnvironment
from market_env.models import MarketAction
from market_env.reward import (
    AUX_DIRECTION_BONUS,
    AgentStats,
    _direction_alignment_bonus,
    compute_reward,
)
from training import ablations


# ---------------------------------------------------------------------------
# Direction-alignment bonus: pure unit tests on the reward helper
# ---------------------------------------------------------------------------

class TestDirectionAlignmentBonus:
    def test_zero_when_weight_zero(self):
        assert _direction_alignment_bonus(10, {"a": 5.0}, weight=0.0) == 0.0

    def test_zero_when_no_position(self):
        assert _direction_alignment_bonus(0, {"a": 5.0}, weight=0.1) == 0.0

    def test_zero_when_no_signals(self):
        assert _direction_alignment_bonus(10, {}, weight=0.1) == 0.0
        assert _direction_alignment_bonus(10, None, weight=0.1) == 0.0

    def test_zero_when_signal_sum_is_zero(self):
        assert _direction_alignment_bonus(10, {"a": 1.0, "b": -1.0}, weight=0.1) == 0.0

    def test_aligned_long_position_pays_bonus(self):
        # signal sum +5 → max bonus
        bonus = _direction_alignment_bonus(10, {"a": 5.0}, weight=0.1)
        assert bonus == pytest.approx(0.1)

    def test_aligned_short_position_pays_bonus(self):
        bonus = _direction_alignment_bonus(-10, {"a": -3.0}, weight=0.1)
        assert bonus == pytest.approx(0.1 * 0.6)  # 3/5 of max

    def test_misaligned_pays_nothing(self):
        bonus = _direction_alignment_bonus(10, {"a": -5.0}, weight=0.1)
        assert bonus == 0.0

    def test_bonus_saturates_at_weight(self):
        bonus = _direction_alignment_bonus(10, {"a": 100.0}, weight=0.1)
        assert bonus == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# compute_reward defaults are byte-identical to Stage 1 (no regressions)
# ---------------------------------------------------------------------------

class TestRewardBackwardsCompat:
    def test_default_call_does_not_change_total(self):
        stats = AgentStats(orders_placed=5, holds=10)
        without = compute_reward(
            cash_final=10_500.0,
            shares_final=10,
            true_value=51.0,
            stats=stats,
        )
        with_aux_off = compute_reward(
            cash_final=10_500.0,
            shares_final=10,
            true_value=51.0,
            stats=stats,
            private_signals={"earnings": 2.0},
            aux_direction_weight=0.0,
        )
        assert without.total == with_aux_off.total
        assert without.aux_direction_bonus == 0.0

    def test_aux_only_fires_when_explicitly_enabled(self):
        stats = AgentStats(orders_placed=5)
        with_aux = compute_reward(
            cash_final=10_500.0,
            shares_final=10,
            true_value=51.0,
            stats=stats,
            private_signals={"earnings": 5.0},
            aux_direction_weight=AUX_DIRECTION_BONUS,
        )
        assert with_aux.aux_direction_bonus == pytest.approx(AUX_DIRECTION_BONUS)
        assert with_aux.total > 0


# ---------------------------------------------------------------------------
# Environment integration: aux reward only affects the trainable agent
# ---------------------------------------------------------------------------

class TestEnvironmentAuxReward:
    def test_default_environment_has_zero_aux(self):
        env = MarketEnvironment()
        obs = env.reset(seed=0, difficulty="easy", episode_length=3)
        for _ in range(3):
            obs, _, done, info = env.step(obs.episode_id, MarketAction(action_type="hold"))
            if done:
                break
        bd = info["reward_breakdown"]
        assert bd["aux_direction_bonus"] == 0.0

    def test_aux_environment_records_bonus_for_trainable_agent(self):
        env = MarketEnvironment(aux_direction_weight=0.10)
        obs = env.reset(seed=0, difficulty="easy", episode_length=3)
        # Place at least one buy so position != 0; aux bonus only fires when
        # net position is signed.
        action = MarketAction(action_type="buy", price=55.0, quantity=20)
        obs, _, _, _ = env.step(obs.episode_id, action)
        for _ in range(2):
            obs, _, done, info = env.step(obs.episode_id, MarketAction(action_type="hold"))
            if done:
                break
        bd = info["reward_breakdown"]
        # bonus is in [0, 0.10]; whether non-zero depends on signal sign vs trade
        assert 0.0 <= bd["aux_direction_bonus"] <= 0.10
        # And opponents (in all_agent_rewards) must have aux_direction_bonus = 0
        all_rewards = info["all_agent_rewards"]
        for agent_id, br in all_rewards.items():
            if agent_id != "agent_1":
                assert br["aux_direction_bonus"] == 0.0


# ---------------------------------------------------------------------------
# AblationConfig + scheduler
# ---------------------------------------------------------------------------

class TestAblationConfig:
    def test_known_presets_exist(self):
        for name in ("baseline_replay", "no_curriculum", "aux_direction"):
            cfg = ablations.get_preset(name)
            assert cfg.name == name

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="unknown ablation preset"):
            ablations.get_preset("does_not_exist")

    def test_baseline_matches_stage1(self):
        cfg = ablations.get_preset("baseline_replay")
        assert cfg.use_curriculum is True
        assert cfg.aux_direction_weight == 0.0

    def test_no_curriculum_disables_curriculum(self):
        cfg = ablations.get_preset("no_curriculum")
        assert cfg.use_curriculum is False

    def test_aux_direction_enables_bonus(self):
        cfg = ablations.get_preset("aux_direction")
        assert cfg.aux_direction_weight > 0.0

    def test_to_dict_is_json_safe(self):
        import json
        cfg = ablations.get_preset("aux_direction")
        json.dumps(cfg.to_dict())  # must not raise


class TestDifficultyScheduler:
    def test_curriculum_scheduler_returns_easy_at_step_0(self):
        cfg = ablations.get_preset("baseline_replay")
        scheduler = ablations.make_difficulty_scheduler(cfg)
        assert scheduler(0, random.Random(0)) == "easy"

    def test_no_curriculum_scheduler_always_returns_medium(self):
        cfg = ablations.get_preset("no_curriculum")
        scheduler = ablations.make_difficulty_scheduler(cfg)
        rng = random.Random(0)
        for step in (0, 50, 500, 1500, 2999):
            assert scheduler(step, rng) == "medium"

    def test_eval_seeds_outside_training_range(self):
        cfg = ablations.get_preset("baseline_replay")
        # SFT teacher uses seeds 0-499; eval seeds must not overlap.
        assert all(s >= 100 for s in cfg.eval_seeds)
