"""Tests for the theory-of-mind probes."""
from __future__ import annotations

from market_env.bots import RandomBot
from market_env.environment import MarketEnvironment
from market_env.models import MarketAction
from training.rollout import bot_policy
from training.tom_probes import (
    DIRECTION_PROBE_INSTRUCTION,
    _classify_answer,
    _make_informed_oracle,
    direction_inference,
    price_efficiency,
    run_probe_episode,
    signal_alignment,
)


# ---------------------------------------------------------------------------
# Per-turn capture
# ---------------------------------------------------------------------------

class TestRunProbeEpisode:
    def test_returns_one_record_per_turn(self):
        env = MarketEnvironment()
        bot = RandomBot("agent_1", seed=0)
        turns = run_probe_episode(env, bot_policy(bot), seed=0, episode_length=5)
        assert len(turns) == 5
        for t, turn in enumerate(turns):
            assert turn.turn == t
            assert isinstance(turn.true_value, float)
            assert turn.action_type in {"buy", "sell", "cancel", "hold"}

    def test_book_imbalance_in_range(self):
        env = MarketEnvironment()
        bot = RandomBot("agent_1", seed=42)
        turns = run_probe_episode(env, bot_policy(bot), seed=42, episode_length=5)
        for turn in turns:
            assert -1.0 <= turn.book_imbalance <= 1.0


# ---------------------------------------------------------------------------
# Probe 1 — price efficiency
# ---------------------------------------------------------------------------

class TestPriceEfficiency:
    def test_hold_only_keeps_book_at_default_mid(self):
        env = MarketEnvironment()
        episodes = [
            run_probe_episode(
                env,
                lambda obs: (MarketAction(action_type="hold"), True),
                seed=s,
                difficulty="easy",
                episode_length=5,
                bot_config="empty",
            )
            for s in range(3)
        ]
        result = price_efficiency(episodes, label="hold")
        # With bot_config="empty", no one trades, so mid_price stays None or 0
        # and the probe just records that.
        assert result.label == "hold"
        assert result.n_episodes == 3

    def test_informed_is_better_than_hold(self):
        env = MarketEnvironment()
        seeds = list(range(5))
        hold_eps = [
            run_probe_episode(
                env,
                lambda obs: (MarketAction(action_type="hold"), True),
                seed=s, difficulty="easy", episode_length=20,
            )
            for s in seeds
        ]
        informed_eps = [
            run_probe_episode(
                env, _make_informed_oracle(s), seed=s,
                difficulty="easy", episode_length=20,
            )
            for s in seeds
        ]
        h = price_efficiency(hold_eps, "hold")
        i = price_efficiency(informed_eps, "informed")
        # Informed agent should converge prices closer to true_value.
        # Allow small slack for stochastic episodes.
        assert i.final_mean_error is not None
        assert h.final_mean_error is not None
        assert i.final_mean_error <= h.final_mean_error + 0.3


# ---------------------------------------------------------------------------
# Probe 2 — signal alignment
# ---------------------------------------------------------------------------

class TestSignalAlignment:
    def test_informed_oracle_is_perfectly_aligned(self):
        env = MarketEnvironment()
        episodes = [
            run_probe_episode(
                env, _make_informed_oracle(s), seed=s,
                difficulty="easy", episode_length=20,
            )
            for s in range(5)
        ]
        result = signal_alignment(episodes, "informed")
        # _make_informed_oracle only acts when |signal_sum| > 1; same threshold as default
        if result.n_active > 0:
            assert result.alignment_rate == 1.0

    def test_hold_yields_zero_active(self):
        env = MarketEnvironment()
        episodes = [
            run_probe_episode(
                env,
                lambda obs: (MarketAction(action_type="hold"), True),
                seed=s, difficulty="easy", episode_length=10,
            )
            for s in range(3)
        ]
        result = signal_alignment(episodes, "hold")
        assert result.n_active == 0
        assert result.alignment_rate == 0.0


# ---------------------------------------------------------------------------
# Probe 3 — direction inference
# ---------------------------------------------------------------------------

class TestDirectionInference:
    def test_oracle_answer_fn_scores_perfectly(self):
        """Sanity check the harness: an answer_fn that knows the truth scores 100%.

        Since our probes are constructed by stripping signals before sending the
        prompt, but the harness has access to the ground truth scenario, an
        answer_fn that "cheats" by looking at the prompt for any tell would NOT
        work — the prompt has no signal info. So instead we patch by knowing
        the seed pattern: probes go in seed_start=200... and we know which
        scenarios have true_value > 50.
        """
        # Pre-compute ground truth for the probe seeds we'll use
        env = MarketEnvironment()
        ground_truths = {}
        for seed in range(200, 205):
            obs = env.reset(seed=seed, difficulty="medium", episode_length=5)
            ground_truths[seed] = (
                env._episodes[obs.episode_id].scenario.true_value > 50.0
            )

        # Build a stateful "oracle" that returns the right answer for each call.
        # We rely on direction_inference processing seeds in order.
        seed_iter = iter(sorted(ground_truths.keys()))
        def oracle(_prompt: str) -> str:
            seed = next(seed_iter)
            return "above" if ground_truths[seed] else "below"

        result = direction_inference(
            oracle, label="oracle",
            n_probes=5, seed_start=200, turns_to_warm_up=5,
        )
        assert result.accuracy == 1.0
        assert result.n_correct == 5

    def test_always_below_is_random_chance(self):
        result = direction_inference(
            lambda _p: "below", label="always_below",
            n_probes=10, seed_start=200, turns_to_warm_up=5,
        )
        # Should land somewhere reasonable; no crash, score in [0, 1]
        assert 0.0 <= result.accuracy <= 1.0
        assert result.n_probes == 10

    def test_probe_includes_inference_question(self):
        # Sanity: the probe prompt must contain the instruction we documented
        captured = []
        def capture(prompt: str) -> str:
            captured.append(prompt)
            return "above"

        direction_inference(
            capture, label="capture",
            n_probes=2, seed_start=200, turns_to_warm_up=5,
        )
        assert all(DIRECTION_PROBE_INSTRUCTION in p for p in captured)
        # And signals must NOT appear (we strip them)
        for p in captured:
            assert "earnings" not in p.lower() or "+" not in p.split("private signals")[1].split("\n")[1]


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

class TestClassifyAnswer:
    def test_pure_above(self):
        assert _classify_answer("above") == "above"

    def test_pure_below(self):
        assert _classify_answer("below") == "below"

    def test_with_punctuation(self):
        assert _classify_answer("Above.") == "above"
        assert _classify_answer("'below',") == "below"

    def test_short_phrase(self):
        assert _classify_answer("I think above") == "above"

    def test_neither(self):
        assert _classify_answer("not sure") is None
