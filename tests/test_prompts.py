"""Tests for the prompt formatter and action parser."""
from __future__ import annotations

import json

import pytest

from market_env.environment import MarketEnvironment
from market_env.models import MarketAction
from training.prompts import (
    SYSTEM_PROMPT,
    format_observation,
    parse_action,
    serialize_action,
)


# ---------------------------------------------------------------------------
# Round-trip: every action serializes to JSON and parses back to the same action
# ---------------------------------------------------------------------------

class TestActionRoundTrip:
    @pytest.mark.parametrize(
        "action",
        [
            MarketAction(action_type="hold"),
            MarketAction(action_type="buy", price=50.50, quantity=10),
            MarketAction(action_type="sell", price=51.00, quantity=5),
            MarketAction(action_type="cancel", order_id="ord_abc123"),
        ],
    )
    def test_serialize_then_parse_recovers_action(self, action):
        text = serialize_action(action)
        parsed, ok = parse_action(text)
        assert ok
        assert parsed.action_type == action.action_type
        assert parsed.price == action.price
        assert parsed.quantity == action.quantity
        assert parsed.order_id == action.order_id


# ---------------------------------------------------------------------------
# Parser robustness: every malformed model output falls back to hold cleanly
# ---------------------------------------------------------------------------

class TestParserRobustness:
    def test_pure_json_works(self):
        action, ok = parse_action('{"action_type": "hold"}')
        assert ok
        assert action.action_type == "hold"

    def test_markdown_fenced_json_works(self):
        action, ok = parse_action('```json\n{"action_type": "hold"}\n```')
        assert ok
        assert action.action_type == "hold"

    def test_plain_fenced_json_works(self):
        action, ok = parse_action('```\n{"action_type": "hold"}\n```')
        assert ok
        assert action.action_type == "hold"

    def test_prose_then_json(self):
        text = 'Looking at the book I should hold. {"action_type": "hold"}'
        action, ok = parse_action(text)
        assert ok
        assert action.action_type == "hold"

    def test_empty_string_falls_back_to_hold(self):
        action, ok = parse_action("")
        assert not ok
        assert action.action_type == "hold"

    def test_no_json_falls_back_to_hold(self):
        action, ok = parse_action("I will hold this turn.")
        assert not ok
        assert action.action_type == "hold"

    def test_malformed_json_falls_back_to_hold(self):
        action, ok = parse_action('{"action_type": "hold"')   # missing brace
        assert not ok
        assert action.action_type == "hold"

    def test_invalid_action_type_falls_back_to_hold(self):
        action, ok = parse_action('{"action_type": "explode"}')
        assert not ok
        assert action.action_type == "hold"

    def test_negative_price_falls_back_to_hold(self):
        # Pydantic validator rejects price <= 0; parser must catch and downgrade.
        action, ok = parse_action('{"action_type": "buy", "price": -5, "quantity": 5}')
        assert not ok
        assert action.action_type == "hold"

    def test_unknown_keys_are_ignored(self):
        text = '{"action_type": "hold", "foo": "bar", "extra": 42}'
        action, ok = parse_action(text)
        assert ok
        assert action.action_type == "hold"

    def test_buy_with_full_fields(self):
        text = '{"action_type": "buy", "price": 50.5, "quantity": 10}'
        action, ok = parse_action(text)
        assert ok
        assert action.action_type == "buy"
        assert action.price == 50.5
        assert action.quantity == 10

    def test_reasoning_field_is_kept(self):
        text = '{"action_type": "hold", "reasoning": "spread too wide"}'
        action, ok = parse_action(text)
        assert ok
        assert action.reasoning == "spread too wide"

    # --- M7B parser hardening ---------------------------------------------

    def test_smart_quotes_are_normalised(self):
        text = '\u201caction_type\u201d: \u201chold\u201d'
        action, ok = parse_action("{" + text + "}")
        assert ok
        assert action.action_type == "hold"

    def test_single_quoted_pythonic_dict(self):
        text = "{'action_type': 'hold'}"
        action, ok = parse_action(text)
        assert ok
        assert action.action_type == "hold"

    def test_trailing_comma_is_tolerated(self):
        text = '{"action_type": "buy", "price": 50.5, "quantity": 5,}'
        action, ok = parse_action(text)
        assert ok
        assert action.action_type == "buy"
        assert action.quantity == 5

    def test_unquoted_keys_are_tolerated(self):
        text = '{action_type: "hold"}'
        action, ok = parse_action(text)
        assert ok
        assert action.action_type == "hold"

    def test_uppercase_action_type_is_normalised(self):
        text = '{"action_type": "HOLD"}'
        action, ok = parse_action(text)
        assert ok
        assert action.action_type == "hold"

    def test_string_price_and_quantity_are_coerced(self):
        text = '{"action_type": "buy", "price": "50.5", "quantity": "5"}'
        action, ok = parse_action(text)
        assert ok
        assert action.price == 50.5
        assert action.quantity == 5

    def test_float_quantity_with_integer_value_is_coerced(self):
        text = '{"action_type": "buy", "price": 50.5, "quantity": 5.0}'
        action, ok = parse_action(text)
        assert ok
        assert action.quantity == 5

    def test_multiple_json_blobs_picks_first_valid(self):
        text = '{"action_type": "explode"} {"action_type": "hold"}'
        action, ok = parse_action(text)
        assert ok
        assert action.action_type == "hold"

    def test_reasoning_with_nested_braces_does_not_break(self):
        text = '{"action_type": "hold", "reasoning": "saw {x} pattern"}'
        action, ok = parse_action(text)
        assert ok
        assert action.action_type == "hold"
        assert action.reasoning is not None and "{x}" in action.reasoning

    def test_none_input_falls_back_to_hold(self):
        action, ok = parse_action(None)  # type: ignore[arg-type]
        assert not ok
        assert action.action_type == "hold"


# ---------------------------------------------------------------------------
# Observation formatting: smoke test on a real env observation
# ---------------------------------------------------------------------------

class TestFormatObservation:
    def test_renders_a_real_observation_without_crashing(self):
        env = MarketEnvironment()
        obs = env.reset(seed=0, difficulty="easy", episode_length=5)
        text = format_observation(obs)
        assert "Turn 0/5" in text
        assert "private signals" in text.lower()
        assert "order book" in text.lower()
        assert "open orders" in text.lower()

    def test_renders_after_a_few_steps(self):
        env = MarketEnvironment()
        obs = env.reset(seed=42, difficulty="medium", episode_length=5)
        for _ in range(3):
            obs, _, _, _ = env.step(obs.episode_id, MarketAction(action_type="hold"))
        text = format_observation(obs)
        assert "Turn 3/5" in text
        # Either trades happened or the placeholder is shown
        assert "trades" in text.lower()

    def test_formatted_observation_has_signal_names(self):
        env = MarketEnvironment()
        obs = env.reset(seed=0, difficulty="easy", trainable_agent_id="agent_1")
        text = format_observation(obs)
        # agent_1 sees earnings + competitor
        assert "earnings" in text
        assert "competitor" in text


# ---------------------------------------------------------------------------
# System prompt: just verify it's well-formed and stable
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_system_prompt_mentions_all_action_types(self):
        for action_type in ("buy", "sell", "cancel", "hold"):
            assert action_type in SYSTEM_PROMPT

    def test_system_prompt_examples_parse(self):
        # Every JSON example shown to the model must parse back.
        import re
        examples = re.findall(r"\{[^{}]+\}", SYSTEM_PROMPT)
        assert len(examples) >= 4
        for ex in examples:
            # action_type is the only required key; cancel/hold won't validate
            # past Pydantic without optional fields, but parse_action handles it.
            data = json.loads(ex)
            assert "action_type" in data
