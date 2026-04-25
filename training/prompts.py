"""
Prompt formatting and action parsing for LLM agents.

Three jobs:
1. format_system_message()  — fixed instruction that frames the task
2. format_observation(obs)  — render a MarketObservation as a user message
3. parse_action(text)       — robustly turn model output into a MarketAction
                              (falls back to "hold" on any failure)

Also exposes serialize_action() for the SFT data generator: it writes the
exact JSON string the model is supposed to emit, so SFT teaches the model
to imitate this format.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from market_env.models import MarketAction, MarketObservation


# ---------------------------------------------------------------------------
# System prompt  (sent once per conversation; kept short to save tokens)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a profit-maximizing trader in a multi-agent stock market.

GOAL
Maximize your final profit measured at the asset's hidden true value, which is
revealed only at episode end. true_value ≈ $50.00 + sum of signal components.
You see only some signal components; other agents see different ones.

ACTIONS (output ONE JSON object per turn, no prose, no markdown)
  {"action_type": "buy",  "price": 50.50, "quantity": 5}
  {"action_type": "sell", "price": 51.00, "quantity": 5}
  {"action_type": "cancel", "order_id": "ord_abc123"}
  {"action_type": "hold"}

RULES
- Buy/sell are limit orders. Price must be > 0 and quantity a positive integer.
- A buy at or above the best ask executes immediately at the ask.
- A sell at or below the best bid executes immediately at the bid.
- Otherwise the order rests on the book until filled or cancelled.
- Cancel removes one of YOUR resting orders by order_id.
- Hold does nothing this turn.

STRATEGY
Your private signals hint at where true_value sits vs. $50. Trade in the
direction your signals suggest. Adjust for the order flow you see —
other agents have their own signals and their orders leak information.
"""


# ---------------------------------------------------------------------------
# Observation rendering
# ---------------------------------------------------------------------------

def _format_signals(signals: dict[str, float]) -> str:
    if not signals:
        return "  (none)"
    return "\n".join(f"  {name}: {value:+.2f}" for name, value in signals.items())


def _format_book_side(levels, label: str) -> str:
    if not levels:
        return f"  {label}: (empty)"
    parts = [f"${lvl.price:.2f} x{lvl.quantity}" for lvl in levels[:5]]
    return f"  {label}: " + "  ".join(parts)


def _format_recent_trades(trades) -> str:
    if not trades:
        return "  (no trades yet)"
    lines = []
    for t in trades[-5:]:
        lines.append(f"  ${t.price:.2f} x{t.quantity}  ({t.aggressor_side} aggressor)")
    return "\n".join(lines)


def _format_open_orders(orders) -> str:
    if not orders:
        return "  (none)"
    lines = []
    for o in orders:
        lines.append(
            f"  {o.order_id}: {o.side.upper()} ${o.price:.2f} "
            f"x{o.quantity} (filled {o.filled}/{o.quantity})"
        )
    return "\n".join(lines)


def format_observation(obs: MarketObservation) -> str:
    """Render a MarketObservation as the user message for the model."""
    book = obs.order_book
    mid = f"${book.mid_price:.2f}" if book.mid_price else "n/a"
    spread = f"${book.spread:.2f}" if book.spread else "n/a"

    return (
        f"Turn {obs.turn}/{obs.max_turns}\n"
        f"Cash: ${obs.cash:.2f} | Shares: {obs.shares_held:+d}\n"
        f"\n"
        f"Your private signals:\n"
        f"{_format_signals(obs.private_signals)}\n"
        f"\n"
        f"Order book (top 5):\n"
        f"{_format_book_side(book.asks, 'Asks')}\n"
        f"{_format_book_side(book.bids, 'Bids')}\n"
        f"  Mid: {mid}   Spread: {spread}\n"
        f"\n"
        f"Recent trades:\n"
        f"{_format_recent_trades(obs.recent_trades)}\n"
        f"\n"
        f"Your open orders:\n"
        f"{_format_open_orders(obs.open_orders)}\n"
    )


# ---------------------------------------------------------------------------
# Action serialization (used by SFT data gen)
# ---------------------------------------------------------------------------

def serialize_action(action: MarketAction) -> str:
    """Render an action as the exact JSON string the model is supposed to emit.

    Only includes fields relevant to the action_type, so SFT examples stay
    minimal and we don't teach the model to emit None for irrelevant keys.
    """
    if action.action_type == "buy" or action.action_type == "sell":
        body = {
            "action_type": action.action_type,
            "price": round(action.price, 2) if action.price is not None else None,
            "quantity": action.quantity,
        }
    elif action.action_type == "cancel":
        body = {"action_type": "cancel", "order_id": action.order_id}
    else:  # hold
        body = {"action_type": "hold"}
    return json.dumps(body, separators=(", ", ": "))


# ---------------------------------------------------------------------------
# Action parsing  (model output → MarketAction, never raises)
# ---------------------------------------------------------------------------

# Strip ```json ... ``` or ``` ... ``` wrappers if the model adds them.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_block(text: str) -> Optional[str]:
    """Pull the first balanced {...} substring out of the model output.

    Tolerates surrounding prose, markdown fences, and trailing commentary.
    Returns None if no plausible JSON object is present.
    """
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1)

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_action(text: str) -> tuple[MarketAction, bool]:
    """Parse model output into a MarketAction.

    Returns (action, parse_ok). On any failure — missing JSON, bad JSON,
    invalid fields — returns (hold, False). The caller is expected to
    increment AgentStats.parse_failures when parse_ok is False so the
    reward function can apply the parse-failure penalty.
    """
    block = _extract_json_block(text)
    if block is None:
        return MarketAction(action_type="hold"), False

    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return MarketAction(action_type="hold"), False

    if not isinstance(data, dict):
        return MarketAction(action_type="hold"), False

    # Drop anything Pydantic doesn't know about — keep parser permissive.
    allowed = {"action_type", "price", "quantity", "order_id", "reasoning"}
    cleaned = {k: v for k, v in data.items() if k in allowed}

    try:
        return MarketAction(**cleaned), True
    except Exception:
        return MarketAction(action_type="hold"), False
