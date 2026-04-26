"""Frame-by-frame market replay for the dashboard.

Runs a single ``MarketEnvironment`` episode and records the public state of
the order book, every agent's position, and the action played by the
"surrogate trainable agent" each turn. The output is a list of frames that
the Streamlit app animates with a play button / scrub slider.

Why this exists
---------------
The actual GRPO-trained Qwen-3B is too slow for live demos on CPU
(~30 s/turn). Instead we let the user pick one of the *scripted bots* as
the trainable agent — Random for "dumb baseline", Informed for "oracle
upper bound". Side-by-side runs of the same scenario with different
surrogates give the audience a clean visual story:

    seed 100 :: Random   →  P&L  -0.06
    seed 100 :: Informed →  P&L  +0.21
    seed 100 :: Trained  →  (numeric placeholder until adapter rollout exists)

No LoRA / GPU dependencies are imported here, so the dashboard runs on a
laptop without CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from market_env.bots import (
    Bot,
    InformedBot,
    MarketMakerBot,
    MeanReversionBot,
    MomentumBot,
    RandomBot,
)
from market_env.environment import MarketEnvironment
from market_env.models import INITIAL_CASH, MarketAction, MarketObservation


# ---------------------------------------------------------------------------
# Frame data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BookLevel:
    price: float
    quantity: int


@dataclass(frozen=True)
class TradeEvent:
    price: float
    quantity: int
    aggressor_side: str  # 'buy' or 'sell'


@dataclass(frozen=True)
class AgentSnapshot:
    agent_id: str
    cash: float
    shares: int
    pnl_marked: float  # cash + shares * mid - INITIAL_CASH


@dataclass(frozen=True)
class Frame:
    """One turn of a market episode, ready for plotting."""

    turn: int
    best_bid: Optional[float]
    best_ask: Optional[float]
    mid: Optional[float]
    bids_top5: list[BookLevel]
    asks_top5: list[BookLevel]
    last_trade_price: Optional[float]
    new_trades: list[TradeEvent]
    agents: list[AgentSnapshot]
    surrogate_action: dict[str, Any]  # action dict for the chosen surrogate
    true_value: Optional[float]       # only populated on the final frame
    done: bool


# ---------------------------------------------------------------------------
# Surrogate selection
# ---------------------------------------------------------------------------

SURROGATES = {
    "Random (dumb baseline)": "random",
    "Informed (oracle upper bound)": "informed",
    "Hold (do-nothing)": "hold",
    "Momentum (trend follower)": "momentum",
    "Mean-reversion (contrarian)": "mean_rev",
}


def _make_surrogate(kind: str, scenario_seed: int, true_value: float) -> Bot:
    """Build a scripted bot that plays the role of the trainable agent."""
    if kind == "random":
        return RandomBot("agent_1", seed=scenario_seed + 999)
    if kind == "momentum":
        return MomentumBot("agent_1", seed=scenario_seed + 998)
    if kind == "mean_rev":
        return MeanReversionBot("agent_1", anchor=true_value)  # rough anchor
    if kind == "hold":
        return _HoldBot("agent_1")
    if kind == "informed":
        bot = InformedBot("agent_1", seed=scenario_seed + 997)
        bot.set_true_value(true_value)
        return bot
    raise ValueError(f"unknown surrogate: {kind!r}")


class _HoldBot(Bot):
    def act(self, observation: MarketObservation) -> MarketAction:  # noqa: D401
        return MarketAction(action_type="hold")


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def _snapshot_agents(env: MarketEnvironment, episode_id: str, mid: Optional[float]) -> list[AgentSnapshot]:
    state = env.state(episode_id)
    snaps: list[AgentSnapshot] = []
    mark = mid if mid is not None else 50.0
    for agent_id, pos in state["positions"].items():
        cash = pos["cash"]
        shares = pos["shares_held"]
        pnl_marked = cash + shares * mark - INITIAL_CASH
        snaps.append(
            AgentSnapshot(
                agent_id=agent_id,
                cash=cash,
                shares=shares,
                pnl_marked=pnl_marked,
            )
        )
    return snaps


def _book_levels(snapshot, depth: int = 5) -> tuple[list[BookLevel], list[BookLevel]]:
    bids = [BookLevel(price=lv.price, quantity=lv.quantity) for lv in snapshot.bids[:depth]]
    asks = [BookLevel(price=lv.price, quantity=lv.quantity) for lv in snapshot.asks[:depth]]
    return bids, asks


def _action_summary(action: MarketAction) -> dict[str, Any]:
    out: dict[str, Any] = {"type": action.action_type}
    if action.price is not None:
        out["price"] = round(action.price, 2)
    if action.quantity is not None:
        out["quantity"] = action.quantity
    if action.order_id is not None:
        out["order_id"] = action.order_id
    return out


def run_replay(
    *,
    seed: int,
    difficulty: str = "medium",
    bot_config: str = "default",
    surrogate: str = "informed",
    max_turns: int = 50,
) -> dict[str, Any]:
    """Run a full episode and return all frames + episode meta.

    Returns
    -------
    dict with keys:
        ``frames`` — list of Frame objects (one per turn after step 0)
        ``initial_frame`` — Frame at turn 0 before any actions
        ``true_value`` — float (revealed for the demo)
        ``surrogate`` — kind string echoed back
        ``seed`` / ``difficulty`` — echoed back
        ``final_pnl`` — dict[agent_id -> normalized PnL]
    """
    env = MarketEnvironment()
    obs0 = env.reset(
        seed=seed,
        difficulty=difficulty,
        bot_config=bot_config,
        episode_length=max_turns,
    )
    episode_id = obs0.episode_id

    # We need the true value to seed the InformedBot surrogate. The
    # environment hides it from the public observation — pull it directly
    # from the underlying state.
    state0 = env._episodes[episode_id]
    true_value = float(state0.scenario.true_value)

    surrogate_bot = _make_surrogate(surrogate, seed, true_value)

    initial_book = state0.book.get_snapshot(depth=10)
    initial_bids, initial_asks = _book_levels(initial_book, depth=5)
    initial_frame = Frame(
        turn=0,
        best_bid=initial_book.bids[0].price if initial_book.bids else None,
        best_ask=initial_book.asks[0].price if initial_book.asks else None,
        mid=initial_book.mid_price,
        bids_top5=initial_bids,
        asks_top5=initial_asks,
        last_trade_price=None,
        new_trades=[],
        agents=_snapshot_agents(env, episode_id, initial_book.mid_price),
        surrogate_action={"type": "init"},
        true_value=None,
        done=False,
    )

    frames: list[Frame] = []
    last_trade_price: Optional[float] = None
    # Note: ``OrderBook._recent_trades`` is append-only (no trimming), so we
    # use it as a complete trade log for the replay. Reading the attribute
    # is intentional — the public ``get_recent_trades`` only returns the
    # last N which is awkward when N is unknown ahead of time.
    prev_trade_count = len(state0.book._recent_trades)

    obs = obs0
    for _turn in range(max_turns):
        action = surrogate_bot.act(obs)
        obs, reward, done, info = env.step(episode_id, action)

        state = env._episodes[episode_id]
        snapshot = state.book.get_snapshot(depth=10)
        bids5, asks5 = _book_levels(snapshot, depth=5)

        # Trades that closed during this step
        new_trades_records = state.book._recent_trades[prev_trade_count:]
        prev_trade_count = len(state.book._recent_trades)
        new_trades: list[TradeEvent] = [
            TradeEvent(
                price=t.price,
                quantity=t.quantity,
                aggressor_side=t.aggressor_side,
            )
            for t in new_trades_records
        ]
        if new_trades:
            last_trade_price = new_trades[-1].price

        frames.append(
            Frame(
                turn=state.turn,
                best_bid=snapshot.bids[0].price if snapshot.bids else None,
                best_ask=snapshot.asks[0].price if snapshot.asks else None,
                mid=snapshot.mid_price,
                bids_top5=bids5,
                asks_top5=asks5,
                last_trade_price=last_trade_price,
                new_trades=new_trades,
                agents=_snapshot_agents(env, episode_id, snapshot.mid_price),
                surrogate_action=_action_summary(action),
                true_value=true_value if done else None,
                done=done,
            )
        )
        if done:
            break

    final_pnl: dict[str, float] = {}
    for snap in frames[-1].agents:
        # Mark to true value for the headline final number
        marked = snap.cash + snap.shares * true_value - INITIAL_CASH
        final_pnl[snap.agent_id] = marked / INITIAL_CASH

    return {
        "frames": frames,
        "initial_frame": initial_frame,
        "true_value": true_value,
        "surrogate": surrogate,
        "seed": seed,
        "difficulty": difficulty,
        "bot_config": bot_config,
        "final_pnl": final_pnl,
    }


def run_three_way(
    *,
    seed: int,
    difficulty: str = "medium",
    bot_config: str = "default",
    max_turns: int = 50,
) -> dict[str, dict[str, Any]]:
    """Run the same scenario with three different surrogates for side-by-side."""
    return {
        "random": run_replay(
            seed=seed, difficulty=difficulty, bot_config=bot_config,
            surrogate="random", max_turns=max_turns,
        ),
        "informed": run_replay(
            seed=seed, difficulty=difficulty, bot_config=bot_config,
            surrogate="informed", max_turns=max_turns,
        ),
        "hold": run_replay(
            seed=seed, difficulty=difficulty, bot_config=bot_config,
            surrogate="hold", max_turns=max_turns,
        ),
    }
