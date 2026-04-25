"""
Theory-of-mind probes for the trained Stage 1 agent.

Three measurements, each designed to test a different aspect of the
"agent infers from others' behavior" claim:

    Probe 1 — Price Efficiency
        Hypothesis: a market populated by an *informed* policy converges
        toward the asset's true value faster than one populated by an
        uninformed (random/hold) policy. The agent's information leaks
        into prices through its orders.
        Measurement: mean |mid_price − true_value| at each turn,
        averaged across N episodes per policy.

    Probe 2 — Signal Alignment
        Hypothesis: an agent that uses its private information will buy
        more often when its signal sum is positive and sell more often
        when it is negative. RandomBot scores ~50%; an informed policy
        should score noticeably higher.
        Measurement: among "active" turns (buy or sell) where the
        absolute signal sum exceeds a threshold, the fraction that
        traded in the direction the signals predicted.

    Probe 3 — Direction Inference  (the actual theory-of-mind probe)
        Hypothesis: WITHOUT its private signals, an agent that has
        learned to read other agents' order flow should be able to
        predict whether true_value sits above or below $50 at better
        than chance accuracy.
        Measurement: take K turns of market activity driven by other
        bots, hand the trained model a SIGNAL-FREE observation snapshot
        plus a yes/no question, and report binary accuracy across N
        probes vs. the 50% chance baseline.

Probes 1 and 2 run on any policy (scripted or LLM) and so are runnable
locally without a GPU. Probe 3 requires an LLM callable; it lives behind
the `direction_inference` API and is invoked from the Colab eval cell.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Optional

from market_env.bots import MarketMakerBot, RandomBot
from market_env.environment import MarketEnvironment
from market_env.models import MarketAction, MarketObservation

from training.prompts import format_observation
from training.rollout import Policy, bot_policy


# ---------------------------------------------------------------------------
# Per-turn data capture
# ---------------------------------------------------------------------------

@dataclass
class ProbeTurn:
    turn: int
    mid_price: Optional[float]
    true_value: float
    signal_sum: float
    action_type: str
    action_price: Optional[float]
    action_quantity: Optional[int]
    book_imbalance: float    # (bid_qty - ask_qty) / (bid_qty + ask_qty); -1..+1


def _book_imbalance(snapshot) -> float:
    bid_qty = sum(level.quantity for level in snapshot.bids[:5])
    ask_qty = sum(level.quantity for level in snapshot.asks[:5])
    total = bid_qty + ask_qty
    if total == 0:
        return 0.0
    return (bid_qty - ask_qty) / total


def run_probe_episode(
    env: MarketEnvironment,
    policy: Policy,
    *,
    seed: int,
    difficulty: str = "medium",
    episode_length: int = 50,
    bot_config: str = "default",
    trainable_agent_id: str = "agent_1",
) -> list[ProbeTurn]:
    """Run one episode and return a per-turn record suitable for analysis."""
    obs = env.reset(
        seed=seed,
        difficulty=difficulty,
        episode_length=episode_length,
        bot_config=bot_config,
        trainable_agent_id=trainable_agent_id,
    )
    true_value = env._episodes[obs.episode_id].scenario.true_value

    turns: list[ProbeTurn] = []
    done = False
    while not done:
        action, _ = policy(obs)
        turns.append(ProbeTurn(
            turn=obs.turn,
            mid_price=obs.order_book.mid_price or None,
            true_value=true_value,
            signal_sum=sum(obs.private_signals.values()),
            action_type=action.action_type,
            action_price=action.price,
            action_quantity=action.quantity,
            book_imbalance=_book_imbalance(obs.order_book),
        ))
        obs, _, done, _ = env.step(obs.episode_id, action)

    return turns


# ---------------------------------------------------------------------------
# Probe 1 — Price efficiency
# ---------------------------------------------------------------------------

@dataclass
class PriceEfficiency:
    label: str
    n_episodes: int
    mean_error_by_turn: list[Optional[float]]   # length = episode_length
    final_mean_error: Optional[float]
    initial_mean_error: Optional[float]
    improvement: Optional[float]                # initial - final (positive = converged toward TV)


def price_efficiency(
    episodes: list[list[ProbeTurn]],
    label: str,
) -> PriceEfficiency:
    if not episodes:
        return PriceEfficiency(label, 0, [], None, None, None)

    n_turns = max(len(ep) for ep in episodes)
    per_turn: list[list[float]] = [[] for _ in range(n_turns)]

    for ep in episodes:
        tv = ep[0].true_value
        for t, turn in enumerate(ep):
            if turn.mid_price is not None:
                per_turn[t].append(abs(turn.mid_price - tv))

    mean_by_turn: list[Optional[float]] = [
        statistics.mean(errs) if errs else None for errs in per_turn
    ]
    valid = [m for m in mean_by_turn if m is not None]
    final = mean_by_turn[-1] if mean_by_turn and mean_by_turn[-1] is not None else None
    initial = next((m for m in mean_by_turn if m is not None), None)
    improvement = (initial - final) if (initial is not None and final is not None) else None

    return PriceEfficiency(
        label=label,
        n_episodes=len(episodes),
        mean_error_by_turn=mean_by_turn,
        final_mean_error=final,
        initial_mean_error=initial,
        improvement=improvement,
    )


# ---------------------------------------------------------------------------
# Probe 2 — Signal alignment
# ---------------------------------------------------------------------------

@dataclass
class SignalAlignment:
    label: str
    n_active: int            # turns counted (buy or sell, with strong-enough signal)
    n_aligned: int           # acted in the direction signals suggested
    n_misaligned: int
    alignment_rate: float    # n_aligned / n_active   (random ~ 0.5)
    threshold: float


def signal_alignment(
    episodes: list[list[ProbeTurn]],
    label: str,
    *,
    threshold: float = 1.0,   # ignore turns where |signal_sum| < this
) -> SignalAlignment:
    aligned = misaligned = 0
    for ep in episodes:
        for turn in ep:
            if abs(turn.signal_sum) < threshold:
                continue
            if turn.action_type == "buy":
                if turn.signal_sum > 0:
                    aligned += 1
                else:
                    misaligned += 1
            elif turn.action_type == "sell":
                if turn.signal_sum < 0:
                    aligned += 1
                else:
                    misaligned += 1

    n_active = aligned + misaligned
    rate = aligned / n_active if n_active else 0.0
    return SignalAlignment(
        label=label,
        n_active=n_active,
        n_aligned=aligned,
        n_misaligned=misaligned,
        alignment_rate=rate,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Probe 3 — Direction inference (the ToM probe; needs the LLM)
# ---------------------------------------------------------------------------

DIRECTION_PROBE_INSTRUCTION = (
    "\n\n=== INFERENCE QUESTION ===\n"
    "Based ONLY on the order book and recent trades above, do you think "
    "the asset's hidden true value is ABOVE or BELOW $50.00?\n"
    "Answer with exactly one word: 'above' or 'below'.\n"
    "Do not output JSON. Do not explain. One word only."
)


@dataclass
class DirectionProbe:
    seed: int
    true_value: float
    ground_truth: str        # 'above' or 'below'
    model_answer: str        # whatever the model said (cleaned)
    correct: bool


@dataclass
class DirectionInference:
    label: str
    n_probes: int
    n_correct: int
    accuracy: float          # n_correct / n_probes  (chance = 0.5)
    chance_baseline: float = 0.5
    probes: list[DirectionProbe] = field(default_factory=list)


def _strip_signals_from_observation(obs: MarketObservation) -> MarketObservation:
    """Return a copy of obs with private_signals zeroed and signal_names blank."""
    return obs.model_copy(update={
        "private_signals": {},
        "signal_names": [],
    })


def _classify_answer(text: str) -> Optional[str]:
    """Very forgiving parser: look for 'above' / 'below' as the first match."""
    t = text.strip().lower()
    # First, common one-word answers
    for word in t.split():
        clean = word.strip(".,!?'\"")
        if clean in ("above", "below"):
            return clean
    # Substring fallback
    if "above" in t and "below" not in t:
        return "above"
    if "below" in t and "above" not in t:
        return "below"
    return None


def direction_inference(
    answer_fn: Callable[[str], str],
    *,
    label: str,
    n_probes: int = 20,
    seed_start: int = 200,
    difficulty: str = "medium",
    turns_to_warm_up: int = 25,
    bot_config: str = "default",
) -> DirectionInference:
    """ToM probe: ask the model to predict true_value direction from public data only.

    Args:
        answer_fn: callable taking a probe prompt string, returning the model's
                   one-word answer ('above' or 'below', possibly with extra
                   text — the parser is forgiving). For local testing, pass
                   a hand-rolled oracle. In the Colab notebook, pass a lambda
                   that calls model.generate.
        n_probes: number of independent probes (one episode each).
        seed_start: first scenario seed; the eval seed range (100-109) is
                    avoided so this doesn't double-count.
        turns_to_warm_up: how many turns of bot activity to run before
                          asking the model. More turns = more order flow
                          to read, easier inference.
    """
    env = MarketEnvironment()
    probes: list[DirectionProbe] = []
    correct = 0
    rng = random.Random(seed_start)

    for i in range(n_probes):
        seed = seed_start + i
        # Use a MarketMaker so there's two-sided liquidity to read
        bot = MarketMakerBot("agent_1", seed=seed)
        obs = env.reset(
            seed=seed,
            difficulty=difficulty,
            episode_length=turns_to_warm_up,
            bot_config=bot_config,
        )
        true_value = env._episodes[obs.episode_id].scenario.true_value

        # Step the env forward with the bot in the trainable slot, until
        # the last turn (so we have an observation with maximum order flow
        # but the episode is not yet done).
        for _ in range(turns_to_warm_up - 1):
            action = bot.act(obs)
            obs, _, done, _ = env.step(obs.episode_id, action)
            if done:
                break

        # Strip the agent's private signals before forming the probe prompt
        signal_free_obs = _strip_signals_from_observation(obs)
        probe_prompt = format_observation(signal_free_obs) + DIRECTION_PROBE_INSTRUCTION

        try:
            raw_answer = answer_fn(probe_prompt)
        except Exception as exc:   # pragma: no cover  (model errors are runtime)
            raw_answer = f"<error: {exc}>"

        parsed = _classify_answer(raw_answer) or "below"   # tie-break to "below"
        ground_truth = "above" if true_value > 50.0 else "below"
        is_correct = (parsed == ground_truth)

        probes.append(DirectionProbe(
            seed=seed,
            true_value=true_value,
            ground_truth=ground_truth,
            model_answer=parsed,
            correct=is_correct,
        ))
        if is_correct:
            correct += 1

    return DirectionInference(
        label=label,
        n_probes=n_probes,
        n_correct=correct,
        accuracy=correct / n_probes if n_probes else 0.0,
        probes=probes,
    )


# ---------------------------------------------------------------------------
# Multi-policy comparison harness (probes 1 & 2)
# ---------------------------------------------------------------------------

def compare_policies(
    policy_factories: dict[str, Callable[[int], Policy]],
    *,
    seeds: list[int],
    difficulty: str = "medium",
    episode_length: int = 50,
) -> dict[str, dict]:
    """Run each policy on the same seed set; return a dict of probe results.

    `policy_factories` maps a policy name to a (seed -> Policy) factory so
    each policy gets a fresh seeded RNG for each episode (matches the
    pattern used by training/evaluate.py).
    """
    env = MarketEnvironment()
    out: dict[str, dict] = {}

    for name, factory in policy_factories.items():
        all_episodes: list[list[ProbeTurn]] = []
        for seed in seeds:
            policy = factory(seed)
            episode_turns = run_probe_episode(
                env, policy, seed=seed,
                difficulty=difficulty, episode_length=episode_length,
            )
            all_episodes.append(episode_turns)

        out[name] = {
            "price_efficiency": asdict(price_efficiency(all_episodes, name)),
            "signal_alignment": asdict(signal_alignment(all_episodes, name)),
        }

    return out


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def make_price_efficiency_plot(
    results: dict[str, dict],
    out_path: Path,
) -> None:
    """One line per policy: |mid - true_value| over turns."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.5))
    for name, payload in results.items():
        pe = payload["price_efficiency"]
        ys = [v if v is not None else float("nan") for v in pe["mean_error_by_turn"]]
        xs = list(range(len(ys)))
        plt.plot(xs, ys, label=f"{name} (final={pe['final_mean_error']:.2f})", linewidth=2)
    plt.xlabel("turn")
    plt.ylabel("mean |mid_price − true_value|  ($)")
    plt.title("Probe 1 — Price efficiency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def make_signal_alignment_plot(
    results: dict[str, dict],
    out_path: Path,
) -> None:
    """Bar chart of alignment rate per policy with a 50% reference line."""
    import matplotlib.pyplot as plt

    names = list(results.keys())
    rates = [results[n]["signal_alignment"]["alignment_rate"] for n in names]
    n_active = [results[n]["signal_alignment"]["n_active"] for n in names]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, rates, color=["#888", "#888", "#2a7"])
    plt.axhline(0.5, color="red", linestyle="--", label="chance (50%)")
    for bar, na in zip(bars, n_active):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={na}",
            ha="center", fontsize=9,
        )
    plt.ylim(0, 1.05)
    plt.ylabel("alignment rate")
    plt.title("Probe 2 — Signal alignment\n(buy when signals + ; sell when −)")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_results(results: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


# ---------------------------------------------------------------------------
# CLI — runs probes 1 & 2 with scripted baselines (no GPU needed).
# ---------------------------------------------------------------------------

# A "cheating" informed baseline: trades on signals every turn, no inference
# needed. This is the upper bound for probes 1 and 2 under scripted policies.
def _make_informed_oracle(seed: int) -> Policy:
    """Policy that always trades in the direction signals suggest."""
    def policy(obs: MarketObservation) -> tuple[MarketAction, bool]:
        signal_sum = sum(obs.private_signals.values())
        ob = obs.order_book
        if signal_sum > 1.0 and ob.asks:
            return MarketAction(
                action_type="buy",
                price=ob.asks[0].price,
                quantity=10,
            ), True
        if signal_sum < -1.0 and ob.bids:
            return MarketAction(
                action_type="sell",
                price=ob.bids[0].price,
                quantity=10,
            ), True
        return MarketAction(action_type="hold"), True
    return policy


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("training/runs/stage1_2026-04-25"),
    )
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(100, 110)))
    args = parser.parse_args()

    print(f"Running ToM probes 1 & 2 on {len(args.seeds)} scenarios ...")
    factories = {
        "hold":     lambda _seed: lambda obs: (MarketAction(action_type="hold"), True),
        "random":   lambda seed:  bot_policy(RandomBot("agent_1", seed=seed)),
        "informed": _make_informed_oracle,
    }
    results = compare_policies(factories, seeds=args.seeds)

    save_results(results, args.out_dir / "tom_probes_baselines.json")
    make_price_efficiency_plot(results, args.out_dir / "tom_price_efficiency.png")
    make_signal_alignment_plot(results, args.out_dir / "tom_signal_alignment.png")

    print()
    for name, payload in results.items():
        pe = payload["price_efficiency"]
        sa = payload["signal_alignment"]
        print(f"  {name:8s} | "
              f"final |mid-tv|=${pe['final_mean_error']:.2f}  | "
              f"alignment={sa['alignment_rate']:.1%}  ({sa['n_active']} active turns)")
    print(f"\nSaved JSON + 2 plots to {args.out_dir}")


if __name__ == "__main__":
    main()
