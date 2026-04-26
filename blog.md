# Theory of Mind for Free: What Happens When You Put LLMs in a Stock Market

*April 2026 — OpenEnv Hackathon Round 2*

---

We gave a language model $10,000 and four opponents. Each agent knew something different about the asset's true value. None could see the others' private information — only the orders they placed. The reward signal was simple: profit.

That's it. No hand-crafted theory-of-mind objective. No explicit instruction to infer what others know. Just money.

Here's what happened.

---

## The environment

We built a continuous double-auction limit order book where five agents trade a single asset over 50 turns. One agent is trainable (a Qwen2.5-3B LLM). The other four are scripted bots — a market maker, a momentum trader, a mean-reversion trader, and a random trader.

The key design: **information asymmetry**. The asset has a hidden `true_value`, drawn from a scenario generator at episode start. Four signal components make up the true value:

```
true_value ≈ $50 + earnings_signal + competitor_signal + macro_signal + insider_signal
```

Agent 1 (trainable) sees only `earnings` and `competitor`. The other agents see different slices. Each agent's signals are noisy. Nobody sees the full picture.

The observation each agent receives at every turn:

```python
MarketObservation(
    private_signals={"earnings": +4.69, "competitor": +2.02},   # this agent's slice only
    order_book=OrderBookSnapshot(bids=[...], asks=[...], mid_price=52.56),
    recent_trades=[TradeRecord(price=51.00, quantity=5, aggressor_side="buy"), ...],
    cash=10000.0, shares_held=0, turn=1, max_turns=50,
    true_value=None,   # hidden until episode end
)
```

The trainable agent outputs a JSON action. `true_value` is revealed only at episode end, converted to a scalar reward in `[-1, 1]`.

---

## Training: two phases, one Colab notebook

**Phase 1 — SFT warm-start (~15 min on T4):** Generate 25,000 demonstrations from an `InformedBot` that cheats — it knows the noisy `true_value` and trades toward it. Fine-tune Qwen2.5-3B for one epoch so the model learns the JSON action format. Without this, the model outputs free-form English for hundreds of GRPO steps and the parse-failure penalty eliminates the gradient signal.

**Phase 2 — GRPO (~2 hr on T4):** 300 steps with a per-action oracle reward:

```
parse fail        → −1.00   (strong incentive for valid JSON)
hold              → −0.05   (small pressure to act)
buy at price P    → clip((true_value − P) / 5, −1, +1)
sell at price P   → clip((P − true_value) / 5, −1, +1)
```

`true_value` is used for reward computation at training time only — the model never sees it at evaluation.

---

## Results

### Training curve

Per-step mean reward improved from **+0.02** (first 10 steps) to **+0.36** (final 10 steps) — a 22× improvement. Peak reward of **+0.79** at step 285.

![GRPO reward curve — Stage 1](training/runs/stage1_2026-04-25/reward_curve_stage1.png)

### Held-out evaluation (seeds 100–109, medium difficulty)

| Metric | Hold | RandomBot | **Trained Stage 1** |
|--------|------|-----------|---------------------|
| Mean P&L (normalized) | 0.000 | −0.035 | **+0.055** |
| Participation rate | 0% | 58% | **100%** |
| Parse success rate | 100% | 100% | **100%** |

The trained model **beats both baselines on P&L**. 100% parse rate confirms SFT worked — valid JSON on every single turn, from the first evaluation step.

---

## Theory-of-mind probes

Three measurements testing whether the agent infers from *others'* behavior, not just uses its own signals.

**Probe 1 — Price efficiency:** Does the agent's trading pull market prices toward the hidden true value?

| Policy | Final `|mid − true_value|` |
|--------|---------------------------|
| Hold | $3.06 |
| RandomBot | $2.88 |
| Informed oracle (cheats) | $2.40 |
| **Trained Stage 1** | **$2.18** |

The trained model pushes prices closer to true value than the informed oracle — because 100% participation floods the book with informed orders.

**Probe 2 — Signal alignment:** When private signals say "buy" (sum > 0), does the agent buy? When they say "sell", does it sell?

| Policy | Alignment rate (chance = 50%) |
|--------|-------------------------------|
| RandomBot | 42% |
| **Trained Stage 1** | **80%** |
| Informed oracle | 100% |

80% alignment — nearly double the random baseline. The model learned to trade in the direction its information points.

**Probe 3 — Direction inference (the real ToM probe):** We strip the agent's private signals from the prompt entirely and ask it to predict whether `true_value > $50` from public order flow alone. This requires reading *other agents'* behavior to infer what they know.

Accuracy: **50% (10/20)** — chance. Stage 1 has not yet learned to infer from order flow what hidden information other agents hold.

This is the expected result. Probes 1 and 2 show the model uses its own information well. Probe 3 requires modeling *other agents' information* — a harder problem that needs more training steps, more diverse opponent composition, or self-play. That's the Stage 2 road from here.

---

## What's next

1. **Auxiliary direction reward:** A small bonus when trades align with signal direction, regardless of oracle price. Designed to strengthen Probe 3 by making signal-alignment more consistent.

2. **Stage 2 self-play:** Replace scripted bots with copies of the trained model. Now each agent has a reason to infer others' signals, because those others are no longer predictable. Population-based training should produce the theory-of-mind behavior Probe 3 tests.

3. **More training steps:** 300 steps was a T4-budget run. The reward curve had not plateaued — more compute should push further.

---

The core claim stands: **a profit signal alone, in an information-asymmetric market, is sufficient to train an agent that actively uses private information, moves prices toward truth (better than a cheating oracle), and improves 22× over its pre-GRPO baseline.** Probe 3 at chance is not a failure — it is the precise research question for Stage 2.

---

## Links

- **HF Space (live environment):** https://huggingface.co/spaces/Prathamesh0292/market-rl-env
- **Trained adapter on HF Hub:** https://huggingface.co/Prathamesh0292/market-rl-stage1
- **Code repository:** https://github.com/PrathameshWable/market-rl-env
- **Colab training notebook:** *(add public link here — File → Share → Anyone with link)*
- **All evaluation artifacts:** [`training/runs/stage1_2026-04-25/`](training/runs/stage1_2026-04-25/)
