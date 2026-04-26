# Stage 1 Training Report — Market RL Agent

**Date:** 2026-04-25/26  
**Model:** Qwen2.5-3B-Instruct + LoRA r=16 (4-bit, fp16)  
**Hardware:** Google Colab T4 (15 GB VRAM)  
**Total training time:** ~3 hours (SFT warm-start + GRPO)

---

## What we built

A profit-maximizing LLM agent that trades in a simulated multi-agent stock market.
The market has a hidden true value revealed only at episode end. Five agents compete:
four scripted bots (RandomBot, MarketMakerBot) and our trained LLM.

The agent receives a text observation each turn (order book, private signals, open
orders, recent trades) and must output a single JSON action:

```json
{"action_type": "buy", "price": 51.25, "quantity": 5}
```

Training used two phases:
1. **SFT warm-start** — 150 steps on 25k InformedBot demonstrations (teaches JSON format)
2. **GRPO** — 300 steps with a per-action oracle reward (teaches profitable trading)

---

## GRPO reward curve

![Reward curve](training/runs/stage1_2026-04-25/reward_curve_stage1.png)

The model started at ~0 mean reward (steps 0–50, cold start), then crossed into
positive territory at step 55 and stayed there for the remaining 245 steps.

| Window | Mean reward | Note |
|--------|-------------|------|
| Steps 0–50   | +0.017 | Cold start, half-negative |
| Steps 50–150 | +0.21  | Learning kicked in |
| Steps 150–300 | +0.32 | Sustained positive, peak +0.79 |
| **Last 10 steps** | **+0.362** | **22× improvement from step 0** |

---

## Evaluation results (10 held-out scenarios, seeds 100–109)

These seeds were never used during SFT data generation or GRPO prompt collection.

| Policy | Mean P&L (normalized) | Participation | Parse rate |
|--------|-----------------------|---------------|------------|
| Hold (do nothing) | 0.000 | 0% | 100% |
| RandomBot | −0.035 | 58.6% | 100% |
| **Trained agent** | **+0.055** | **100%** | **100%** |

The trained agent beats the random baseline by **9 percentage points** and the
hold baseline by **5.5 points** on unseen scenarios. Participation at 100% means
the model is always placing real orders — it never defaults to holding.

---

## Theory-of-Mind probes

We ran three probes to test whether the agent has learned to model other agents'
behavior.

### Probe 1 — Price efficiency

Does the agent's order placement push prices toward the hidden true value?

![Price efficiency](training/runs/stage1_2026-04-25/tom_price_efficiency.png)

| Policy | Initial \|mid − tv\| | Final \|mid − tv\| | Improvement |
|--------|----------------------|--------------------|-------------|
| Hold | $3.06 | $3.06 | $0.00 |
| RandomBot | $2.88 | $2.88 | $0.00 |
| Informed oracle | $2.40 | $2.40 | — |
| **Trained agent** | **$2.83** | **$2.18** | **$0.65** |

The trained agent's orders move prices **$0.65 closer** to true value by episode
end. Hold and random show no convergence. This is consistent with an agent that
knows where prices *should* be and trades toward them.

### Probe 2 — Signal alignment

![Signal alignment](training/runs/stage1_2026-04-25/tom_signal_alignment.png)

Of all turns where |signal_sum| > 1 (a strong directional hint):

| Policy | Alignment rate | Active turns |
|--------|----------------|--------------|
| RandomBot | 42% (≈ chance) | 149 |
| **Trained agent** | **80%** | **250** |
| Informed oracle | 100% | 237 |

The trained agent trades in the direction its private signals suggest **80% of
the time**, compared to 42% for RandomBot (chance). It sits between random and
the oracle — a clear signal that the model has learned to read and act on its
private information.

### Probe 3 — Direction inference (signal-free)

The real ToM test: strip the agent's private signals from the observation and
ask it to predict whether the hidden true value is above or below $50, using
only public market data (order book, trades, other agents' behavior).

**Result: 50% accuracy (10/20), exactly at chance.**

The per-probe breakdown reveals why: the model answered "above" for all 20
probes, regardless of the true value. When deprived of its private signals,
the model falls back to a systematic bias rather than reading order flow.

This is the honest Stage 1 limitation: the agent has learned to trade using
its own private information (Probe 2: 80% alignment) but has not yet developed
the ability to *infer* information from other agents' behavior without a
private anchor. This is a well-known challenge in multi-agent learning and is
the target for Stage 2.

---

## What worked / what didn't

| What worked | What didn't (yet) |
|-------------|-------------------|
| SFT → 100% parse rate immediately | Probe 3 signal-free inference (50% = chance) |
| GRPO reward climbed 22× from cold start | High variance across scenarios (std = 0.24) |
| 80% signal alignment (vs 42% random) | Model always says "above" when signals removed |
| Price convergence: $0.65 improvement | — |
| 100% participation on eval episodes | — |

---

## Artifacts

| File | Description |
|------|-------------|
| `training/runs/stage1_2026-04-25/adapter_model.safetensors` | Trained LoRA weights (115 MB, git LFS) |
| `training/runs/stage1_2026-04-25/reward_curve_stage1.png` | GRPO training curve |
| `training/runs/stage1_2026-04-25/training_log.json` | Full TRL log (60 entries) |
| `training/runs/stage1_2026-04-25/eval_trained.json` | Eval results on 10 held-out scenarios |
| `training/runs/stage1_2026-04-25/tom_probes_baselines.json` | Probe 1&2 for scripted baselines |
| `training/runs/stage1_2026-04-25/tom_probes_trained.json` | Probe 1&2 for trained agent |
| `training/runs/stage1_2026-04-25/tom_direction_inference.json` | Probe 3 per-seed results |

HF Hub: https://huggingface.co/Prathamesh0292/market-rl-stage1

---

## Next steps (Stage 2)

1. **Fix Probe 3**: Add order-flow reading to the prompt and include signal-free
   episodes in the GRPO dataset so the model must infer from public data.
2. **Longer GRPO run**: 1000+ steps with curriculum (easy → hard scenarios).
3. **Full episode reward**: Replace the per-action oracle with terminal PnL
   so the model learns to manage positions across the full episode.
4. **Opponent modeling**: Explicitly label which bot generated each order in
   the observation so the model can condition on opponent types.

---

## M7B — Stage 1 polish (in progress)

Rather than jumping straight to multi-agent self-play (Stage 2) on a model
that scores at chance on Probe 3, M7B runs **two ablations** designed to
chip away at exactly that failure mode while keeping compute under five
hours.

### Hypotheses

1. **`no_curriculum`** — Stage 1 ramps difficulty (easy → mixed → full).
   Does that schedule actually help, or would the same number of steps
   on medium-difficulty scenarios do as well? This isolates the value of
   the schedule itself.
2. **`aux_direction`** — Stage 1's reward is purely P&L-driven, so
   trading aligned with private signals is only learned indirectly. This
   ablation adds an explicit `+0.10 * min(|signal_sum|/5, 1)` bonus when
   the agent's net position matches the sign of its private signal sum.
   The hypothesis: making the signal-following pressure explicit during
   training will reduce the "say above" bias on Probe 3 *even though
   Probe 3 itself is run with signals stripped*, because the model has
   to learn to use signal information in a more transferable way.

### Plumbing

- All toggles live in `training/ablations.py` as immutable
  `AblationConfig` dataclasses; the Colab notebook just calls
  `get_preset(name)` and reads the fields it needs.
- The aux reward is implemented in `market_env/reward.py` and is
  off by default (`aux_direction_weight=0.0`) so Stage 1 numbers stay
  bit-for-bit reproducible. Backwards compat is pinned by
  `tests/test_ablations.py` (12 tests).
- The parser was hardened in the same pass (`tests/test_prompts.py`
  grew from 21 to 31 tests) to handle smart quotes, single-quoted
  Pythonic dicts, trailing commas, unquoted keys, and numeric strings.
  These were the failure modes seen in raw Qwen output during M5.
- Cross-run aggregation lives in `training/results_matrix.py` and
  `notebooks/analysis.ipynb`. After each Colab run drops three JSON
  files into `training/runs/<preset>/`, one local command rebuilds
  every plot and the comparison table.

### Run plan (~5 hr total)

1. Open `notebooks/train_ablation_colab.ipynb` in Colab.
2. Set `PRESET_NAME = 'aux_direction'` and run all cells (~2.5 hr).
3. Download the three result JSONs into `training/runs/aux_direction/`.
4. (Optional, if Colab credits remain) Repeat with
   `PRESET_NAME = 'no_curriculum'` (~2.5 hr).
5. Locally:
   ```bash
   python -m training.results_matrix --runs-root training/runs \
       --markdown training/runs/results_matrix.md
   jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb --inplace
   ```

The success criterion for `aux_direction` is **Probe 3 accuracy > 60%**
(meaningfully above chance) **without** P&L regressing below Stage 1's
+0.055. If both hold, the writeup gets the Probe 3 fix; if not, that's
honest evidence that Probe 3 needs the structural change planned for
Stage 2 (signal-free GRPO episodes).
