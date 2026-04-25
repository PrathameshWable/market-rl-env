# `training/` — SFT warm-start + GRPO training

> **Stage 1 result (2026-04-26):** **+0.27 mean P&L on 5-seed smoke test, 100% parse rate, GRPO ran 300 steps in ~2 hr on a Colab T4.**
>
> Mean per-step reward climbed from **+0.02** (first 10 steps) to **+0.36** (last 10 steps) — a 22× improvement over the SFT-only baseline. Peak reward of +0.79 at step 285.
>
> Full artifacts: [`runs/stage1_2026-04-25/`](runs/stage1_2026-04-25/) (reward curve, full TRL log, LoRA adapter via git LFS).
>
> ![Stage 1 reward curve](runs/stage1_2026-04-25/reward_curve_stage1.png)

Everything in this folder is **training-time only**. The runtime
environment server (`market_env/`) does not import any of it, so the
HF Space image stays slim.

## Layout

| File | Purpose |
|------|---------|
| `prompts.py` | System prompt, observation formatter, action serializer, robust parser (always falls back to `hold` on failure) |
| `rollout.py` | `run_episode(env, policy, ...)` — one loop, used by both SFT data gen and GRPO eval |
| `curriculum.py` | Difficulty scheduler: easy 0–600, mixed 600–1500, full 1500+ |
| `generate_sft_data.py` | Plays `InformedBot` for 500 episodes, writes `sft_data.jsonl` (~46 MB, gitignored) |
| `sft_data.jsonl` | Generated SFT corpus — never committed; regenerated in the Colab notebook in ~45 s |

## How it fits with M5

The Colab notebook (`notebooks/train_colab.ipynb`) is the only thing
judges run. It clones this repo, calls `training/generate_sft_data.py`
to build the SFT corpus, then SFTs and GRPOs the model — saving LoRA
adapters to Google Drive at each phase.

Splitting the logic out of the notebook into this package means:

- We can unit-test the parser, formatter, and rollout loop locally
  without a GPU (see `tests/test_prompts.py` and `tests/test_rollout.py`).
- The notebook stays short and readable.
- Any change to prompt format or action grammar happens in **one
  place** — guaranteeing SFT and GRPO see the same prompts. (Tiny
  drifts here are the most common cause of GRPO collapse.)

## M6 — Evaluation + Theory-of-Mind Probes

Two scripts live here:

- **`evaluate.py`** — runs a policy on the held-out scenario set
  (seeds 100–109, medium difficulty, default 4-bot opponents) and reports
  mean P&L, participation rate, and parse rate. CLI for scripted policies
  (`--policy random|hold`); programmatic API for the LLM (called from
  the Colab cell, since the model lives on the GPU).

- **`tom_probes.py`** — three measurements designed to test the
  *theory-of-mind* claim from the project thesis. Each probe targets a
  different aspect of "does the agent infer from others?":

  | Probe | What it measures | Chance / floor | Pass threshold |
  |-------|------------------|----------------|----------------|
  | 1. Price efficiency  | Mean \|mid_price − true_value\| over turns. Lower = the agent's information is moving prices toward the truth. | hold ≈ $3.06 | trained < random |
  | 2. Signal alignment  | When private signals say "above $50", does the agent buy? When they say "below", does it sell? | random ≈ 50% | trained > 60% |
  | 3. Direction inference | Strip the trainable agent's signals from the prompt and ask "is true_value above or below $50?" — pure inference from order flow. | 50% (chance) | trained > 60% |

  Probes 1 and 2 run on any policy, including locally with `python -m
  training.tom_probes`. Probe 3 needs the LLM and is invoked from the
  Colab eval cell (`notebooks/eval_cell.py`).

**Baseline numbers (from running `python -m training.tom_probes`):**

| Policy | Probe 1 — final \|mid−tv\| | Probe 2 — alignment |
|--------|----------------------------|---------------------|
| hold (no trades)        | $3.06 | 0% (no actions)    |
| RandomBot               | $2.88 | 42%                |
| Informed oracle (cheats)| $2.40 | 100%               |

The trained model results live in
[`runs/stage1_2026-04-25/`](runs/stage1_2026-04-25/) and are populated by
the Colab eval cell.

## Reward design (per-action oracle, used in GRPO cell)

Full episode rollouts inside GRPO are expensive (50 turns × per-token
generation per completion × K completions per prompt). For Stage 1 we
use a per-action surrogate that exploits the fact that we know
`true_value` at training time:

```
parse fail        → -1.00
hold              → -0.05      (small "do something" pressure)
cancel            → +0.02      (valid format)
buy  at price P   → clip((true_value - P) / 5,  -1, +1)
sell at price P   → clip((P - true_value) / 5,  -1, +1)
```

The full-episode evaluation in M6 (`training/evaluate.py`) is the
real test of whether the policy actually wins games. The surrogate
just needs to give a learnable signal per step.

## Local commands

```bash
# Generate the 500-episode SFT dataset (~45 sec on CPU)
python -m training.generate_sft_data --episodes 500 --out training/sft_data.jsonl

# Quick smoke (5 episodes, 5 turns each → 25 examples)
python -m training.generate_sft_data --episodes 5 --episode-length 5 --out training/sft_smoke.jsonl

# Run the training-layer unit tests
python -m pytest tests/test_prompts.py tests/test_rollout.py -v
```

## Running the Colab notebook

1. Open `notebooks/train_colab.ipynb` in Colab and pick an A10G or A100
   runtime.
2. Add two **Colab secrets** (left sidebar → key icon):
   - `WANDB_API_KEY` — get one at https://wandb.ai/authorize
     (regenerate if you ever paste it into a chat or commit)
   - `HF_TOKEN` — write-scope token from
     https://huggingface.co/settings/tokens (only needed if you want to
     push the trained adapter to the Hub)
3. Run cells top to bottom. The `MAX_GRPO_STEPS` knob in the GRPO cell
   controls runtime — set it to `50` for a ~2 min smoke test, `3000`
   for the full Stage 1 run.

Outputs land in `/content/drive/MyDrive/market-rl-stage1/`:
- `sft-checkpoint/` — LoRA adapter after warm-start
- `grpo-checkpoint/` — LoRA adapter after GRPO
- `reward_curve_stage1.png` — embedded in the README + blog post
- `training_log.json` — full TRL log history for offline analysis

## Regenerating the notebook

The notebook is generated from `notebooks/_build_notebook.py` so the
source stays diffable. After editing the builder:

```bash
python -m notebooks._build_notebook
```

Always commit both `_build_notebook.py` and the regenerated
`train_colab.ipynb` together — judges read the notebook directly.

## Wandb

The notebook auto-logs to:
- entity: `prathameshwable155-wandb`
- project: `market-rl-stage1`
- runs:    `sft-warmstart`, `grpo-stage1`

If `WANDB_API_KEY` isn't set as a Colab secret, the notebook falls back
to `report_to="none"` — training still works, just no online logging.
