# Stage 1 GRPO run — 2026-04-25/26

## Headline

**+0.27 mean P&L on 5-seed smoke test, 100% parse rate, GRPO ran 300 steps in ~2 hr on a Colab T4.**

## Run config

| Setting | Value |
|---------|-------|
| Base model | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (4-bit + LoRA r=16) |
| GRPO steps | 300 |
| Per-device batch | 4 prompts × 4 generations |
| Prompt dataset size | 1500 (mixed RandomBot + MarketMakerBot rollouts) |
| Hardware | Google Colab, single T4 (16 GB VRAM) |
| Wall-clock | ~2 hours (SFT warm-start + GRPO combined) |
| Reward fn | Per-action oracle: `clip((true_value - price) / 5, -1, +1)` for buys, etc. |

## Reward curve

![Reward curve](reward_curve_stage1.png)

Per-step mean reward across 60 logging windows (one log every 5 steps).

| Window | Mean reward |
|--------|-------------|
| First 10 steps  | +0.017 (baseline noise) |
| Mid (steps 100–200) | +0.18 |
| Last 10 steps   | +0.362 |
| Peak (step 285) | +0.787 |

## Files

| File | Size | Purpose |
|------|------|---------|
| `reward_curve_stage1.png` | 84 KB | Embedded in blog and root README |
| `training_log.json` | 55 KB | Full TRL `state.log_history` for offline analysis |
| `adapter_config.json` | 1.2 KB | PEFT/LoRA config (so you can reconstruct the adapter shape) |
| `adapter_model.safetensors` | 115 MB | The trained LoRA weights — tracked via git LFS |
| `checkpoint_README.md` | 5.2 KB | TRL's auto-generated model card |

## How to reload the adapter

```python
from unsloth import FastLanguageModel
from peft import PeftModel

base, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit", load_in_4bit=True,
)
model = PeftModel.from_pretrained(base, "training/runs/stage1_2026-04-25")
FastLanguageModel.for_inference(model)
```

## Reproducing this run

The Colab notebook (`notebooks/train_colab.ipynb`) is the source of truth.
With `MAX_GRPO_STEPS = 300` it will reproduce within stochastic noise on
any T4 or A10G runtime.
