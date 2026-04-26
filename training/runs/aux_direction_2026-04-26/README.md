# M7B Ablation — aux_direction — 2026-04-26

## What this run tests

`AblationConfig(curriculum=True, aux_direction_weight=0.10)`

In addition to the oracle P&L reward, buy/sell actions get a bonus when
the trade direction matches the agent's private signal sum:

```
bonus = min(|signal_sum| / 5, 1.0) * 0.10   (only if trade matches signal sign)
```

This intervention directly targets **Probe 3 (direction inference)**:
Stage 1 Probe 3 sat at 50% (chance). If the model gets extra reward for
signal-aligned trades, it should be more consistently directional, making
its order flow more informative to Probe 3's signal-stripping test.

## Status

- [x] SFT warm-start (150 steps) — same as Stage 1
- [x] GRPO with aux_direction_weight=0.10 — **ran, cell 8 completed**
- [ ] Eval (eval_trained.json) — needs GPU to run eval_cell.py
- [ ] ToM probes (tom_probes_trained.json, tom_direction_inference.json) — same

## Expected files (to download from Drive and place here)

| File | Source on Drive | Status |
|------|-----------------|--------|
| `training_log.json` | `/MyDrive/market-rl-stage1/aux_direction/training_log.json` | TODO |
| `reward_curve_stage1.png` | `/MyDrive/market-rl-stage1/aux_direction/reward_curve_stage1.png` | TODO |
| `eval_trained.json` | run `notebooks/eval_cell.py` on fresh Colab with this checkpoint | TODO |
| `tom_probes_trained.json` | same | TODO |
| `tom_direction_inference.json` | same | TODO |

## How to evaluate on a fresh Colab account

The checkpoint needs to be pushed to HF Hub first (or shared Drive):

```python
# In Colab — push the aux_direction adapter to HF Hub
from peft import PeftModel
model.push_to_hub("Prathamesh0292/market-rl-aux-direction", private=False)

# Then on a fresh Colab account:
model, tokenizer = FastLanguageModel.from_pretrained(
    "Prathamesh0292/market-rl-aux-direction", load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
%run -i /content/Meta_ROUND2/notebooks/eval_cell.py
```

Place the three downloaded JSONs into this folder and re-run
`python -m training.results_matrix` to update the comparison table.
