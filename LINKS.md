# Important Links — Multi-Agent Market RL

## Submission

| What | URL |
|------|-----|
| **HF Space** (live environment) | https://huggingface.co/spaces/Prathamesh0292/market-rl-env |
| **Code repository** (GitHub) | https://github.com/PrathameshWable/market-rl-env |
| **Colab training notebook** | https://colab.research.google.com/drive/1dVUBw60a5JrGvVYdcL3wdZVQ1QGfXnre?usp=sharing |
| **Blog / writeup** (HF model card) | https://huggingface.co/Prathamesh0292/market-rl-stage1 |

## Trained Models (HF Hub)

| Model | URL |
|-------|-----|
| Stage 1 — Qwen2.5-3B + GRPO (300 steps) | https://huggingface.co/Prathamesh0292/market-rl-stage1 |
| M7B ablation — aux_direction (200 steps) | https://huggingface.co/Prathamesh0292/market-rl-aux-direction |

## Live API (HF Space endpoints)

| Endpoint | URL |
|----------|-----|
| Health check | https://prathamesh0292-market-rl-env.hf.space/health |
| API docs (Swagger UI) | https://prathamesh0292-market-rl-env.hf.space/docs |
| Reset episode | `POST` https://prathamesh0292-market-rl-env.hf.space/reset |
| Step episode | `POST` https://prathamesh0292-market-rl-env.hf.space/step |
| List tasks | `GET` https://prathamesh0292-market-rl-env.hf.space/tasks |

## Key Results

| Metric | Value |
|--------|-------|
| GRPO training improvement | +0.02 → +0.36 mean reward (22×) |
| Mean P&L vs random | +0.055 trained vs −0.035 random |
| Parse success rate | 100% |
| Probe 2 — signal alignment | 80% (vs 42% random) |
| Probe 1 — price efficiency | $2.18 trained vs $2.88 random |
| Probe 3 — direction inference (ToM) | 50% (chance — Stage 2 target) |
