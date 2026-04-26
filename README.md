---
title: Multi-Agent Market RL
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: OpenEnv trading environment for theory-of-mind LLM training.
tags:
  - openenv
  - multi-agent
  - reinforcement-learning
  - theory-of-mind
  - finance
  - grpo
---

# Multi-Agent Market RL Environment

> **Thesis:** an LLM agent that is rewarded only for profit, but must infer
> other agents' hidden information to profit, will develop implicit
> theory-of-mind.

OpenEnv-compliant multi-agent trading environment. A trainable LLM agent
trades a single asset on a continuous double-auction order book against
scripted bots, each holding a private slice of the asset's true value. The
true value is revealed only at episode end and converted to a sparse,
clamped scalar reward in [-1, 1] suitable for GRPO.

Built for the **April 2026 OpenEnv Hackathon (Round 2)**.

---

## Submission links

| What | Where |
|------|-------|
| Hugging Face Space (env server) | https://huggingface.co/spaces/Prathamesh0292/market-rl-env |
| Trained Stage 1 adapter (HF Hub) | https://huggingface.co/Prathamesh0292/market-rl-stage1 |
| Training blog + eval report | [blog.md](blog.md) |
| Colab training notebook | [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) |

---

## What this environment exposes

Standard OpenEnv HTTP surface. All payloads are Pydantic models defined in
[`market_env/models.py`](market_env/models.py).

| Method | Path     | Purpose                                          |
|--------|----------|--------------------------------------------------|
| GET    | /health  | Liveness probe — returns `{"status": "ok"}`      |
| GET    | /tasks   | List 50 eval scenarios + 3 difficulty demos     |
| POST   | /reset   | Start a new episode, return initial observation |
| POST   | /step    | Apply an action, return obs/reward/done/info    |
| GET    | /state   | Inspect any open episode                        |

A single server instance manages many concurrent episodes keyed by
`episode_id`, so one Space serves many parallel rollouts.

---

## Quickstart — run locally

```bash
pip install -e .
uvicorn market_env.server:app --host 0.0.0.0 --port 7860
```

Then in another shell:

```bash
curl http://localhost:7860/health
# → {"status":"ok"}
```

Or drive a full episode from Python:

```python
from client import MarketClient
from market_env.models import MarketAction

with MarketClient("http://localhost:7860") as c:
    obs = c.reset(seed=42, difficulty="easy", episode_length=10)
    while True:
        obs, reward, done, info = c.step(obs.episode_id, MarketAction(action_type="hold"))
        if done:
            print("final reward:", reward)
            print("true value:", obs.true_value)
            break
```

---

## Quickstart — run with Docker

```bash
docker build . -t market-rl
docker run -p 7860:7860 market-rl
curl http://localhost:7860/health
```

This is the same image HF Spaces builds.

---

## Tests

```bash
pip install -e .[dev]
pytest tests/ -v
```

232 tests across the order book, scenario generator, scripted bots, reward
function, environment, server, integration, evaluation harness, and ToM probes.
Run in ~2 s.

---

## Project structure

```
market_env/
  order_book.py     # continuous double-auction limit order book
  scenario.py       # asset value + per-agent private signals
  bots.py           # 5 scripted bots (Random, Momentum, MeanReversion,
                    #                  MarketMaker, InformedBot)
  reward.py         # sparse end-of-episode reward in [-1, 1]
  environment.py    # multi-session MarketEnvironment (reset/step/state)
  server.py         # FastAPI HTTP layer
  models.py         # all Pydantic request/response shapes
client/
  client.py         # synchronous MarketClient (httpx)
tests/              # 187 tests, ~99% coverage
examples/
  run_episode.py    # 50-turn 5-bot smoke run
Dockerfile          # python:3.11-slim, exposes 7860
openenv.yaml        # OpenEnv manifest
```

---

## Stage 1 training results

![GRPO reward curve](training/runs/stage1_2026-04-25/reward_curve_stage1.png)

| Metric | Value |
|--------|-------|
| Mean reward (last 10 steps) | +0.362 (22× improvement from cold start) |
| Mean P&L on 10 held-out scenarios | **+0.055** vs random −0.035, hold 0.000 |
| Participation rate | 100% |
| Parse success rate | 100% |
| Signal alignment (Probe 2) | **80%** vs 42% random, 100% oracle |
| Price efficiency improvement (Probe 1) | $0.65 closer to true value by episode end |
| Direction inference / ToM (Probe 3) | 50% (chance) — Stage 2 target |

Full details: [blog.md](blog.md)

---

## Status

- [x] M1 — Order book engine
- [x] M2 — Scenario generator + scripted bots
- [x] M3 — Environment + reward + FastAPI server + HTTP client
- [x] M4 — Docker + HF Space deployment
- [x] M5 — Colab notebook + Stage 1 GRPO training (300 steps, T4, ~3 hr)
- [x] M6 — Evaluation harness + theory-of-mind probes (3 probes, scripted baselines)
- [ ] M7 — Stage 2: signal-free ToM training (Probe 3 target > 65%)
- [ ] M8 — Demo + final submission polish

---

## License

Apache-2.0.
