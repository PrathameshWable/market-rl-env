"""
Generate notebooks/train_hf.ipynb — HuggingFace JupyterLab variant.

Differences from train_colab.ipynb:
- No google.colab imports (Drive, userdata)
- Secrets read from os.environ (HF Spaces injects them via Space settings)
- Checkpoints saved to /workspace/ (persistent within session) + pushed to HF Hub
- Repo cloned into /workspace/ instead of /content/

Run:  python -m notebooks._build_notebook_hf
"""
from __future__ import annotations

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


# =============================================================================
# Title
# =============================================================================

md("""
# Stage 1 Training — Multi-Agent Market RL  
### HuggingFace JupyterLab Edition

**This notebook runs entirely on HuggingFace compute — no Google Colab needed.**

Two phases:
1. **SFT warm-start** (~15 min) — fine-tune Qwen2.5-3B on 25k InformedBot demos
2. **GRPO** (~4–6 hr on A10G large) — reinforce profitable actions

**Setup before running:**
- Create this as a **JupyterLab Space** on HuggingFace with GPU hardware
- Add `WANDB_API_KEY` and `HF_TOKEN` as Space secrets (Settings → Variables and secrets)
- Then run cells top to bottom

**Recommended hardware:** A10G large (46 GB · $1.50/hr)  
**Smoke test:** set `MAX_GRPO_STEPS = 50` in cell 9 first (~5 min, ~$0.15)  
**Full run:** set `MAX_GRPO_STEPS = 3000` (~5–6 hr, ~$9)
""")

# =============================================================================
# Cell 1 — install
# =============================================================================

md("## 1. Install dependencies")

code("""\
%%capture
!pip install -q --upgrade pip
!pip install -q "unsloth"
!pip install -q --no-deps "trl>=0.13.0" "peft>=0.13.0"
!pip install -q "datasets>=3.0" "wandb>=0.18" "matplotlib>=3.9"
!pip install -q "pydantic==2.12.5" "fastapi==0.136.1" "uvicorn[standard]==0.43.0" "httpx==0.28.1"
""")

# =============================================================================
# Cell 2 — clone repo
# =============================================================================

md("""## 2. Clone the env repo

Pulls `market_env/`, `client/`, and `training/` into `/workspace/`.
""")

code("""\
import os, sys, subprocess

REPO_URL = "https://github.com/PrathameshWable/market-rl-env.git"
REPO_DIR = "/workspace/market-rl-env"

if not os.path.isdir(REPO_DIR):
    subprocess.check_call(["git", "clone", REPO_URL, REPO_DIR])

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

from market_env.environment import MarketEnvironment
from training.prompts import SYSTEM_PROMPT, format_observation, parse_action, serialize_action
print("env + training package import OK")
""")

# =============================================================================
# Cell 3 — secrets
# =============================================================================

md("""## 3. Secrets & wandb

Secrets are set in your Space → **Settings → Variables and secrets**.  
Add:
- `WANDB_API_KEY` — from https://wandb.ai/authorize  
- `HF_TOKEN` — from https://huggingface.co/settings/tokens (write scope)

HF Spaces injects them as environment variables automatically.
""")

code("""\
import os

SAVE_DIR = "/workspace/market-rl-stage1"
os.makedirs(SAVE_DIR, exist_ok=True)

WANDB_ENABLED = bool(os.environ.get("WANDB_API_KEY"))
HF_PUSH_ENABLED = bool(os.environ.get("HF_TOKEN"))

if not WANDB_ENABLED:
    print("WANDB_API_KEY not found in env — wandb logging disabled")
if not HF_PUSH_ENABLED:
    print("HF_TOKEN not found in env — checkpoint push to Hub disabled")

if WANDB_ENABLED:
    import wandb
    wandb.login()
    WANDB_ENTITY = "prathameshwable155-wandb"
    WANDB_PROJECT = "market-rl-stage1"
    print(f"wandb OK — logging to {WANDB_ENTITY}/{WANDB_PROJECT}")
""")

# =============================================================================
# Cell 4 — generate SFT data
# =============================================================================

md("""## 4. Generate SFT warm-start data

Plays InformedBot as teacher for 500 episodes (~45 sec on CPU).
""")

code("""\
from pathlib import Path
from training.generate_sft_data import generate

SFT_PATH = Path(REPO_DIR) / "training/sft_data.jsonl"
stats = generate(n_episodes=500, out_path=SFT_PATH, episode_length=50)
print(stats)
""")

# =============================================================================
# Cell 5 — load model
# =============================================================================

md("""## 5. Load Qwen2.5-3B in 4-bit with LoRA

~3 min on first run (downloads ~2 GB model weights).
""")

code("""\
import torch
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_SEQ_LEN = 4096

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print(f"Model loaded. Trainable params: {model.get_nb_trainable_parameters()}")
print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
""")

# =============================================================================
# Cell 6 — SFT training
# =============================================================================

md("""## 6. SFT warm-start

1 epoch on 25k examples. Target: ≥90% parse rate after this.
""")

code("""\
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

ds = load_dataset("json", data_files=str(SFT_PATH), split="train")

def render_chat(example):
    return {"text": tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False,
    )}

ds = ds.map(render_chat, remove_columns=["messages"])

sft_config = SFTConfig(
    output_dir=f"{SAVE_DIR}/sft_out",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=20,
    learning_rate=2e-4,
    logging_steps=20,
    save_strategy="no",
    bf16=True,
    max_seq_length=MAX_SEQ_LEN,
    report_to=("wandb" if WANDB_ENABLED else "none"),
    run_name="sft-warmstart",
)

sft_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=sft_config,
)

sft_trainer.train()
sft_trainer.save_model(f"{SAVE_DIR}/sft-checkpoint")
print(f"SFT done. Adapter saved to {SAVE_DIR}/sft-checkpoint")

if HF_PUSH_ENABLED:
    sft_trainer.push_to_hub("PrathameshWable/market-rl-stage1-sft", private=False)
    print("SFT adapter pushed to HF Hub.")
""")

# =============================================================================
# Cell 7 — SFT eval
# =============================================================================

md("""## 7. SFT eval — parse rate

Must be ≥ 85% before running GRPO. If below, re-run cell 6 with more epochs.
""")

code("""\
from market_env.environment import MarketEnvironment

FastLanguageModel.for_inference(model)

env = MarketEnvironment()
N_TEST = 50
n_ok = 0

for i in range(N_TEST):
    obs = env.reset(seed=10_000 + i, difficulty="medium", episode_length=5)
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(obs)},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs, max_new_tokens=80, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    _, ok = parse_action(response)
    n_ok += int(ok)

parse_rate = n_ok / N_TEST
print(f"SFT parse rate: {parse_rate:.1%}")
if parse_rate < 0.85:
    print("WARNING: below 85% — re-run cell 6 with num_train_epochs=2 before proceeding")
else:
    print("OK — ready for GRPO")
""")

# =============================================================================
# Cell 8 — GRPO explanation
# =============================================================================

md("""## 8. GRPO — reward design

**Per-action oracle reward** (true_value is known at training time, never at eval):

| Action | Reward |
|--------|--------|
| Parse fail | -1.00 |
| Hold | -0.05 (pressure to act) |
| Cancel | +0.02 (valid format) |
| Buy at price P | `clip((true_value − P) / 5, −1, +1)` |
| Sell at price P | `clip((P − true_value) / 5, −1, +1)` |

This avoids running full 50-turn episodes inside GRPO, which would be
very slow. Full-episode evaluation is M6 (`training/evaluate.py`).
""")

# =============================================================================
# Cell 9 — GRPO training
# =============================================================================

code("""\
import json, random
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from market_env.bots import RandomBot, MarketMakerBot
from market_env.environment import MarketEnvironment
from training.rollout import bot_policy, run_episode

# ── KNOBS ────────────────────────────────────────────────────────────────────
MAX_GRPO_STEPS      = 50        # ← set to 50 for smoke test, 3000 for full run
GRPO_BATCH_SIZE     = 4
GRPO_GROUP_SIZE     = 4
PROMPTS_DATASET_SIZE = 1500
# ─────────────────────────────────────────────────────────────────────────────

# Build diverse (observation, true_value) prompt dataset
def collect_prompts(n: int) -> list[dict]:
    env = MarketEnvironment()
    rng = random.Random(0)
    out, seed = [], 0
    while len(out) < n:
        diff = rng.choice(["easy", "medium"])
        bot = RandomBot("agent_1", seed=seed) if rng.random() < 0.5 else MarketMakerBot("agent_1", seed=seed)
        traj = run_episode(env, bot_policy(bot), seed=seed, difficulty=diff, episode_length=50)
        for i in range(0, len(traj.turns), 5):
            out.append({
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": traj.turns[i].prompt}],
                    tokenize=False, add_generation_prompt=True,
                ),
                "true_value": traj.true_value,
            })
            if len(out) >= n:
                break
        seed += 1
    return out

prompts = collect_prompts(PROMPTS_DATASET_SIZE)
prompt_ds = Dataset.from_list(prompts)
print(f"Built {len(prompts)} GRPO prompts.")

# Reward function
HOLD_PENALTY, CANCEL_BONUS, PARSE_FAIL_PENALTY, PNL_SCALE = -0.05, 0.02, -1.0, 5.0

def oracle_reward(prompts, completions, true_value, **_):
    rewards = []
    for completion, tv in zip(completions, true_value):
        action, ok = parse_action(completion if isinstance(completion, str) else completion[0])
        if not ok:                          rewards.append(PARSE_FAIL_PENALTY); continue
        if action.action_type == "hold":    rewards.append(HOLD_PENALTY);       continue
        if action.action_type == "cancel":  rewards.append(CANCEL_BONUS);       continue
        if action.price is None:            rewards.append(PARSE_FAIL_PENALTY); continue
        r = (tv - action.price if action.action_type == "buy" else action.price - tv) / PNL_SCALE
        rewards.append(max(-1.0, min(1.0, r)))
    return rewards

# Train
FastLanguageModel.for_training(model)

grpo_config = GRPOConfig(
    output_dir=f"{SAVE_DIR}/grpo_out",
    max_steps=MAX_GRPO_STEPS,
    per_device_train_batch_size=GRPO_BATCH_SIZE,
    num_generations=GRPO_GROUP_SIZE,
    learning_rate=5e-6,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    bf16=True,
    max_prompt_length=1024,
    max_completion_length=80,
    temperature=0.9,
    report_to=("wandb" if WANDB_ENABLED else "none"),
    run_name="grpo-stage1",
)

grpo_trainer = GRPOTrainer(
    model=model,
    reward_funcs=oracle_reward,
    args=grpo_config,
    train_dataset=prompt_ds,
)

grpo_trainer.train()
grpo_trainer.save_model(f"{SAVE_DIR}/grpo-checkpoint")
print(f"GRPO done. Adapter saved to {SAVE_DIR}/grpo-checkpoint")

if HF_PUSH_ENABLED:
    grpo_trainer.push_to_hub("PrathameshWable/market-rl-stage1", private=False)
    print("GRPO adapter pushed to HF Hub.")
""")

# =============================================================================
# Cell 10 — reward curve
# =============================================================================

md("## 9. Reward curve")

code("""\
import matplotlib.pyplot as plt
import numpy as np, json
from pathlib import Path

rewards = [log["reward"] for log in grpo_trainer.state.log_history if "reward" in log]
steps   = [log["step"]   for log in grpo_trainer.state.log_history if "reward" in log]

plt.figure(figsize=(10, 4))
plt.plot(steps, rewards, alpha=0.4, label="step reward")
if len(rewards) >= 50:
    smooth = np.convolve(rewards, np.ones(50)/50, mode="valid")
    plt.plot(steps[49:], smooth, color="red", label="50-step rolling mean")
plt.xlabel("GRPO step"); plt.ylabel("mean reward")
plt.title("Stage 1 GRPO reward curve"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()

out_dir = Path(SAVE_DIR)
plt.savefig(out_dir / "reward_curve_stage1.png", dpi=150)
plt.show()

with open(out_dir / "training_log.json", "w") as fh:
    json.dump(grpo_trainer.state.log_history, fh, indent=2)
print(f"Saved plot + log to {SAVE_DIR}")
""")

# =============================================================================
# Cell 11 — smoke test
# =============================================================================

md("""## 10. Quick eval — does it trade?

5 episodes, reports normalized P&L per episode. Positive = making money.
""")

code("""\
import statistics
from training.rollout import run_episode

FastLanguageModel.for_inference(model)

def llm_policy(obs):
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": format_observation(obs)},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.7,
                         pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_action(text)

env = MarketEnvironment()
pnls = []
for seed in range(5):
    traj = run_episode(env, llm_policy, seed=20_000 + seed, difficulty="medium")
    pnls.append(traj.reward_breakdown.get("pnl_normalized", 0.0))
    print(f"  seed {seed}: pnl={pnls[-1]:+.4f}  parse_fail={traj.parse_failure_rate:.1%}")

print(f"\\nMean P&L: {statistics.mean(pnls):+.4f}  (positive = beating hold baseline)")
""")

# =============================================================================
# Build
# =============================================================================

nb.cells = cells
out = Path(__file__).parent / "train_hf.ipynb"
with out.open("w", encoding="utf-8") as fh:
    nbf.write(nb, fh)
print(f"Wrote {out} ({len(cells)} cells)")
