"""
All-in-one training script for HuggingFace JupyterLab Spaces.
Equivalent to train_hf.ipynb but runs as a single `python` command.

Usage (from the terminal inside HF JupyterLab Space):
    cd ~/app/market-rl-env
    python training/run_train.py --smoke       # 50 steps, ~5 min, ~$0.15
    python training/run_train.py --full         # 3000 steps, ~5-6 hr, ~$9
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Stage 1 SFT + GRPO training")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="50 GRPO steps (~5 min)")
    group.add_argument("--full", action="store_true", help="3000 GRPO steps (~5-6 hr)")
    group.add_argument("--steps", type=int, help="Custom number of GRPO steps")
    args = parser.parse_args()

    MAX_GRPO_STEPS = 50 if args.smoke else (3000 if args.full else args.steps)
    SAVE_DIR = "/workspace/market-rl-stage1"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 1. Secrets ────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Checking secrets")
    print("=" * 60)

    WANDB_ENABLED = bool(os.environ.get("WANDB_API_KEY"))
    HF_PUSH_ENABLED = bool(os.environ.get("HF_TOKEN"))

    if not WANDB_ENABLED:
        print("  WANDB_API_KEY not set — wandb logging disabled")
    if not HF_PUSH_ENABLED:
        print("  HF_TOKEN not set — checkpoint push to Hub disabled")

    if WANDB_ENABLED:
        import wandb
        wandb.login()
        print(f"  wandb OK — logging to prathameshwable155-wandb/market-rl-stage1")

    # ── 2. Imports ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Importing packages")
    print("=" * 60)

    from market_env.environment import MarketEnvironment
    from market_env.bots import RandomBot, MarketMakerBot
    from training.prompts import SYSTEM_PROMPT, format_observation, parse_action, serialize_action
    from training.generate_sft_data import generate
    from training.rollout import bot_policy, run_episode
    print("  All imports OK")

    # ── 3. Generate SFT data ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Generating SFT data (500 episodes, ~45 sec)")
    print("=" * 60)

    SFT_PATH = Path("training/sft_data.jsonl")
    stats = generate(n_episodes=500, out_path=SFT_PATH, episode_length=50)
    print(f"  {stats['n_examples']} examples written")
    print(f"  action distribution: {stats['action_counts']}")
    print(f"  hold ratio: {stats['hold_ratio']:.1%}")

    # ── 4. Load model ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Loading Qwen2.5-3B (4-bit + LoRA)")
    print("=" * 60)

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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"  Trainable params: {model.get_nb_trainable_parameters()}")
    print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # ── 5. SFT warm-start ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: SFT warm-start (1 epoch, ~15 min)")
    print("=" * 60)

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
        model=model, tokenizer=tokenizer, train_dataset=ds, args=sft_config,
    )

    sft_trainer.train()
    sft_trainer.save_model(f"{SAVE_DIR}/sft-checkpoint")
    print(f"  SFT done. Adapter saved to {SAVE_DIR}/sft-checkpoint")

    # ── 6. SFT eval ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: SFT parse rate check (50 test obs)")
    print("=" * 60)

    FastLanguageModel.for_inference(model)
    env = MarketEnvironment()
    n_ok = 0
    N_TEST = 50

    for i in range(N_TEST):
        obs = env.reset(seed=10_000 + i, difficulty="medium", episode_length=5)
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation(obs)},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        pl = int(inputs["input_ids"].shape[1])
        out = model.generate(
            **inputs, max_length=pl + 80, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        _, ok = parse_action(response)
        n_ok += int(ok)

    parse_rate = n_ok / N_TEST
    print(f"  Parse rate: {parse_rate:.1%}")
    if parse_rate < 0.85:
        print("  WARNING: below 85% — GRPO may struggle")
    else:
        print("  OK — ready for GRPO")

    # ── 7. GRPO ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STEP 7: GRPO training ({MAX_GRPO_STEPS} steps)")
    print("=" * 60)

    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    GRPO_BATCH_SIZE = 4
    GRPO_GROUP_SIZE = 4
    PROMPTS_DATASET_SIZE = 1500

    def collect_prompts(n: int) -> list[dict]:
        env2 = MarketEnvironment()
        rng = random.Random(0)
        out, seed = [], 0
        while len(out) < n:
            diff = rng.choice(["easy", "medium"])
            bot = RandomBot("agent_1", seed=seed) if rng.random() < 0.5 else MarketMakerBot("agent_1", seed=seed)
            traj = run_episode(env2, bot_policy(bot), seed=seed, difficulty=diff, episode_length=50)
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
    print(f"  Built {len(prompts)} GRPO prompts")

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
    print(f"  GRPO done. Adapter saved to {SAVE_DIR}/grpo-checkpoint")

    if HF_PUSH_ENABLED:
        grpo_trainer.push_to_hub("Prathamesh0292/market-rl-stage1", private=False)
        print("  Pushed to HF Hub")

    # ── 8. Reward curve ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8: Saving reward curve")
    print("=" * 60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rewards = [log["reward"] for log in grpo_trainer.state.log_history if "reward" in log]
    steps = [log["step"] for log in grpo_trainer.state.log_history if "reward" in log]

    plt.figure(figsize=(10, 4))
    plt.plot(steps, rewards, alpha=0.4, label="step reward")
    if len(rewards) >= 50:
        smooth = np.convolve(rewards, np.ones(50)/50, mode="valid")
        plt.plot(steps[49:], smooth, color="red", label="50-step rolling mean")
    plt.xlabel("GRPO step"); plt.ylabel("mean reward")
    plt.title("Stage 1 GRPO reward curve"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/reward_curve_stage1.png", dpi=150)
    print(f"  Saved to {SAVE_DIR}/reward_curve_stage1.png")

    with open(f"{SAVE_DIR}/training_log.json", "w") as fh:
        json.dump(grpo_trainer.state.log_history, fh, indent=2)

    # ── 9. Quick eval ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 9: Smoke eval (5 episodes)")
    print("=" * 60)

    FastLanguageModel.for_inference(model)

    def llm_policy(obs):
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation(obs)},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        pl = int(inputs["input_ids"].shape[1])
        out = model.generate(
            **inputs, max_length=pl + 80, do_sample=True, temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_action(text)

    env = MarketEnvironment()
    pnls = []
    for seed in range(5):
        traj = run_episode(env, llm_policy, seed=20_000 + seed, difficulty="medium")
        pnls.append(traj.reward_breakdown.get("pnl_normalized", 0.0))
        print(f"  seed {seed}: pnl={pnls[-1]:+.4f}  parse_fail={traj.parse_failure_rate:.1%}")

    print(f"\n  Mean P&L: {statistics.mean(pnls):+.4f}")
    print(f"  (positive = beating hold baseline)")

    # ── Done ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Checkpoints: {SAVE_DIR}/")
    print(f"  Reward curve: {SAVE_DIR}/reward_curve_stage1.png")
    print(f"  Training log: {SAVE_DIR}/training_log.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
