# =============================================================================
# AUX-DIRECTION EVAL — fresh Colab account, no Drive needed.
#
# Paste cells 1-5 sequentially into a new Colab notebook.
# Runtime: ~15 min on a free T4.
# Output: 3 JSON files in the Colab file browser → download → drop into
#   training/runs/aux_direction_2026-04-26/ on your local machine.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install (run first, ignore red warnings, takes ~3 min)
# ─────────────────────────────────────────────────────────────────────────────
"""
%%capture
!pip install -q --upgrade pip
!pip install -q "unsloth"
!pip install -q --no-deps "trl>=0.13.0" "peft>=0.13.0"
!pip install -q "datasets>=3.0" "pydantic==2.12.5" "fastapi==0.136.1" "uvicorn[standard]==0.43.0" "httpx==0.28.1"
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Clone repo + load model from HF Hub
# ─────────────────────────────────────────────────────────────────────────────
"""
import os, sys, subprocess

# Clone the env + training code
REPO_DIR = "/content/Meta_ROUND2"
if not os.path.isdir(REPO_DIR):
    subprocess.check_call(["git", "clone",
        "https://github.com/PrathameshWable/market-rl-env.git", REPO_DIR])

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

# Verify imports
from market_env.environment import MarketEnvironment
from training.prompts import SYSTEM_PROMPT, format_observation, parse_action
from training.tom_probes import direction_inference, price_efficiency, signal_alignment, run_probe_episode
from training.evaluate import run_evaluation, save_summary, print_summary
print("Imports OK")
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Load the aux_direction adapter (~2 min to download)
# ─────────────────────────────────────────────────────────────────────────────
"""
import torch
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Prathamesh0292/market-rl-aux-direction",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("Model loaded:", model.__class__.__name__)
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Run eval + 3 ToM probes (~10 min)
# ─────────────────────────────────────────────────────────────────────────────
"""
import json
from dataclasses import asdict
from pathlib import Path
from market_env.environment import MarketEnvironment

RUN_DIR = Path("training/runs/aux_direction_2026-04-26")
RUN_DIR.mkdir(parents=True, exist_ok=True)


def _generate(prompt_text, *, max_new_tokens=80, temperature=0.7, do_sample=True):
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    pl = int(inputs["input_ids"].shape[1])
    out = model.generate(
        **inputs,
        max_length=pl + max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def llm_policy(obs):
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": format_observation(obs)},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return parse_action(_generate(prompt))


# ── Part 1: Standard eval ────────────────────────────────────────────────────
print("=" * 55)
print("PART 1 — eval on seeds 100-109 (medium difficulty)")
print("=" * 55)
summary = run_evaluation(llm_policy, label="trained_stage1_aux",
                         seeds=range(100, 110), difficulty="medium")
print_summary(summary)
save_summary(summary, RUN_DIR / "eval_trained.json")

# ── Part 2: ToM probes 1 & 2 ────────────────────────────────────────────────
print("=" * 55)
print("PART 2 — ToM Probe 1 (price efficiency) + Probe 2 (signal alignment)")
print("=" * 55)
env = MarketEnvironment()
episodes = []
for seed in range(100, 110):
    print(f"  episode seed={seed}...")
    episodes.append(run_probe_episode(env, llm_policy, seed=seed, difficulty="medium"))

pe = price_efficiency(episodes, "trained_aux")
sa = signal_alignment(episodes, "trained_aux")
print(f"\n  Probe 1: final |mid-tv| = ${pe.final_mean_error:.2f}  (Stage 1 was $2.18)")
print(f"  Probe 2: alignment = {sa.alignment_rate:.1%}  (Stage 1 was 80%)")

with (RUN_DIR / "tom_probes_trained.json").open("w") as fh:
    json.dump({"trained": {"price_efficiency": asdict(pe),
                           "signal_alignment": asdict(sa)}}, fh, indent=2)

# ── Part 3: Direction inference (the actual ToM probe) ───────────────────────
print("=" * 55)
print("PART 3 — Probe 3 (direction inference, no signals)")
print("Chance = 50%. Stage 1 was 50%. Does aux reward improve this?")
print("=" * 55)

def model_answer_fn(probe_prompt):
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": probe_prompt},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return _generate(prompt, max_new_tokens=8, temperature=0.0, do_sample=False)

di = direction_inference(model_answer_fn, label="trained_aux",
                         n_probes=20, seed_start=200, turns_to_warm_up=25)

print(f"\n  Probe 3 accuracy: {di.accuracy:.1%}  ({di.n_correct}/20)")
print(f"  (Chance baseline = 50%, Stage 1 = 50%)")
for p in di.probes:
    mark = "V" if p.correct else "X"
    print(f"    {mark}  seed={p.seed}  tv=${p.true_value:.2f}  truth={p.ground_truth}  model={p.model_answer}")

with (RUN_DIR / "tom_direction_inference.json").open("w") as fh:
    json.dump(asdict(di), fh, indent=2)

print("=" * 55)
print("DONE — download these 3 files from the Colab file panel (left sidebar):")
for f in ["eval_trained.json", "tom_probes_trained.json", "tom_direction_inference.json"]:
    print(f"  training/runs/aux_direction_2026-04-26/{f}")
print("=" * 55)
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — (Optional) quick sanity check before downloading
# ─────────────────────────────────────────────────────────────────────────────
"""
import json
from pathlib import Path
RUN_DIR = Path("training/runs/aux_direction_2026-04-26")

e = json.load(open(RUN_DIR / "eval_trained.json"))
p = json.load(open(RUN_DIR / "tom_probes_trained.json"))
d = json.load(open(RUN_DIR / "tom_direction_inference.json"))

print(f"Mean P&L     : {e['mean_pnl_normalized']:+.4f}  (Stage 1: +0.0546)")
print(f"Participation: {e['participation_rate']:.0%}")
print(f"Parse rate   : {e['parse_success_rate']:.0%}")
print(f"Probe 2 align: {p['trained']['signal_alignment']['alignment_rate']:.0%}  (Stage 1: 80%)")
print(f"Probe 3 ToM  : {d['accuracy']:.0%}  ({d['n_correct']}/20)  (Stage 1: 50%)")
"""
