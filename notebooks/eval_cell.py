# =============================================================================
# M6 EVALUATION CELL — paste this at the bottom of your training notebook
# in Colab (after the smoke test cell). Reuses the already-loaded model.
# =============================================================================

import json
from pathlib import Path

from training.evaluate import run_evaluation, save_summary, print_summary
from training.prompts import SYSTEM_PROMPT, format_observation, parse_action

# Make sure inference mode is on
FastLanguageModel.for_inference(model)


def llm_policy(obs):
    """Trained LLM policy: format observation -> generate -> parse."""
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(obs)},
    ]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )
    return parse_action(text)


# ---- Run on the 10 held-out scenarios (seeds 100-109, medium difficulty) ----
summary = run_evaluation(
    llm_policy,
    label="trained_stage1",
    seeds=range(100, 110),
    difficulty="medium",
)

print_summary(summary)

# Save into the run folder so it gets committed alongside the other artifacts
out_path = Path("training/runs/stage1_2026-04-25/eval_trained.json")
save_summary(summary, out_path)
print(f"\n  Wrote {out_path}")
print("  Now download this JSON file and commit it to the repo.")
