# =============================================================================
# M6 EVALUATION + ToM PROBES — paste this at the bottom of your training
# notebook (after the smoke-test cell). Reuses the already-loaded model.
#
# Runtime: ~5–10 min on T4. Produces three artifacts:
#   training/runs/stage1_2026-04-25/eval_trained.json
#   training/runs/stage1_2026-04-25/tom_probes_trained.json
#   training/runs/stage1_2026-04-25/tom_direction_inference.json
# =============================================================================

import json
from dataclasses import asdict
from pathlib import Path

from training.evaluate import run_evaluation, save_summary, print_summary
from training.prompts import SYSTEM_PROMPT, format_observation, parse_action
from training.tom_probes import (
    direction_inference,
    price_efficiency,
    signal_alignment,
    run_probe_episode,
)
from market_env.environment import MarketEnvironment


FastLanguageModel.for_inference(model)
RUN_DIR = Path("training/runs/stage1_2026-04-25")
RUN_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared LLM-call helper
# ---------------------------------------------------------------------------

def _generate(prompt_text: str, *, max_new_tokens: int = 80, temperature: float = 0.7,
              do_sample: bool = True) -> str:
    """Run one model.generate call and return decoded completion text."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )


def llm_policy(obs):
    """Trained LLM as a Policy: format observation -> generate -> parse."""
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(obs)},
    ]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True,
    )
    text = _generate(prompt)
    return parse_action(text)


# ===========================================================================
# Part 1 — Standard evaluation on 10 held-out scenarios (seeds 100–109)
# ===========================================================================

print("\n" + "=" * 60)
print("PART 1 — Standard eval (10 held-out scenarios)")
print("=" * 60)

eval_summary = run_evaluation(
    llm_policy,
    label="trained_stage1",
    seeds=range(100, 110),
    difficulty="medium",
)
print_summary(eval_summary)
save_summary(eval_summary, RUN_DIR / "eval_trained.json")
print(f"\n  Wrote {RUN_DIR / 'eval_trained.json'}")


# ===========================================================================
# Part 2 — ToM Probes 1 & 2 with the trained policy
# ===========================================================================

print("\n" + "=" * 60)
print("PART 2 — ToM Probes 1 (price efficiency) + 2 (signal alignment)")
print("=" * 60)

env = MarketEnvironment()
trained_episodes = []
for seed in range(100, 110):
    print(f"  collecting episode seed={seed}...")
    trained_episodes.append(
        run_probe_episode(env, llm_policy, seed=seed, difficulty="medium")
    )

pe = price_efficiency(trained_episodes, "trained")
sa = signal_alignment(trained_episodes, "trained")

print(f"\n  Probe 1 (price efficiency)")
print(f"    initial mean |mid - tv| = ${pe.initial_mean_error:.2f}")
print(f"    final   mean |mid - tv| = ${pe.final_mean_error:.2f}")
print(f"    improvement              = ${pe.improvement:.2f}  (positive = converged toward true value)")

print(f"\n  Probe 2 (signal alignment)")
print(f"    aligned     : {sa.n_aligned}")
print(f"    misaligned  : {sa.n_misaligned}")
print(f"    rate        : {sa.alignment_rate:.1%}   (chance = 50%)")

with (RUN_DIR / "tom_probes_trained.json").open("w") as fh:
    json.dump({
        "trained": {
            "price_efficiency": asdict(pe),
            "signal_alignment": asdict(sa),
        }
    }, fh, indent=2)
print(f"\n  Wrote {RUN_DIR / 'tom_probes_trained.json'}")


# ===========================================================================
# Part 3 — ToM Probe 3: direction inference (the actual ToM probe)
# ===========================================================================

print("\n" + "=" * 60)
print("PART 3 — ToM Probe 3: direction inference (signal-free)")
print("=" * 60)
print("  This is the real ToM test: can the model predict true_value > $50")
print("  using ONLY public market data (no private signals)?")
print("  Chance baseline = 50%. Anything > 60% with N=20 is meaningful.\n")


def model_answer_fn(probe_prompt: str) -> str:
    """Wrap the model.generate call for a yes/no probe.

    The probe prompt already contains the full observation + DIRECTION_PROBE_INSTRUCTION.
    We treat it as a system+user chat message pair so the chat template applies.
    """
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": probe_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True,
    )
    return _generate(prompt, max_new_tokens=8, temperature=0.0, do_sample=False)


di = direction_inference(
    model_answer_fn,
    label="trained_stage1",
    n_probes=20,
    seed_start=200,
    turns_to_warm_up=25,
    difficulty="medium",
)

print(f"  accuracy: {di.accuracy:.1%}  ({di.n_correct}/{di.n_probes} correct)")
print(f"  chance baseline: {di.chance_baseline:.0%}")
print()
print("  Per-probe breakdown:")
for p in di.probes:
    mark = "✓" if p.correct else "✗"
    print(f"    {mark}  seed={p.seed}  tv=${p.true_value:.2f}  "
          f"truth={p.ground_truth}  model={p.model_answer}")

with (RUN_DIR / "tom_direction_inference.json").open("w") as fh:
    json.dump(asdict(di), fh, indent=2)
print(f"\n  Wrote {RUN_DIR / 'tom_direction_inference.json'}")

print("\n" + "=" * 60)
print("ALL DONE — download the three JSON files from the Colab files panel")
print("and paste the headline numbers (mean P&L, alignment rate, ToM accuracy)")
print("back to your assistant for the README + blog.")
print("=" * 60)
