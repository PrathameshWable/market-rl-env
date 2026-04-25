"""
Generate the SFT warm-start dataset.

Plays InformedBot (the cheating teacher) as the trainable agent across N
episodes and dumps every (system, observation, action) triple as a single
chat-style JSONL line. The resulting file is what SFTTrainer ingests in
the Colab notebook.

Run from the repo root:
    python -m training.generate_sft_data --episodes 500 --out training/sft_data.jsonl

Defaults are chosen for a quick local run (no GPU, ~2 min on CPU).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from market_env.bots import InformedBot
from market_env.environment import MarketEnvironment

from training.prompts import SYSTEM_PROMPT, format_observation, serialize_action


def generate(
    n_episodes: int,
    out_path: Path,
    *,
    episode_length: int = 50,
    difficulties: tuple[str, ...] = ("easy", "medium"),
    teacher_edge: float = 0.20,
    teacher_qty: int = 10,
) -> dict:
    """Write the SFT JSONL and return summary stats."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = MarketEnvironment()
    action_counts: Counter[str] = Counter()
    n_examples = 0

    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(n_episodes):
            difficulty = difficulties[i % len(difficulties)]
            obs = env.reset(
                seed=i,
                difficulty=difficulty,
                episode_length=episode_length,
                bot_config="default",      # 4 scripted opponents
                trainable_agent_id="agent_1",
            )

            # Seed the teacher with the (private) true value for this episode.
            true_value = env._episodes[obs.episode_id].scenario.true_value
            teacher = InformedBot(
                "agent_1", seed=i, edge=teacher_edge, qty=teacher_qty,
            )
            teacher.set_true_value(true_value)

            done = False
            while not done:
                action = teacher.act(obs)
                example = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": format_observation(obs)},
                        {"role": "assistant", "content": serialize_action(action)},
                    ]
                }
                fh.write(json.dumps(example) + "\n")
                action_counts[action.action_type] += 1
                n_examples += 1

                obs, _, done, _ = env.step(obs.episode_id, action)

            if (i + 1) % 50 == 0:
                print(
                    f"  ep {i + 1:>4}/{n_episodes} | examples so far: {n_examples}"
                )

    return {
        "n_episodes": n_episodes,
        "n_examples": n_examples,
        "out_path": str(out_path),
        "action_counts": dict(action_counts),
        "hold_ratio": action_counts["hold"] / max(n_examples, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument(
        "--out", type=Path, default=Path("training/sft_data.jsonl"),
    )
    parser.add_argument("--episode-length", type=int, default=50)
    args = parser.parse_args()

    print(f"Generating SFT data: {args.episodes} episodes -> {args.out}")
    stats = generate(
        n_episodes=args.episodes,
        out_path=args.out,
        episode_length=args.episode_length,
    )
    print()
    print("Done.")
    print(f"  examples written : {stats['n_examples']}")
    print(f"  action counts    : {stats['action_counts']}")
    print(f"  hold ratio       : {stats['hold_ratio']:.2%}")
    print(f"  output           : {stats['out_path']}")


if __name__ == "__main__":
    main()
