"""
Regenerate all M6 plots with the trained model included.
Run from the repo root:
    python training/runs/stage1_2026-04-25/_regenerate_plots.py
"""
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RUN = Path("training/runs/stage1_2026-04-25")

# ── Load data ─────────────────────────────────────────────────────────────────
baselines = json.loads((RUN / "tom_probes_baselines.json").read_text())
trained_probes = json.loads((RUN / "tom_probes_trained.json").read_text())
eval_trained = json.loads((RUN / "eval_trained.json").read_text())
eval_random = json.loads((RUN / "eval_random.json").read_text())
eval_hold = json.loads((RUN / "eval_hold.json").read_text())
direction = json.loads((RUN / "tom_direction_inference.json").read_text())

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    "hold":     "#aaaaaa",
    "random":   "#e07b54",
    "informed": "#5b9bd5",
    "trained":  "#2ca02c",
}


# ── Plot 1: Price efficiency with trained model added ─────────────────────────
def plot_price_efficiency():
    fig, ax = plt.subplots(figsize=(9, 4.5))

    all_data = {**baselines, "trained": trained_probes["trained"]}
    for name, payload in all_data.items():
        pe = payload["price_efficiency"]
        ys = [v if v is not None else float("nan")
              for v in pe["mean_error_by_turn"]]
        xs = list(range(len(ys)))
        final = pe["final_mean_error"]
        ax.plot(xs, ys, label=f"{name}  (final=${final:.2f})",
                color=COLORS.get(name, "#888"), linewidth=2)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean |mid_price − true_value|  ($)")
    ax.set_title("Probe 1 — Price efficiency\n"
                 "Lower = market prices closer to hidden true value")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RUN / "tom_price_efficiency.png", dpi=150)
    plt.close()
    print("  tom_price_efficiency.png updated")


# ── Plot 2: Signal alignment bar chart with trained model ─────────────────────
def plot_signal_alignment():
    all_data = {**baselines, "trained": trained_probes["trained"]}
    names = list(all_data.keys())
    rates = [all_data[n]["signal_alignment"]["alignment_rate"] for n in names]
    n_active = [all_data[n]["signal_alignment"]["n_active"] for n in names]
    colors = [COLORS.get(n, "#888") for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, rates, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Chance (50%)")
    for bar, na, rate in zip(bars, n_active, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{rate:.0%}\n(n={na})",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Alignment rate")
    ax.set_title("Probe 2 — Signal alignment\n"
                 "Buy when signals positive, sell when negative")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RUN / "tom_signal_alignment.png", dpi=150)
    plt.close()
    print("  tom_signal_alignment.png updated")


# ── Plot 3: P&L comparison (new plot) ────────────────────────────────────────
def plot_pnl_comparison():
    policies = {
        "hold":    eval_hold,
        "random":  eval_random,
        "trained": eval_trained,
    }
    names = list(policies.keys())
    means = [p["mean_pnl_normalized"] for p in policies.values()]
    stds  = [p["pnl_std"] for p in policies.values()]
    colors = [COLORS.get(n, "#888") for n in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, means, color=colors, alpha=0.85, edgecolor="white",
                  yerr=stds, capsize=6, error_kw={"linewidth": 1.5})
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                max(mean, 0) + std + 0.01,
                f"{mean:+.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean normalized P&L  (±1 std)")
    ax.set_title("Evaluation — 10 held-out scenarios\n"
                 "Trained agent beats both baselines")
    ax.set_ylim(min(means) - max(stds) - 0.05,
                max(means) + max(stds) + 0.08)
    fig.tight_layout()
    fig.savefig(RUN / "eval_pnl_comparison.png", dpi=150)
    plt.close()
    print("  eval_pnl_comparison.png created")


# ── Plot 4: Probe 3 per-seed result (new plot) ────────────────────────────────
def plot_direction_inference():
    probes = direction["probes"]
    seeds = [p["seed"] for p in probes]
    correct = [p["correct"] for p in probes]
    tv = [p["true_value"] for p in probes]

    xs = list(range(len(probes)))
    colors_dot = [COLORS["trained"] if c else COLORS["random"] for c in correct]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 5),
                                          gridspec_kw={"height_ratios": [1, 2]})

    # Top: cumulative accuracy
    cumulative = [sum(correct[:i+1]) / (i+1) for i in range(len(correct))]
    ax_top.plot(xs, cumulative, color=COLORS["trained"], linewidth=2)
    ax_top.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Chance")
    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel("Cumulative accuracy")
    ax_top.set_title(f"Probe 3 — Direction inference (signal-free)\n"
                     f"Final accuracy: {direction['accuracy']:.0%} "
                     f"({direction['n_correct']}/{direction['n_probes']} correct) "
                     f"vs 50% chance")
    ax_top.legend(fontsize=9)
    ax_top.grid(alpha=0.3)

    # Bottom: per-probe bar showing true_value
    bar_colors = [COLORS["trained"] if c else "#dddddd" for c in correct]
    ax_bot.bar(xs, tv, color=bar_colors, alpha=0.85, edgecolor="white")
    ax_bot.axhline(50, color="black", linewidth=1, linestyle="--", label="$50 midpoint")
    ax_bot.set_xticks(xs)
    ax_bot.set_xticklabels([str(s) for s in seeds], rotation=45, fontsize=8)
    ax_bot.set_ylabel("True value ($)")
    ax_bot.set_xlabel("Probe seed")
    ax_bot.legend(fontsize=9)
    ax_bot.grid(alpha=0.2, axis="y")

    correct_patch = mpatches.Patch(color=COLORS["trained"], label="Correct")
    wrong_patch = mpatches.Patch(color="#dddddd", label="Wrong (predicted 'above')")
    ax_bot.legend(handles=[correct_patch, wrong_patch], fontsize=9)

    fig.tight_layout()
    fig.savefig(RUN / "tom_direction_inference.png", dpi=150)
    plt.close()
    print("  tom_direction_inference.png created")


# ── Plot 5: Full summary dashboard ────────────────────────────────────────────
def plot_summary_dashboard():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: P&L
    policies = {"hold": eval_hold, "random": eval_random, "trained": eval_trained}
    names = list(policies.keys())
    means = [p["mean_pnl_normalized"] for p in policies.values()]
    stds = [p["pnl_std"] for p in policies.values()]
    colors = [COLORS.get(n, "#888") for n in names]
    axes[0].bar(names, means, color=colors, alpha=0.85,
                yerr=stds, capsize=5, error_kw={"linewidth": 1.2})
    axes[0].axhline(0, color="black", linewidth=0.8)
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[0].text(i, max(m, 0) + s + 0.01, f"{m:+.3f}",
                     ha="center", fontsize=9, fontweight="bold")
    axes[0].set_title("P&L (10 held-out scenarios)")
    axes[0].set_ylabel("Mean normalized P&L")

    # Panel 2: Signal alignment
    all_data = {**baselines, "trained": trained_probes["trained"]}
    sa_names = list(all_data.keys())
    sa_rates = [all_data[n]["signal_alignment"]["alignment_rate"] for n in sa_names]
    sa_colors = [COLORS.get(n, "#888") for n in sa_names]
    axes[1].bar(sa_names, sa_rates, color=sa_colors, alpha=0.85)
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Chance")
    for i, r in enumerate(sa_rates):
        axes[1].text(i, r + 0.02, f"{r:.0%}", ha="center", fontsize=9)
    axes[1].set_ylim(0, 1.2)
    axes[1].set_title("Probe 2 — Signal alignment")
    axes[1].set_ylabel("Alignment rate")
    axes[1].legend(fontsize=8)

    # Panel 3: Probe 3 gauge
    acc = direction["accuracy"]
    theta = np.linspace(0, np.pi, 200)
    axes[2].plot(np.cos(theta), np.sin(theta), color="#dddddd", linewidth=20, solid_capstyle="round")
    fill_theta = np.linspace(0, np.pi * acc, 200)
    color = COLORS["trained"] if acc > 0.6 else COLORS["random"]
    axes[2].plot(np.cos(fill_theta), np.sin(fill_theta), color=color, linewidth=20, solid_capstyle="round")
    axes[2].set_xlim(-1.3, 1.3)
    axes[2].set_ylim(-0.2, 1.2)
    axes[2].set_aspect("equal")
    axes[2].axis("off")
    axes[2].text(0, -0.15, f"{acc:.0%}", ha="center", va="top", fontsize=28, fontweight="bold", color=color)
    axes[2].text(0, 0.55, f"Probe 3\nDirection inference\n(chance = 50%)", ha="center", fontsize=9)

    fig.suptitle("Stage 1 — M6 Evaluation Summary", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RUN / "m6_summary_dashboard.png", dpi=150)
    plt.close()
    print("  m6_summary_dashboard.png created")


if __name__ == "__main__":
    print("Regenerating M6 plots...")
    plot_price_efficiency()
    plot_signal_alignment()
    plot_pnl_comparison()
    plot_direction_inference()
    plot_summary_dashboard()
    print("Done. 5 plots saved to", RUN)
