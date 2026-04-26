"""
Plot helpers for M6 / M7B writeups.

Same primitives are reused by:
- training/runs/<date>/_regenerate_plots.py (single-run script)
- notebooks/analysis.ipynb                  (cross-run dashboard)
- training/results_matrix.py                (printed via inline mpl)

Each function takes pre-loaded JSON dicts and returns a matplotlib figure
so callers control where to save / display.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "hold":              "#aaaaaa",
    "random":            "#e07b54",
    "informed":          "#5b9bd5",
    "trained":           "#2ca02c",
    "stage1_2026-04-25": "#2ca02c",
    "baseline_replay":   "#1f77b4",
    "no_curriculum":     "#9467bd",
    "aux_direction":     "#d62728",
}


def _color(name: str) -> str:
    if name in PALETTE:
        return PALETTE[name]
    digest = abs(hash(name)) % len(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    return plt.rcParams["axes.prop_cycle"].by_key()["color"][digest]


# ---------------------------------------------------------------------------
# Single-run plots (replacements for _regenerate_plots.py)
# ---------------------------------------------------------------------------

def plot_price_efficiency(baselines: dict, trained_block: dict | None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    series = dict(baselines)
    if trained_block is not None:
        series["trained"] = trained_block
    for name, payload in series.items():
        pe = payload.get("price_efficiency") or {}
        ys = [v if v is not None else float("nan") for v in pe.get("mean_error_by_turn", [])]
        if not ys:
            continue
        xs = list(range(len(ys)))
        final = pe.get("final_mean_error", float("nan"))
        ax.plot(xs, ys, label=f"{name}  (final=${final:.2f})",
                color=_color(name), linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean |mid_price - true_value|  ($)")
    ax.set_title("Probe 1 - Price efficiency")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_signal_alignment(baselines: dict, trained_block: dict | None) -> plt.Figure:
    series = dict(baselines)
    if trained_block is not None:
        series["trained"] = trained_block
    names = list(series.keys())
    rates = [series[n]["signal_alignment"]["alignment_rate"] for n in names]
    n_active = [series[n]["signal_alignment"]["n_active"] for n in names]
    colors = [_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, rates, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Chance (50%)")
    for bar, na, rate in zip(bars, n_active, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{rate:.0%}\n(n={na})", ha="center", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Alignment rate")
    ax.set_title("Probe 2 - Signal alignment")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pnl_comparison(policies: dict[str, dict]) -> plt.Figure:
    """policies: ordered map of {label -> eval_summary_dict}."""
    names = list(policies.keys())
    means = [p["mean_pnl_normalized"] for p in policies.values()]
    stds = [p.get("pnl_std", 0.0) for p in policies.values()]
    colors = [_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(names)), 4))
    bars = ax.bar(names, means, color=colors, alpha=0.85, edgecolor="white",
                  yerr=stds, capsize=6, error_kw={"linewidth": 1.5})
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                max(mean, 0) + std + 0.01,
                f"{mean:+.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean normalized P&L  (+/- 1 std)")
    ax.set_title("Evaluation - 10 held-out scenarios")
    if means and stds:
        ax.set_ylim(min(means) - max(stds) - 0.05, max(means) + max(stds) + 0.08)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    return fig


def plot_direction_inference(direction: dict) -> plt.Figure:
    probes = direction["probes"]
    correct = [p["correct"] for p in probes]
    seeds = [p["seed"] for p in probes]
    tv = [p["true_value"] for p in probes]
    xs = list(range(len(probes)))

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 5),
                                          gridspec_kw={"height_ratios": [1, 2]})
    cumulative = [sum(correct[:i + 1]) / (i + 1) for i in range(len(correct))]
    ax_top.plot(xs, cumulative, color=PALETTE["trained"], linewidth=2)
    ax_top.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Chance")
    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel("Cumulative accuracy")
    ax_top.set_title(f"Probe 3 - Direction inference (signal-free)\n"
                     f"Final accuracy: {direction['accuracy']:.0%} "
                     f"({direction['n_correct']}/{direction['n_probes']} correct)")
    ax_top.legend(fontsize=9)
    ax_top.grid(alpha=0.3)

    bar_colors = [PALETTE["trained"] if c else "#dddddd" for c in correct]
    ax_bot.bar(xs, tv, color=bar_colors, alpha=0.85, edgecolor="white")
    ax_bot.axhline(50, color="black", linewidth=1, linestyle="--", label="$50 midpoint")
    ax_bot.set_xticks(xs)
    ax_bot.set_xticklabels([str(s) for s in seeds], rotation=45, fontsize=8)
    ax_bot.set_ylabel("True value ($)")
    ax_bot.set_xlabel("Probe seed")

    correct_patch = mpatches.Patch(color=PALETTE["trained"], label="Correct")
    wrong_patch = mpatches.Patch(color="#dddddd", label="Wrong (predicted 'above')")
    ax_bot.legend(handles=[correct_patch, wrong_patch], fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cross-run plots (M7B only)
# ---------------------------------------------------------------------------

def plot_pnl_matrix(rows: Iterable[dict]) -> plt.Figure:
    """Bar chart of mean P&L across every run row from results_matrix."""
    rows = [r for r in rows if r.get("mean_pnl_normalized") is not None]
    rows.sort(key=lambda r: r["mean_pnl_normalized"])
    labels = [r["label"] for r in rows]
    means = [r["mean_pnl_normalized"] for r in rows]
    stds = [r.get("pnl_std") or 0.0 for r in rows]
    colors = [_color(n) for n in labels]

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(labels)), 4.2))
    bars = ax.bar(labels, means, color=colors, alpha=0.9, edgecolor="white",
                  yerr=stds, capsize=5, error_kw={"linewidth": 1.2})
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + 0.005 * (1 if mean >= 0 else -1),
                f"{mean:+.3f}", ha="center",
                va="bottom" if mean >= 0 else "top",
                fontsize=9, fontweight="bold")
    ax.set_ylabel("Mean normalized P&L")
    ax.set_title("Stage 1 + M7B ablations - mean P&L (10 held-out seeds)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return fig


def plot_probe_matrix(rows: Iterable[dict]) -> plt.Figure:
    """Grouped bar chart of Probes 1/2/3 across runs that have ToM data."""
    rows = [r for r in rows if r.get("probe2_alignment_rate") is not None]
    if not rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No ToM probe data found in any run.",
                ha="center", va="center")
        ax.axis("off")
        return fig

    labels = [r["label"] for r in rows]
    p1 = [(r.get("probe1_improvement") or 0.0) for r in rows]
    p2 = [r["probe2_alignment_rate"] for r in rows]
    p3 = [(r.get("probe3_accuracy") or 0.0) for r in rows]

    x = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(labels)), 4.5))
    ax.bar(x - width, p1, width, label="Probe 1 (price-err drop, $)",
           color="#1f77b4", alpha=0.85)
    ax.bar(x, p2, width, label="Probe 2 (signal-align rate)",
           color="#ff7f0e", alpha=0.85)
    ax.bar(x + width, p3, width, label="Probe 3 (direction acc)",
           color="#2ca02c", alpha=0.85)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("ToM probes across runs")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_summary_dashboard(
    baselines: dict,
    trained_block: dict | None,
    eval_policies: dict[str, dict],
    direction: dict | None,
) -> plt.Figure:
    """Single-page summary used in the writeup."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: P&L
    names = list(eval_policies.keys())
    means = [p["mean_pnl_normalized"] for p in eval_policies.values()]
    stds = [p.get("pnl_std", 0.0) for p in eval_policies.values()]
    colors = [_color(n) for n in names]
    axes[0].bar(names, means, color=colors, alpha=0.85, yerr=stds, capsize=5)
    axes[0].axhline(0, color="black", linewidth=0.8)
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[0].text(i, max(m, 0) + s + 0.01, f"{m:+.3f}",
                     ha="center", fontsize=9, fontweight="bold")
    axes[0].set_title("P&L (10 held-out scenarios)")
    axes[0].set_ylabel("Mean normalized P&L")
    plt.setp(axes[0].get_xticklabels(), rotation=15, ha="right")

    # Panel 2: Signal alignment
    series = dict(baselines)
    if trained_block is not None:
        series["trained"] = trained_block
    sa_names = list(series.keys())
    sa_rates = [series[n]["signal_alignment"]["alignment_rate"] for n in sa_names]
    sa_colors = [_color(n) for n in sa_names]
    axes[1].bar(sa_names, sa_rates, color=sa_colors, alpha=0.85)
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Chance")
    for i, r in enumerate(sa_rates):
        axes[1].text(i, r + 0.02, f"{r:.0%}", ha="center", fontsize=9)
    axes[1].set_ylim(0, 1.2)
    axes[1].set_title("Probe 2 - Signal alignment")
    axes[1].set_ylabel("Alignment rate")
    axes[1].legend(fontsize=8)
    plt.setp(axes[1].get_xticklabels(), rotation=15, ha="right")

    # Panel 3: Probe 3 gauge
    if direction is not None:
        acc = direction["accuracy"]
        theta = np.linspace(0, np.pi, 200)
        axes[2].plot(np.cos(theta), np.sin(theta), color="#dddddd",
                     linewidth=20, solid_capstyle="round")
        fill_theta = np.linspace(0, np.pi * acc, 200)
        gcolor = PALETTE["trained"] if acc > 0.6 else PALETTE["random"]
        axes[2].plot(np.cos(fill_theta), np.sin(fill_theta), color=gcolor,
                     linewidth=20, solid_capstyle="round")
        axes[2].text(0, -0.15, f"{acc:.0%}", ha="center", va="top",
                     fontsize=28, fontweight="bold", color=gcolor)
        axes[2].text(0, 0.55,
                     f"Probe 3\nDirection inference\n(chance = 50%)",
                     ha="center", fontsize=9)
    axes[2].set_xlim(-1.3, 1.3)
    axes[2].set_ylim(-0.2, 1.2)
    axes[2].set_aspect("equal")
    axes[2].axis("off")

    fig.suptitle("Evaluation Summary", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
