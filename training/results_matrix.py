"""
Aggregate every run under `training/runs/<name>/` into one results matrix.

For each run folder we look for the standard JSON artefacts produced by the
evaluation/probe scripts and assemble a row in a pandas-free table:

    label                | mean_pnl | pnl_std | parse | participation | probe1_imp | probe2 | probe3
    Stage 1 baseline     | +0.0546  |  0.243  | 100%  |   100%        |   0.65    |  0.80  |  0.50
    No curriculum        | ...      |  ...    | ...

Used by `notebooks/analysis.ipynb` and the CLI for the writeup.

CLI:
    python -m training.results_matrix --runs-root training/runs --out training/runs/results_matrix.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Files we look for inside each <runs_root>/<run_name>/ folder.
EVAL_FILE = "eval_trained.json"
TOM_PROBES_FILE = "tom_probes_trained.json"
TOM_DIRECTION_FILE = "tom_direction_inference.json"
BASELINES_FILE = "tom_probes_baselines.json"
EVAL_RANDOM_FILE = "eval_random.json"
EVAL_HOLD_FILE = "eval_hold.json"


@dataclass
class RunRow:
    label: str
    run_dir: str
    mean_pnl_normalized: Optional[float] = None
    pnl_std: Optional[float] = None
    parse_success_rate: Optional[float] = None
    participation_rate: Optional[float] = None
    probe1_improvement: Optional[float] = None
    probe2_alignment_rate: Optional[float] = None
    probe3_accuracy: Optional[float] = None
    probe3_n: Optional[int] = None


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _row_for_run(run_dir: Path, label: Optional[str] = None) -> RunRow:
    label = label or run_dir.name
    row = RunRow(label=label, run_dir=str(run_dir))

    eval_data = _read_json(run_dir / EVAL_FILE)
    if eval_data:
        row.mean_pnl_normalized = eval_data.get("mean_pnl_normalized")
        row.pnl_std = eval_data.get("pnl_std")
        row.parse_success_rate = eval_data.get("parse_success_rate")
        row.participation_rate = eval_data.get("participation_rate")

    probes = _read_json(run_dir / TOM_PROBES_FILE)
    if probes and isinstance(probes, dict):
        # Schema produced by tom_probes.py: {"trained": {"price_efficiency": ..., "signal_alignment": ...}}
        # but earlier runs nested under different keys; tolerate both.
        block = probes.get("trained") or next(iter(probes.values()), {})
        if isinstance(block, dict):
            pe = block.get("price_efficiency", {})
            sa = block.get("signal_alignment", {})
            row.probe1_improvement = pe.get("improvement")
            row.probe2_alignment_rate = sa.get("alignment_rate")

    direction = _read_json(run_dir / TOM_DIRECTION_FILE)
    if direction:
        row.probe3_accuracy = direction.get("accuracy")
        row.probe3_n = direction.get("n_probes")

    return row


def collect_rows(runs_root: Path) -> list[RunRow]:
    """Walk every immediate subdirectory of runs_root that has an eval JSON."""
    rows: list[RunRow] = []
    if not runs_root.exists():
        return rows
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / EVAL_FILE).exists():
            continue
        rows.append(_row_for_run(child))
    return rows


def add_baseline_rows(rows: list[RunRow], runs_root: Path) -> list[RunRow]:
    """Inject scripted-policy baselines (random / hold) from any run folder.

    These don't change between runs — they're properties of the env — so we
    add them once if we find them anywhere under runs_root.
    """
    out = list(rows)
    seen = {r.label for r in rows}
    for run_dir in sorted(runs_root.iterdir()) if runs_root.exists() else []:
        if not run_dir.is_dir():
            continue
        for fname, label in ((EVAL_RANDOM_FILE, "random_baseline"),
                             (EVAL_HOLD_FILE, "hold_baseline")):
            if label in seen:
                continue
            data = _read_json(run_dir / fname)
            if not data:
                continue
            out.append(RunRow(
                label=label,
                run_dir=str(run_dir),
                mean_pnl_normalized=data.get("mean_pnl_normalized"),
                pnl_std=data.get("pnl_std"),
                parse_success_rate=data.get("parse_success_rate"),
                participation_rate=data.get("participation_rate"),
            ))
            seen.add(label)
    return out


def render_markdown(rows: list[RunRow]) -> str:
    """Render rows as a Markdown table for blog.md / README.md."""
    headers = [
        "Run", "Mean P&L", "Std", "Parse OK", "Active",
        "Probe 1 (price-err drop)", "Probe 2 (align)", "Probe 3 (dir acc)",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        def fmt(x, fmt_str=".4f", suffix=""):
            return "—" if x is None else f"{x:{fmt_str}}{suffix}"
        def pct(x):
            return "—" if x is None else f"{x*100:.0f}%"
        lines.append("| " + " | ".join([
            r.label,
            fmt(r.mean_pnl_normalized, "+.4f"),
            fmt(r.pnl_std, ".4f"),
            pct(r.parse_success_rate),
            pct(r.participation_rate),
            fmt(r.probe1_improvement, "+.2f"),
            pct(r.probe2_alignment_rate),
            pct(r.probe3_accuracy),
        ]) + " |")
    return "\n".join(lines)


def save_matrix(rows: list[RunRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rows": [asdict(r) for r in rows]}
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("training/runs"))
    parser.add_argument("--out", type=Path, default=Path("training/runs/results_matrix.json"))
    parser.add_argument("--markdown", type=Path, default=None,
                        help="Optional path to write the Markdown table.")
    args = parser.parse_args()

    rows = collect_rows(args.runs_root)
    rows = add_baseline_rows(rows, args.runs_root)
    save_matrix(rows, args.out)
    md = render_markdown(rows)
    print(md)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md, encoding="utf-8")
        print(f"\n  wrote {args.out} and {args.markdown}")
    else:
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
