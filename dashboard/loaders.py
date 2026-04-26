"""Load saved Stage 1 / M7B artifacts for the dashboard.

All paths are relative to ``training/runs/``. Each run directory is expected
to contain the JSONs written by ``training/evaluate.py`` and
``training/tom_probes.py``:

    eval_trained.json
    eval_random.json
    eval_hold.json
    tom_probes_trained.json
    tom_probes_baselines.json
    tom_direction_inference.json
    training_log.json (optional)

Functions return plain dicts/lists so they're easy to render in Streamlit
without any repo-internal imports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "training" / "runs"


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunInfo:
    name: str
    path: Path
    has_trained_eval: bool
    has_baselines: bool
    has_probes: bool
    has_direction: bool
    has_training_log: bool


def discover_runs() -> list[RunInfo]:
    """Return every run directory under ``training/runs/`` that contains data."""
    if not RUNS_DIR.exists():
        return []
    runs: list[RunInfo] = []
    for child in sorted(RUNS_DIR.iterdir()):
        if not child.is_dir():
            continue
        runs.append(
            RunInfo(
                name=child.name,
                path=child,
                has_trained_eval=(child / "eval_trained.json").exists(),
                has_baselines=(child / "eval_random.json").exists()
                or (child / "eval_hold.json").exists(),
                has_probes=(child / "tom_probes_trained.json").exists(),
                has_direction=(child / "tom_direction_inference.json").exists(),
                has_training_log=(child / "training_log.json").exists(),
            )
        )
    return runs


def default_run() -> Optional[RunInfo]:
    """Pick the most informative run available (preferring stage1 then alphabetical)."""
    runs = [r for r in discover_runs() if r.has_trained_eval or r.has_probes]
    if not runs:
        return None
    # Prefer a run named "stage1*"; otherwise use the last alphabetically.
    stage1 = [r for r in runs if r.name.startswith("stage1")]
    return stage1[0] if stage1 else runs[-1]


# ---------------------------------------------------------------------------
# Eval loaders
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_eval(run: RunInfo, label: str) -> Optional[dict[str, Any]]:
    """Load one of: trained, random, hold."""
    return _read_json(run.path / f"eval_{label}.json")


def load_all_evals(run: RunInfo) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for label in ("trained", "random", "hold"):
        data = load_eval(run, label)
        if data is not None:
            out[label] = data
    return out


# ---------------------------------------------------------------------------
# Probe loaders
# ---------------------------------------------------------------------------

def load_probes(run: RunInfo) -> dict[str, Any]:
    """Merge tom_probes_trained.json + tom_probes_baselines.json into one dict."""
    trained = _read_json(run.path / "tom_probes_trained.json") or {}
    baselines = _read_json(run.path / "tom_probes_baselines.json") or {}
    merged: dict[str, Any] = {}
    merged.update(baselines)
    merged.update(trained)
    return merged


def load_direction_inference(run: RunInfo) -> Optional[dict[str, Any]]:
    return _read_json(run.path / "tom_direction_inference.json")


def load_training_log(run: RunInfo) -> Optional[list[dict[str, Any]]]:
    data = _read_json(run.path / "training_log.json")
    if isinstance(data, list):
        return data
    return None


# ---------------------------------------------------------------------------
# Convenience aggregations
# ---------------------------------------------------------------------------

def summary_kpis(run: RunInfo) -> dict[str, Any]:
    """Return the headline numbers used by the Overview tab."""
    evals = load_all_evals(run)
    probes = load_probes(run)
    direction = load_direction_inference(run)

    out: dict[str, Any] = {
        "run_name": run.name,
    }

    trained = evals.get("trained")
    random_ = evals.get("random")
    hold = evals.get("hold")

    if trained is not None:
        out["trained_pnl_mean"] = trained.get("mean_pnl_normalized")
        out["trained_pnl_median"] = trained.get("median_pnl_normalized")
        out["trained_pnl_std"] = trained.get("pnl_std")
        out["trained_parse_rate"] = trained.get("parse_success_rate")
        out["trained_participation"] = trained.get("participation_rate")
        out["trained_n_episodes"] = trained.get("n_episodes")

    if random_ is not None:
        out["random_pnl_mean"] = random_.get("mean_pnl_normalized")
    if hold is not None:
        out["hold_pnl_mean"] = hold.get("mean_pnl_normalized")

    trained_probe = probes.get("trained", {}) if isinstance(probes, dict) else {}
    if "price_efficiency" in trained_probe:
        pe = trained_probe["price_efficiency"]
        out["probe1_initial_err"] = pe.get("initial_mean_error")
        out["probe1_final_err"] = pe.get("final_mean_error")
        out["probe1_improvement"] = pe.get("improvement")
    if "signal_alignment" in trained_probe:
        sa = trained_probe["signal_alignment"]
        out["probe2_alignment"] = sa.get("alignment_rate")
        out["probe2_n_active"] = sa.get("n_active")

    if direction is not None:
        out["probe3_accuracy"] = direction.get("accuracy")
        out["probe3_n_correct"] = direction.get("n_correct")
        out["probe3_n_probes"] = direction.get("n_probes")

    return out
