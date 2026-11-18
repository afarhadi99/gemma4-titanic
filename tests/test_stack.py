import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stack import C_GRID, logit, nested_cv, run, select_runs, top_k_means
from titanic.features import load_raw

FIXTURES = Path(__file__).parent / "fixtures"


def _fake_run(root: Path, name: str, noise: float, seed: int, **metrics) -> Path:
    train, test = load_raw(FIXTURES)
    rng = np.random.RandomState(seed)
    y = train["Survived"].to_numpy()
    d = root / name
    d.mkdir(parents=True)
    oof = np.clip(y * 0.7 + 0.15 + rng.randn(len(y)) * noise, 0.02, 0.98)
    pd.DataFrame({"PassengerId": train["PassengerId"], "prob": oof}).to_csv(d / "oof.csv", index=False)
    pd.DataFrame({"PassengerId": test["PassengerId"], "prob": rng.rand(len(test))}).to_csv(d / "test_probs.csv", index=False)
    (d / "metrics.json").write_text(json.dumps(metrics))
    return d


@pytest.fixture
def runs(tmp_path: Path) -> list[Path]:
    root = tmp_path / "outputs"
    return [
        _fake_run(root, "baseline", 0.15, 1, model="baseline:x", split_seed=42),
        _fake_run(root, "gemma-a", 0.3, 2, model="llm-lora:g", split_seed=42),
        _fake_run(root, "gemma-b", 0.3, 3, model="llm-lora:g", split_seed=42),
    ]


def test_stack_writes_validated_submission_and_json(runs: list[Path], tmp_path: Path) -> None:
    out = tmp_path / "submissions" / "stack.csv"
    result = run(runs, FIXTURES, out, seeds=(0, 1))
    assert out.is_file() and (out.parent / "stack.json").is_file()
    saved = json.loads((out.parent / "stack.json").read_text())
    assert saved["runs"] == ["baseline", "gemma-a", "gemma-b"]
    assert saved["C"] in C_GRID
    assert set(saved["coefficients"]) == set(saved["runs"])
    assert 0.0 <= saved["nested_cv"]["mean"] <= 1.0 and set(saved["nested_cv"]["by_seed"]) == {"0", "1"}
    assert set(saved["comparison"]) >= {"best_single_run", "ensemble_blend_in_sample", "top_k_mean_in_sample", "stack_nested_cv"}
    assert result["nested_cv"]["mean"] == saved["nested_cv"]["mean"]


def test_group_same_model_averages_runs(runs: list[Path], tmp_path: Path) -> None:
    result = run(runs, FIXTURES, tmp_path / "s.csv", seeds=(0,), group_same_model=True)
    assert result["runs"] == ["baseline:x", "llm-lora:g (x2)"]


def test_stack_needs_two_runs(runs: list[Path], tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run(runs[:1], FIXTURES, tmp_path / "s.csv")


def test_select_runs_skips_other_split_seed_and_smoke(runs: list[Path], tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    other = _fake_run(root, "other-split", 0.2, 4, split_seed=7)
    smoke = _fake_run(root, "smoke", 0.2, 5, smoke=True)
    kept, skipped = select_runs(runs + [other, smoke], FIXTURES)
    assert kept == runs
    assert any("split_seed 7" in m for m in skipped) and any("smoke" in m for m in skipped)


def test_nested_cv_is_deterministic_and_logit_is_finite() -> None:
    rng = np.random.RandomState(0)
    y = np.array([0, 1] * 30)
    x = logit(np.clip(y * 0.6 + 0.2 + rng.randn(60) * 0.2, 0, 1))[:, None]
    x = np.hstack([x, rng.randn(60, 1)])
    a, b = nested_cv(x, y, (0,)), nested_cv(x, y, (0,))
    assert a == b and a["mean"] > 0.7
    assert np.isfinite(logit(np.array([0.0, 1.0]))).all()


def test_top_k_means_rank_by_own_accuracy() -> None:
    y = np.array([0, 1, 0, 1])
    probs = np.stack([y * 0.8 + 0.1, 1 - (y * 0.8 + 0.1), np.full(4, 0.5)], axis=1)
    out = top_k_means(probs, y, ["good", "bad", "flat"])
    assert out["top2"] == 1.0 and out["all"] == 0.5
    assert set(out) == {"top2", "all"}
