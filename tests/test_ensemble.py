from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import json

from ensemble import discover_runs, find_best_blend, load_probs, run, usable_runs, weight_grid

FIXTURES = Path(__file__).parent / "fixtures"


def test_weight_grid_is_convex() -> None:
    for n in (1, 2, 3, 4):
        grid = weight_grid(n)
        assert grid.shape[1] == n
        assert np.allclose(grid.sum(axis=1), 1.0)
        assert (grid >= 0).all()
    assert len(weight_grid(2)) == 21


def test_blend_prefers_informative_run() -> None:
    y = np.array([0, 1] * 10)
    good = np.where(y == 1, 0.9, 0.1)
    noise = np.full(len(y), 0.5)
    blend = find_best_blend(np.stack([good, noise]), y)
    assert blend.accuracy == 1.0
    assert blend.weights[0] > 0


def test_single_run_finds_threshold() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([[0.1, 0.2, 0.38, 0.42, 0.6, 0.9]])
    blend = find_best_blend(probs, y)
    assert blend.accuracy == 1.0
    assert 0.38 < blend.threshold <= 0.42
    assert blend.weights.tolist() == [1.0]


def test_ties_prefer_threshold_near_half_then_equal_weights() -> None:
    y = np.array([0, 1])
    same = np.array([0.2, 0.8])
    blend = find_best_blend(np.stack([same, same]), y)
    assert blend.threshold == 0.5
    assert np.allclose(blend.weights, [0.5, 0.5])


def test_blend_complementary_runs() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    a = np.array([0.1, 0.1, 0.9, 0.9, 0.9, 0.1])
    b = np.array([0.1, 0.1, 0.9, 0.9, 0.1, 0.9])
    blend = find_best_blend(np.stack([a, b]), y)
    assert blend.accuracy >= max(((a >= 0.5) == y).mean(), ((b >= 0.5) == y).mean())


def write_run(root: Path, name: str, oof: pd.DataFrame, test: pd.DataFrame) -> Path:
    path = root / name
    path.mkdir(parents=True)
    oof.to_csv(path / "oof.csv", index=False)
    test.to_csv(path / "test_probs.csv", index=False)
    return path


def make_runs(root: Path) -> list[Path]:
    train = pd.read_csv(FIXTURES / "train.csv")
    test = pd.read_csv(FIXTURES / "test.csv")
    y = train["Survived"].to_numpy()
    rng = np.random.default_rng(0)
    perfect = pd.DataFrame({"PassengerId": train["PassengerId"], "prob": np.where(y == 1, 0.8, 0.2)})
    noisy = pd.DataFrame({"PassengerId": train["PassengerId"], "prob": rng.random(len(y))})
    test_a = pd.DataFrame({"PassengerId": test["PassengerId"], "prob": np.linspace(0.1, 0.9, len(test))})
    test_b = pd.DataFrame({"PassengerId": test["PassengerId"], "prob": rng.random(len(test))})
    shuffled = lambda df: df.sample(frac=1, random_state=1)  # noqa: E731
    return [
        write_run(root, "a", shuffled(perfect), shuffled(test_a)),
        write_run(root, "b", noisy, test_b),
    ]


def test_discover_runs_requires_both_files(tmp_path: Path) -> None:
    make_runs(tmp_path)
    (tmp_path / "incomplete").mkdir()
    (tmp_path / "incomplete" / "oof.csv").write_text("PassengerId,prob\n")
    assert [p.name for p in discover_runs(tmp_path)] == ["a", "b"]
    assert discover_runs(tmp_path / "missing") == []


def test_load_probs_aligns_by_passenger_id(tmp_path: Path) -> None:
    runs = make_runs(tmp_path)
    train = pd.read_csv(FIXTURES / "train.csv")
    probs = load_probs(runs[0] / "oof.csv", train["PassengerId"])
    assert np.array_equal((probs > 0.5).astype(int), train["Survived"].to_numpy())


def test_load_probs_missing_ids_raises(tmp_path: Path) -> None:
    path = tmp_path / "oof.csv"
    pd.DataFrame({"PassengerId": [1], "prob": [0.5]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="lacks"):
        load_probs(path, pd.Series([1, 2]))


def test_run_writes_valid_submission(tmp_path: Path) -> None:
    runs = make_runs(tmp_path / "outputs")
    out = tmp_path / "submissions" / "submission.csv"
    choice = run(runs, FIXTURES, out)
    assert out.is_file()
    assert (out.parent / "ensemble.json").is_file()
    assert choice["oof_accuracy"] == 1.0
    assert choice["run_oof_accuracy"]["a"] == 1.0
    assert choice["weights"]["a"] > choice["weights"]["b"] or choice["weights"]["a"] > 0
    assert len(pd.read_csv(out)) == 10


def test_usable_runs_skips_smoke_and_incomplete(tmp_path: Path) -> None:
    make_runs(tmp_path)
    train = pd.read_csv(FIXTURES / "train.csv")
    test = pd.read_csv(FIXTURES / "test.csv")
    small_oof = pd.DataFrame({"PassengerId": train["PassengerId"][:5], "prob": 0.5})
    full_test = pd.DataFrame({"PassengerId": test["PassengerId"], "prob": 0.5})
    smoke = write_run(tmp_path, "m-smoke", small_oof, full_test)
    (smoke / "metrics.json").write_text(json.dumps({"smoke": True}))
    write_run(tmp_path, "partial", small_oof, full_test)

    usable, skipped = usable_runs(discover_runs(tmp_path), FIXTURES)

    assert [p.name for p in usable] == ["a", "b"]
    assert any("m-smoke" in m and "smoke" in m for m in skipped)
    assert any("partial" in m and "5 oof rows, expected 24" in m for m in skipped)


def test_explicit_incomplete_run_raises(tmp_path: Path) -> None:
    runs = make_runs(tmp_path / "outputs")
    train = pd.read_csv(FIXTURES / "train.csv")
    test = pd.read_csv(FIXTURES / "test.csv")
    bad = write_run(
        tmp_path / "outputs",
        "bad",
        pd.DataFrame({"PassengerId": train["PassengerId"][:5], "prob": 0.5}),
        pd.DataFrame({"PassengerId": test["PassengerId"], "prob": 0.5}),
    )
    with pytest.raises(ValueError, match="incomplete.*5 oof rows, expected 24"):
        run([*runs, bad], FIXTURES, tmp_path / "s.csv")
