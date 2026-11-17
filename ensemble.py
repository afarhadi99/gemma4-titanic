import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from titanic.features import load_raw
from titanic.submission import validate_submission, write_submission

THRESHOLDS = np.round(np.arange(0.30, 0.7001, 0.01), 2)


@dataclass(frozen=True)
class Blend:
    weights: np.ndarray
    threshold: float
    correct: int
    accuracy: float


def weight_grid(n_runs: int, step: float = 0.05) -> np.ndarray:
    units = round(1 / step)
    if n_runs > 3:
        units = min(units, 10)
    rows = [
        (*c, units - sum(c))
        for c in itertools.product(range(units + 1), repeat=n_runs - 1)
        if sum(c) <= units
    ]
    return np.array(rows, dtype=float) / units


def find_best_blend(
    probs: np.ndarray,
    y: np.ndarray,
    step: float = 0.05,
    thresholds: np.ndarray = THRESHOLDS,
) -> Blend:
    """probs has shape (n_runs, n_rows). Maximises accuracy; ties prefer a threshold
    near 0.5, then weights near equal."""
    n_runs = probs.shape[0]
    grid = weight_grid(n_runs, step)
    blended = grid @ probs
    spread = ((grid - 1 / n_runs) ** 2).sum(axis=1)
    best_key: tuple[int, float, float] | None = None
    best: Blend | None = None
    for threshold in thresholds:
        correct = ((blended >= threshold).astype(int) == y).sum(axis=1)
        for i, count in enumerate(correct):
            key = (-int(count), abs(float(threshold) - 0.5), float(spread[i]))
            if best_key is None or key < best_key:
                best_key = key
                best = Blend(grid[i], float(threshold), int(count), int(count) / len(y))
    assert best is not None
    return best


def load_probs(path: Path, ids: pd.Series) -> np.ndarray:
    frame = pd.read_csv(path).set_index("PassengerId")["prob"]
    missing = set(ids) - set(frame.index)
    if missing:
        raise ValueError(f"{path} lacks PassengerIds: {sorted(missing)[:10]}")
    return frame.loc[ids].to_numpy(dtype=float)


def discover_runs(outputs_dir: Path) -> list[Path]:
    if not outputs_dir.is_dir():
        return []
    return sorted(
        d
        for d in outputs_dir.iterdir()
        if (d / "oof.csv").is_file() and (d / "test_probs.csv").is_file()
    )


def is_smoke(run_dir: Path) -> bool:
    metrics = run_dir / "metrics.json"
    if not metrics.is_file():
        return False
    return bool(json.loads(metrics.read_text()).get("smoke", False))


def coverage_problem(run_dir: Path, train_ids: pd.Series, test_ids: pd.Series) -> str | None:
    for filename, expected in (("oof.csv", train_ids), ("test_probs.csv", test_ids)):
        ids = pd.read_csv(run_dir / filename)["PassengerId"]
        if len(ids) != len(expected) or set(ids) != set(expected):
            kind = "oof" if filename == "oof.csv" else "test_probs"
            return f"{len(ids)} {kind} rows, expected {len(expected)}"
    return None


def usable_runs(candidates: list[Path], data_dir: Path) -> tuple[list[Path], list[str]]:
    train, test = load_raw(data_dir)
    usable: list[Path] = []
    skipped: list[str] = []
    for candidate in candidates:
        if is_smoke(candidate):
            skipped.append(f"skipping {candidate}: smoke run")
            continue
        problem = coverage_problem(candidate, train["PassengerId"], test["PassengerId"])
        if problem:
            skipped.append(f"skipping {candidate}: {problem}")
        else:
            usable.append(candidate)
    return usable, skipped


def run(runs: list[Path], data_dir: Path, out: Path) -> dict:
    train, test = load_raw(data_dir)
    for r in runs:
        problem = coverage_problem(r, train["PassengerId"], test["PassengerId"])
        if problem:
            raise ValueError(f"Run {r} is incomplete: {problem}")
    y = train["Survived"].to_numpy().astype(int)
    oof = np.stack([load_probs(r / "oof.csv", train["PassengerId"]) for r in runs])
    test_probs = np.stack([load_probs(r / "test_probs.csv", test["PassengerId"]) for r in runs])

    blend = find_best_blend(oof, y)
    final = (blend.weights @ test_probs >= blend.threshold).astype(int)
    path = write_submission(test["PassengerId"], final, out)
    validate_submission(path, Path(data_dir) / "test.csv")

    names = [r.name for r in runs]
    choice = {
        "runs": names,
        "weights": dict(zip(names, blend.weights.round(4).tolist())),
        "threshold": blend.threshold,
        "oof_accuracy": blend.accuracy,
        "run_oof_accuracy": {
            n: float(((p >= 0.5).astype(int) == y).mean()) for n, p in zip(names, oof)
        },
        "submission": str(path),
    }
    (path.parent / "ensemble.json").write_text(json.dumps(choice, indent=2))
    return choice


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend run probabilities into a Kaggle submission.")
    parser.add_argument("--runs", nargs="+", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("submissions/submission.csv"))
    args = parser.parse_args()

    runs = args.runs
    if not runs:
        runs, skipped = usable_runs(discover_runs(Path("outputs")), args.data_dir)
        for message in skipped:
            print(message)
    if not runs:
        parser.error("no usable runs found; pass --runs or create outputs/<run>/{oof.csv,test_probs.csv}")

    choice = run(runs, args.data_dir, args.out)
    print("Run OOF accuracy at 0.5:")
    for name, acc in choice["run_oof_accuracy"].items():
        print(f"  {name:20s} {acc:.4f}")
    print(f"Weights:   {choice['weights']}")
    print(f"Threshold: {choice['threshold']:.2f}")
    print(f"Blend OOF accuracy: {choice['oof_accuracy']:.4f}")
    print(f"Wrote {choice['submission']}")


if __name__ == "__main__":
    main()
