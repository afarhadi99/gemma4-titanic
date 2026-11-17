import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from titanic.features import CATEGORICAL_COLS, NUMERIC_COLS, engineer, load_raw
from titanic.submission import validate_submission, write_submission

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def make_model(name: str, seed: int) -> Pipeline:
    encoder = OneHotEncoder(handle_unknown="ignore")
    if name == "logreg":
        preprocess = ColumnTransformer(
            [("num", StandardScaler(), NUMERIC_COLS), ("cat", encoder, CATEGORICAL_COLS)]
        )
        estimator: ClassifierMixin = LogisticRegression(C=0.5, max_iter=2000)
    else:
        preprocess = ColumnTransformer(
            [("num", "passthrough", NUMERIC_COLS), ("cat", encoder, CATEGORICAL_COLS)]
        )
        estimator = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=150,
            min_samples_leaf=5,
            l2_regularization=1.0,
            random_state=seed,
        )
    return Pipeline([("prep", preprocess), ("clf", estimator)])


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURE_COLS].copy()
    x[NUMERIC_COLS] = x[NUMERIC_COLS].astype(float)
    return x


def accuracy(y: np.ndarray, probs: np.ndarray) -> float:
    return float(((probs >= 0.5).astype(int) == y).mean())


def cross_validate(
    names: list[str], x: pd.DataFrame, y: np.ndarray, folds: int, seed: int
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = [val for _, val in splitter.split(x, y)]
    oof = {name: np.zeros(len(y)) for name in names}
    for val_idx in splits:
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        for name in names:
            model = make_model(name, seed).fit(x.iloc[train_idx], y[train_idx])
            oof[name][val_idx] = model.predict_proba(x.iloc[val_idx])[:, 1]
    return oof, splits


def pick_best(candidates: dict[str, np.ndarray], y: np.ndarray) -> str:
    return max(candidates, key=lambda name: accuracy(y, candidates[name]))


def run(data_dir: Path, out_dir: Path, folds: int, seed: int) -> dict:
    train_raw, test_raw = load_raw(data_dir)
    train, test = engineer(train_raw, test_raw)
    x_train, x_test = model_matrix(train), model_matrix(test)
    y = train["Survived"].to_numpy().astype(int)

    names = ["logreg", "hgb"]
    oof, splits = cross_validate(names, x_train, y, folds, seed)
    oof["average"] = (oof["logreg"] + oof["hgb"]) / 2
    chosen = pick_best(oof, y)

    fitted = {n: make_model(n, seed).fit(x_train, y) for n in names}
    test_by_model = {n: fitted[n].predict_proba(x_test)[:, 1] for n in names}
    test_by_model["average"] = (test_by_model["logreg"] + test_by_model["hgb"]) / 2

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PassengerId": train["PassengerId"], "prob": oof[chosen]}).to_csv(
        out_dir / "oof.csv", index=False
    )
    pd.DataFrame({"PassengerId": test["PassengerId"], "prob": test_by_model[chosen]}).to_csv(
        out_dir / "test_probs.csv", index=False
    )
    metrics = {
        "model": f"baseline:{chosen}",
        "folds": folds,
        "seed": seed,
        "oof_accuracy": accuracy(y, oof[chosen]),
        "fold_accuracies": [accuracy(y[idx], oof[chosen][idx]) for idx in splits],
        "candidate_oof_accuracy": {n: accuracy(y, p) for n, p in oof.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    submission_path = write_submission(
        test["PassengerId"],
        (test_by_model[chosen] >= 0.5).astype(int),
        Path("submissions/baseline.csv"),
    )
    validate_submission(submission_path, Path(data_dir) / "test.csv")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical Titanic baseline with out-of-fold probabilities.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = run(args.data_dir, args.out_dir, args.folds, args.seed)
    print(f"Chosen: {metrics['model']}  OOF accuracy: {metrics['oof_accuracy']:.4f}")
    for name, acc in metrics["candidate_oof_accuracy"].items():
        print(f"  {name:8s} {acc:.4f}")
    print(f"Wrote {args.out_dir} and submissions/baseline.csv")


if __name__ == "__main__":
    main()
