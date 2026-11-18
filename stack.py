"""Stack the out-of-fold probabilities of every complete run in outputs/ with a logistic regression.

Features are the logits of each run's probability. The meta-model's C is tuned by an inner CV and
the whole procedure (inner tuning included) is scored by nested CV: outer 5-fold x several seeds.
Caveat: the base runs' OOF probabilities were themselves produced by one fixed split, so the
nested score is the usual slightly optimistic stacking estimate, not a fresh holdout.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

from ensemble import discover_runs, find_best_blend, load_probs, usable_runs
from titanic.features import load_raw
from titanic.submission import validate_submission, write_submission

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
DEFAULT_SEEDS = (42, 7, 123)
SPLIT_SEED = 42  # the split seed all runs must share for their OOF probabilities to line up
EPS = 1e-4


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def accuracy(y: np.ndarray, probs: np.ndarray) -> float:
    return float(((probs >= 0.5).astype(int) == y).mean())


def read_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def select_runs(candidates: list[Path], data_dir: Path) -> tuple[list[Path], list[str]]:
    """Complete, non-smoke runs whose recorded split_seed (if any) is 42."""
    usable, skipped = usable_runs(candidates, data_dir)
    kept: list[Path] = []
    for run_dir in usable:
        split = read_metrics(run_dir).get("split_seed")
        if split is not None and split != SPLIT_SEED:
            skipped.append(f"skipping {run_dir}: split_seed {split} != {SPLIT_SEED}")
        else:
            kept.append(run_dir)
    return kept, skipped


def build_matrix(
    runs: list[Path], train_ids: pd.Series, test_ids: pd.Series, group_same_model: bool
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Returns (names, oof logits (n_train, k), test logits (n_test, k)). With `group_same_model`,
    runs whose metrics.json "model" matches are averaged (in logit space) into one column."""
    groups: dict[str, list[Path]] = {}
    for run_dir in runs:
        key = read_metrics(run_dir).get("model") if group_same_model else None
        groups.setdefault(key or run_dir.name, []).append(run_dir)
    names, oof_cols, test_cols = [], [], []
    for name, members in groups.items():
        names.append(name if len(members) == 1 else f"{name} (x{len(members)})")
        oof_cols.append(np.mean([logit(load_probs(m / "oof.csv", train_ids)) for m in members], axis=0))
        test_cols.append(np.mean([logit(load_probs(m / "test_probs.csv", test_ids)) for m in members], axis=0))
    return names, np.stack(oof_cols, axis=1), np.stack(test_cols, axis=1)


def make_meta(seed: int, inner_folds: int = 5) -> LogisticRegressionCV:
    """LogisticRegression whose C is chosen from C_GRID by inner CV (log-loss, more stable than accuracy)."""
    return LogisticRegressionCV(
        Cs=C_GRID,
        cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed),
        scoring="neg_log_loss",
        max_iter=2000,
    )


def nested_cv(x: np.ndarray, y: np.ndarray, seeds: tuple[int, ...], folds: int = 5) -> dict:
    """Outer `folds`-fold CV per seed; C is tuned by an inner CV on the outer-train part only."""
    per_seed = []
    for seed in seeds:
        pred = np.zeros(len(y))
        for fit_idx, val_idx in StratifiedKFold(folds, shuffle=True, random_state=seed).split(x, y):
            meta = make_meta(seed).fit(x[fit_idx], y[fit_idx])
            pred[val_idx] = meta.predict_proba(x[val_idx])[:, 1]
        per_seed.append(accuracy(y, pred))
    return {
        "by_seed": {str(s): a for s, a in zip(seeds, per_seed)},
        "mean": float(np.mean(per_seed)),
        "std": float(np.std(per_seed)),
    }


def top_k_means(probs: np.ndarray, y: np.ndarray, names: list[str]) -> dict[str, float]:
    """Simple mean of the k best runs (ranked by their own OOF accuracy, so slightly optimistic)."""
    order = sorted(range(len(names)), key=lambda i: -accuracy(y, probs[:, i]))
    out = {}
    for k in sorted({2, 3, len(names)}):
        if 2 <= k <= len(names):
            out[f"top{k}" if k < len(names) else "all"] = accuracy(y, probs[:, order[:k]].mean(axis=1))
    return out


def run(
    runs: list[Path],
    data_dir: Path,
    out: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    group_same_model: bool = False,
) -> dict:
    train, test = load_raw(data_dir)
    y = train["Survived"].to_numpy().astype(int)
    names, x_train, x_test = build_matrix(runs, train["PassengerId"], test["PassengerId"], group_same_model)
    if len(names) < 2:
        raise ValueError(f"Stacking needs at least 2 runs (or groups), found {len(names)}: {names}")

    probs = 1 / (1 + np.exp(-x_train))  # back to probabilities (grouped runs: mean logit)
    single = {n: accuracy(y, probs[:, i]) for i, n in enumerate(names)}
    blend = find_best_blend(probs.T, y)
    nested = nested_cv(x_train, y, seeds)
    topk = top_k_means(probs, y, names)

    meta = make_meta(seeds[0]).fit(x_train, y)
    final = meta.predict_proba(x_test)[:, 1]
    path = write_submission(test["PassengerId"], (final >= 0.5).astype(int), out)
    validate_submission(path, Path(data_dir) / "test.csv")

    result = {
        "runs": names,
        "group_same_model": group_same_model,
        "split_seed": SPLIT_SEED,
        "coefficients": dict(zip(names, meta.coef_[0].round(4).tolist())),
        "intercept": float(meta.intercept_[0]),
        "C": float(meta.C_[0]),
        "c_grid": C_GRID,
        "nested_cv": nested,
        "comparison": {
            "best_single_run": {"name": max(single, key=single.get), "oof_accuracy": max(single.values())},
            "ensemble_blend_in_sample": {
                "weights": dict(zip(names, blend.weights.round(3).tolist())),
                "oof_accuracy": blend.accuracy,
            },
            "top_k_mean_in_sample": topk,
            "stack_nested_cv": nested["mean"],
        },
        "run_oof_accuracy": single,
        "submission": str(path),
    }
    (path.parent / "stack.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Logistic-regression stack of run OOF logits, with nested CV.")
    parser.add_argument("--runs", nargs="+", type=Path, default=None, help="run dirs (default: all usable under --outputs)")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("submissions/stack.csv"))
    parser.add_argument("--cv-seeds", default=",".join(map(str, DEFAULT_SEEDS)), help="outer-CV seeds (default 42,7,123)")
    parser.add_argument(
        "--group-same-model", action="store_true",
        help="average runs with the same metrics.json 'model' (e.g. Gemma seeds) into one stack feature first",
    )
    args = parser.parse_args()

    candidates = args.runs or discover_runs(args.outputs)
    runs, skipped = select_runs(candidates, args.data_dir)
    for message in skipped:
        print(message)
    if len(runs) < 2:
        parser.error("need at least 2 complete non-smoke runs (split_seed 42) to stack")
    seeds = tuple(int(s) for s in args.cv_seeds.replace(",", " ").split())

    r = run(runs, args.data_dir, args.out, seeds, args.group_same_model)
    print("Stack inputs (own OOF accuracy at 0.5, coefficient on the logit):")
    for name in r["runs"]:
        print(f"  {name:40s} {r['run_oof_accuracy'][name]:.4f}  coef {r['coefficients'][name]:+.3f}")
    cmp_ = r["comparison"]
    print(f"Meta-model C = {r['C']} (inner CV over {r['c_grid']})")
    print("Comparison (OOF accuracy):")
    print(f"  best single run          {cmp_['best_single_run']['oof_accuracy']:.4f}  ({cmp_['best_single_run']['name']})")
    print(f"  ensemble.py blend        {cmp_['ensemble_blend_in_sample']['oof_accuracy']:.4f}  (weights tuned in-sample)")
    for k, v in cmp_["top_k_mean_in_sample"].items():
        print(f"  mean of {k:<14s} {v:.4f}  (ranked in-sample)")
    nested = r["nested_cv"]
    print(f"  nested-CV stack          {nested['mean']:.4f}  std {nested['std']:.4f}  by seed {nested['by_seed']}")
    print(f"Wrote {r['submission']} and {Path(r['submission']).parent / 'stack.json'}")


if __name__ == "__main__":
    main()
