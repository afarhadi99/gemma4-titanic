import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from titanic.env import load_env
from titanic.features import CATEGORICAL_COLS, NUMERIC_COLS, add_group_features, engineer, load_raw
from titanic.submission import validate_submission, write_submission

warnings.filterwarnings("ignore", message=".*probability.*", category=FutureWarning)

GROUP_NUMERIC_COLS = ["GroupSize", "GroupKnownCount", "GroupRate", "GroupNoKnown"]
BASE_MODELS = ["logreg", "hgb", "gb", "rf", "svc"]
EXTRA_MODELS = ["catboost", "xgboost", "lightgbm", "tabpfn", "tabicl"]
ALL_MODELS = BASE_MODELS + EXTRA_MODELS
TABPFN_VERSIONS = ("v2", "v3.5-fast", "v3.5")
OLD_MODELS = ["logreg", "hgb"]
GROUP_PRIOR = 2.0  # pseudo-observations at 0.5 when smoothing the group survival rate


def feature_cols(use_group: bool) -> tuple[list[str], list[str]]:
    return NUMERIC_COLS + (GROUP_NUMERIC_COLS if use_group else []), CATEGORICAL_COLS


def tabpfn_version() -> str:
    """TABPFN_VERSION from the environment; default v3.5-fast with a token, else v2 (no login needed)."""
    default = "v3.5-fast" if os.environ.get("TABPFN_TOKEN") else "v2"
    return os.environ.get("TABPFN_VERSION", "").strip() or default


class FrameModel:
    """Tabular models that take raw categorical columns (no one-hot, no scaling).

    CatBoost, XGBoost and LightGBM use their native categorical support, TabPFN gets integer
    codes plus `categorical_features_indices`, TabICL gets the DataFrame itself.
    """

    def __init__(self, kind: str, seed: int, numeric: list[str], categorical: list[str]):
        self.kind, self.seed, self.numeric, self.categorical = kind, seed, numeric, categorical

    def _prep(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x[self.num_ + self.cat_].copy()
        if self.kind == "catboost":
            x["Pclass"] = x["Pclass"].astype(int).astype(str)
            return x
        if self.kind in ("xgboost", "lightgbm"):
            for col in self.cat_:
                x[col] = x[col].astype(str).astype(pd.CategoricalDtype(self.categories_[col]))  # unseen -> NaN
            return x
        if self.kind == "tabpfn":
            x[self.cat_] = self.encoder_.transform(x[self.cat_].astype(str))
            return x.astype(float)
        for col in self.cat_:  # tabicl encodes object columns itself
            x[col] = x[col].astype(str).astype(object)
        return x

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "FrameModel":
        from sklearn.preprocessing import OrdinalEncoder

        seed = self.seed
        self.num_, self.cat_ = list(self.numeric), list(self.categorical)
        if self.kind == "catboost":  # Pclass is categorical for CatBoost
            self.num_.remove("Pclass")
            self.cat_.append("Pclass")
        self.categories_ = {c: sorted(x[c].astype(str).unique()) for c in self.cat_}
        if self.kind == "tabpfn":
            self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(
                x[self.cat_].astype(str)
            )
        if self.kind == "catboost":
            from catboost import CatBoostClassifier

            self.clf_ = CatBoostClassifier(
                depth=5, iterations=600, learning_rate=0.03, l2_leaf_reg=4, random_seed=seed,
                verbose=0, allow_writing_files=False, thread_count=1, one_hot_max_size=10,
                cat_features=self.cat_,
            )
        elif self.kind == "xgboost":
            from xgboost import XGBClassifier

            self.clf_ = XGBClassifier(
                max_depth=4, n_estimators=400, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
                enable_categorical=True, tree_method="hist", random_state=seed, n_jobs=1,
            )
        elif self.kind == "lightgbm":
            from lightgbm import LGBMClassifier

            self.clf_ = LGBMClassifier(
                num_leaves=12, min_child_samples=15, n_estimators=400, learning_rate=0.04,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                n_jobs=1, verbose=-1,
            )
        elif self.kind == "tabpfn":
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion

            version = ModelVersion(tabpfn_version())
            self.clf_ = TabPFNClassifier.create_default_for_version(
                version,
                device="cpu",
                random_state=seed,
                categorical_features_indices=list(range(len(self.num_), len(self.num_) + len(self.cat_))),
            )
        else:
            from tabicl import TabICLClassifier

            self.clf_ = TabICLClassifier(random_state=seed, device="cpu")
        self.clf_.fit(self._prep(x), y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.clf_.predict_proba(self._prep(x))


def make_model(name: str, seed: int, use_group: bool = True) -> Pipeline | FrameModel:
    numeric, categorical = feature_cols(use_group)
    if name in EXTRA_MODELS:
        return FrameModel(name, seed, numeric, categorical)
    encoder = OneHotEncoder(handle_unknown="ignore")
    scaled = ColumnTransformer([("num", StandardScaler(), numeric), ("cat", encoder, categorical)])
    raw = ColumnTransformer([("num", "passthrough", numeric), ("cat", encoder, categorical)])
    if name == "logreg":
        return Pipeline([("prep", scaled), ("clf", LogisticRegression(C=0.5, max_iter=2000))])
    if name == "svc":
        return Pipeline([("prep", scaled), ("clf", SVC(C=1.0, probability=True, random_state=seed))])
    if name == "rf":
        clf: ClassifierMixin = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, max_features="sqrt", n_jobs=1, random_state=seed
        )
        return Pipeline([("prep", raw), ("clf", clf)])
    if name == "gb":
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed
        )
        return Pipeline([("prep", raw), ("clf", clf)])
    clf = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=150,
        min_samples_leaf=5,
        l2_regularization=1.0,
        random_state=seed,
    )
    return Pipeline([("prep", raw), ("clf", clf)])


def unavailable_reason(name: str) -> str | None:
    """None if the model can run here, else a one-line reason (never includes secrets)."""
    if name not in EXTRA_MODELS:
        return None
    try:
        if name == "tabpfn":
            version = tabpfn_version()
            if version not in TABPFN_VERSIONS:
                return f"TABPFN_VERSION={version!r} is not one of {', '.join(TABPFN_VERSIONS)}"
            if version != "v2" and not os.environ.get("TABPFN_TOKEN"):
                return f"TabPFN {version} needs TABPFN_TOKEN (non-commercial license, ux.priorlabs.ai)"
            os.environ.setdefault("TABPFN_NO_BROWSER", "1")  # never block on a browser login
        # Tiny probe fit: checks the import and that the weights download without a login.
        rng = np.random.RandomState(0)
        probe = pd.DataFrame({"a": rng.rand(40), "Pclass": rng.randint(1, 4, 40).astype(float), "c": ["x", "y"] * 20})
        FrameModel(name, 0, ["a", "Pclass"], ["c"]).fit(probe, np.arange(40) % 2).predict_proba(probe)
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {' '.join(str(exc).split())[:120]}"
    return None


def resolve_models(requested: list[str] | None = None) -> list[str]:
    """Candidate list: `requested` (or BASELINE_MODELS, else all), minus models that cannot run here."""
    if not requested:
        env = os.environ.get("BASELINE_MODELS", "").replace(",", " ").split()
        requested = env or ALL_MODELS
    unknown = [n for n in requested if n not in ALL_MODELS]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}; choose from {ALL_MODELS}")
    names = []
    for name in requested:
        reason = unavailable_reason(name)
        if reason:
            print(f"Skipping {name}: {reason}")
        else:
            names.append(name)
    if not names:
        raise SystemExit("No usable candidate models")
    return names


def model_matrix(df: pd.DataFrame, use_group: bool = True) -> pd.DataFrame:
    numeric, categorical = feature_cols(use_group)
    x = df.copy()
    if use_group:
        n, wins = x["GroupKnownCount"], x["GroupKnownSurvived"]
        x["GroupRate"] = (wins + 0.5 * GROUP_PRIOR) / (n + GROUP_PRIOR)
    x = x[numeric + categorical].copy()
    x[numeric] = x[numeric].astype(float)
    return x


def accuracy(y: np.ndarray, probs: np.ndarray) -> float:
    return float(((probs >= 0.5).astype(int) == y).mean())


def fold_matrices(
    fit: pd.DataFrame, others: list[pd.DataFrame], use_group: bool
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Group features are computed from the fit labels only, then matrices are built."""
    if use_group:
        fit, others = add_group_features(fit, others)
    return model_matrix(fit, use_group), [model_matrix(o, use_group) for o in others]


def cross_validate(
    names: list[str],
    train: pd.DataFrame,
    y: np.ndarray,
    folds: int,
    seed: int,
    use_group: bool = True,
    test: pd.DataFrame | None = None,
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """`test` (label-free) only contributes to group sizes, never to label statistics."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = [val for _, val in splitter.split(train, y)]
    oof = {name: np.zeros(len(y)) for name in names}
    for val_idx in splits:
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        others = [train.iloc[val_idx]] + ([test] if test is not None else [])
        x_fit, (x_val, *_) = fold_matrices(train.iloc[train_idx], others, use_group)
        for name in names:
            model = make_model(name, seed, use_group).fit(x_fit, y[train_idx])
            oof[name][val_idx] = model.predict_proba(x_val)[:, 1]
    return oof, splits


def pick_best(candidates: dict[str, np.ndarray], y: np.ndarray) -> str:
    return max(candidates, key=lambda name: accuracy(y, candidates[name]))


def add_averages(probs: dict[str, np.ndarray], y: np.ndarray, top: int = 3) -> dict[str, np.ndarray]:
    """Add soft-vote averages of the best 2 and best 3 single models (ranked by OOF accuracy)."""
    ranked = sorted(probs, key=lambda n: -accuracy(y, probs[n]))
    out = dict(probs)
    for k in range(2, min(top, len(ranked)) + 1):
        members = ranked[:k]
        out["avg:" + "+".join(members)] = np.mean([probs[m] for m in members], axis=0)
    return out


def average_members(name: str) -> list[str]:
    return name.removeprefix("avg:").split("+")


def run(
    data_dir: Path,
    out_dir: Path,
    folds: int,
    seed: int,
    extra_seeds: tuple[int, ...] = (),
    models: list[str] | None = None,
) -> dict:
    train_raw, test_raw = load_raw(data_dir)
    train, test = engineer(train_raw, test_raw)
    y = train["Survived"].to_numpy().astype(int)

    candidates = resolve_models(models)
    oof, splits = cross_validate(candidates, train, y, folds, seed, test=test)
    oof = add_averages(oof, y)
    chosen = pick_best(oof, y)
    members = average_members(chosen) if chosen.startswith("avg:") else [chosen]

    # Final fit: group statistics from all train labels, applied to train (leave-one-out) and test.
    x_train, (x_test,) = fold_matrices(train, [test], True)
    test_probs = np.mean(
        [make_model(n, seed).fit(x_train, y).predict_proba(x_test)[:, 1] for n in members], axis=0
    )
    test_by_model = {chosen: test_probs}

    seed_oof = {seed: accuracy(y, oof[chosen])}
    for other in extra_seeds:
        other_oof, _ = cross_validate(members, train, y, folds, other, test=test)
        seed_oof[other] = accuracy(y, np.mean([other_oof[n] for n in members], axis=0))

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
    if "tabpfn" in candidates:
        metrics["tabpfn_version"] = tabpfn_version()
    if extra_seeds:
        metrics["chosen_oof_by_seed"] = {str(k): v for k, v in seed_oof.items()}
        metrics["chosen_oof_mean_over_seeds"] = float(np.mean(list(seed_oof.values())))
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
    parser.add_argument(
        "--check-seeds", type=int, nargs="*", default=[7, 123],
        help="extra CV seeds used only to report the chosen model's OOF accuracy (default: 7 123)",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"restrict candidates (default: BASELINE_MODELS env, else all of {' '.join(ALL_MODELS)})",
    )
    load_env()
    args = parser.parse_args()

    metrics = run(args.data_dir, args.out_dir, args.folds, args.seed, tuple(args.check_seeds), args.models)
    print(f"Chosen: {metrics['model']}  OOF accuracy: {metrics['oof_accuracy']:.4f}")
    for name, acc in metrics["candidate_oof_accuracy"].items():
        print(f"  {name:22s} {acc:.4f}")
    if "chosen_oof_mean_over_seeds" in metrics:
        print(f"Chosen model OOF by CV seed: {metrics['chosen_oof_by_seed']}  mean {metrics['chosen_oof_mean_over_seeds']:.4f}")
    print(f"Wrote {args.out_dir} and submissions/baseline.csv")


if __name__ == "__main__":
    main()
