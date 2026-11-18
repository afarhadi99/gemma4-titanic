"""Feature matrices, models and fold-aware cross-validation shared by baseline.py and tabpfn_modal.py.

Everything that decides which rows go into which fold and which columns a model sees lives here,
so the local baseline and the Modal GPU path cannot drift apart. No GPU library is imported at
module level (torch, tabpfn and tabicl are imported lazily inside FrameModel.fit).
"""

import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from titanic.features import (
    CATEGORICAL_COLS,
    FEATURE_GROUPS,
    NUMERIC_COLS,
    add_group_features,
)

warnings.filterwarnings("ignore", message=".*probability.*", category=FutureWarning)

GROUP_NUMERIC_COLS = ["GroupSize", "GroupKnownCount", "GroupRate", "GroupNoKnown"]
BASE_MODELS = ["logreg", "hgb", "gb", "rf", "svc"]
EXTRA_MODELS = ["catboost", "xgboost", "lightgbm", "tabpfn", "tabicl"]
EMB_COMPONENTS = 16
TABPFN_VERSIONS = ("v2", "v3.5-fast", "v3.5")
# Optional feature groups from titanic.features.FEATURE_GROUPS that are switched on (decided on
# repeated-CV mean accuracy, see README). Override with --features.
DEFAULT_FEATURES: list[str] = []
ACTIVE_FEATURES: list[str] = list(DEFAULT_FEATURES)
GROUP_PRIOR = 2.0  # pseudo-observations at 0.5 when smoothing the group survival rate


def set_features(groups: list[str]) -> None:
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"Unknown feature group(s) {unknown}; choose from {list(FEATURE_GROUPS)}")
    ACTIVE_FEATURES[:] = groups


def emb_cols(n_emb: int) -> list[str]:
    return [f"Emb{i}" for i in range(n_emb)]


def feature_cols(use_group: bool, n_emb: int = 0) -> tuple[list[str], list[str]]:
    numeric = NUMERIC_COLS + (GROUP_NUMERIC_COLS if use_group else [])
    categorical = list(CATEGORICAL_COLS)
    for group in ACTIVE_FEATURES:
        if group == "wcg" and not use_group:
            continue
        extra_num, extra_cat = FEATURE_GROUPS[group]
        numeric = numeric + extra_num
        categorical = categorical + extra_cat
    return numeric + emb_cols(n_emb), categorical


def tabpfn_version() -> str:
    """TABPFN_VERSION from the environment; default v3.5-fast with a token, else v2 (no login needed)."""
    default = "v3.5-fast" if os.environ.get("TABPFN_TOKEN") else "v2"
    return os.environ.get("TABPFN_VERSION", "").strip() or default


class FrameModel:
    """Tabular models that take raw categorical columns (no one-hot, no scaling).

    CatBoost, XGBoost and LightGBM use their native categorical support, TabPFN gets integer
    codes plus `categorical_features_indices`, TabICL gets the DataFrame itself.
    """

    def __init__(self, kind: str, seed: int, numeric: list[str], categorical: list[str], device: str = "cpu"):
        self.kind, self.seed, self.numeric, self.categorical = kind, seed, numeric, categorical
        self.device = device

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
                device=self.device,
                random_state=seed,
                categorical_features_indices=list(range(len(self.num_), len(self.num_) + len(self.cat_))),
            )
        else:
            from tabicl import TabICLClassifier

            self.clf_ = TabICLClassifier(random_state=seed, device=self.device)
        self.clf_.fit(self._prep(x), y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.clf_.predict_proba(self._prep(x))


def make_model(
    name: str, seed: int, use_group: bool = True, n_emb: int = 0, device: str = "cpu"
) -> Pipeline | FrameModel:
    name = name.removesuffix("+emb")  # "+emb" only adds the embedding columns (n_emb > 0)
    numeric, categorical = feature_cols(use_group, n_emb)
    if name in EXTRA_MODELS:
        return FrameModel(name, seed, numeric, categorical, device)
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


def model_matrix(df: pd.DataFrame, use_group: bool = True) -> pd.DataFrame:
    numeric, categorical = feature_cols(use_group)  # embedding columns are appended per fold
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


def emb_components(emb: tuple[np.ndarray, np.ndarray], n_rows: int) -> int:
    return max(1, min(EMB_COMPONENTS, emb[0].shape[1], n_rows // 2))


def with_emb(x: pd.DataFrame, z: np.ndarray) -> pd.DataFrame:
    return pd.concat([x, pd.DataFrame(z, index=x.index, columns=emb_cols(z.shape[1]))], axis=1)


def cross_validate(
    names: list[str],
    train: pd.DataFrame,
    y: np.ndarray,
    folds: int,
    seed: int,
    use_group: bool = True,
    test: pd.DataFrame | None = None,
    emb: tuple[np.ndarray, np.ndarray] | None = None,
    device: str = "cpu",
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """`test` (label-free) only contributes to group sizes, never to label statistics.

    `emb` = (train embeddings, test embeddings), rows aligned with `train` / `test`. For "+emb"
    models, PCA is fitted on the fit rows' embeddings only, inside each fold."""
    splits = fold_splits(y, folds, seed)
    oof = {name: np.zeros(len(y)) for name in names}
    wants_emb = any(n.endswith("+emb") for n in names)
    if wants_emb and emb is None:
        raise ValueError("'+emb' models need embeddings")
    for val_idx in splits:
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        others = [train.iloc[val_idx]] + ([test] if test is not None else [])
        x_fit, (x_val, *_) = fold_matrices(train.iloc[train_idx], others, use_group)
        n_emb = 0
        if wants_emb:
            n_emb = emb_components(emb, len(train_idx))
            pca = PCA(n_components=n_emb, random_state=seed).fit(emb[0][train_idx])
            x_fit_e = with_emb(x_fit, pca.transform(emb[0][train_idx]))
            x_val_e = with_emb(x_val, pca.transform(emb[0][val_idx]))
        for name in names:
            plus = name.endswith("+emb")
            model = make_model(name, seed, use_group, n_emb if plus else 0, device)
            model.fit(x_fit_e if plus else x_fit, y[train_idx])
            oof[name][val_idx] = model.predict_proba(x_val_e if plus else x_val)[:, 1]
    return oof, splits


def fold_splits(y: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    """Validation indices of each fold: the single source of the CV split for every path."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return [val for _, val in splitter.split(np.zeros(len(y)), y)]


def fit_predict_full(
    names: list[str],
    seed: int,
    train: pd.DataFrame,
    y: np.ndarray,
    test: pd.DataFrame,
    use_group: bool = True,
    emb: tuple[np.ndarray, np.ndarray] | None = None,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Final fit on all train rows (group statistics from all train labels, leave-one-out for train
    rows) and test probabilities per model."""
    x_train, (x_test,) = fold_matrices(train, [test], use_group)
    n_emb = 0
    if emb is not None:
        n_emb = emb_components(emb, len(y))
        pca = PCA(n_components=n_emb, random_state=seed).fit(emb[0])
        x_train_e, x_test_e = with_emb(x_train, pca.transform(emb[0])), with_emb(x_test, pca.transform(emb[1]))
    out = {}
    for name in names:
        plus = name.endswith("+emb")
        model = make_model(name, seed, use_group, n_emb if plus else 0, device)
        model.fit(x_train_e if plus else x_train, y)
        out[name] = model.predict_proba(x_test_e if plus else x_test)[:, 1]
    return out
