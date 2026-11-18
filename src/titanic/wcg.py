"""Woman-Child-Group (WCG) rule model: the gender rule, adjusted by the fate of one's group.

Train and test passengers are pooled (labels exist only for train) and linked into groups by shared
ticket / family keys. Women live and boys and men die, unless every known woman-or-boy group-mate
went the other way. A passenger's prediction never uses their own label, and groups are built from
label-free columns only, so honest cross-validation is possible: hide the validation labels and
predict from the fit labels alone. All functions here are pure.
"""

import numpy as np
import pandas as pd

from titanic.features import extract_title
from titanic.tabular import fold_splits

DEFAULT, DIE, LIVE = "default", "overridden-die", "overridden-live"
VARIANTS = ["gender", "wcg", "h1", "h2", "h3"]
STRONG_LIVE, STRONG_DIE = 0.8, 0.2


def pool_passengers(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """All passengers, train first; Survived is NaN for test rows."""
    cols = ["PassengerId", "Pclass", "Name", "Sex", "Ticket", "Fare", "Embarked"]
    pooled = pd.concat([train, test], ignore_index=True)
    out = pooled[cols].copy()
    out["Survived"] = pooled["Survived"].astype(float) if "Survived" in pooled else np.nan
    out.loc[len(train) :, "Survived"] = np.nan
    return out


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def link_keys(keys: list, uf: UnionFind) -> None:
    """Union every pair of rows that share a (non-None) key."""
    first: dict = {}
    for i, key in enumerate(keys):
        if key is None:
            continue
        if key in first:
            uf.union(first[key], i)
        else:
            first[key] = i


def surnames(names: pd.Series) -> pd.Series:
    return names.fillna("").astype(str).str.split(",").str[0].str.strip()


def build_groups(pool: pd.DataFrame, nearby: bool = False) -> np.ndarray:
    """Group id per row: connected components over ticket, family and (optionally) nearby-ticket keys."""
    ticket = pool["Ticket"].fillna("").astype(str).str.strip()
    surname = surnames(pool["Name"])
    pclass = pool["Pclass"].astype(str)
    embarked = pool["Embarked"].fillna("?").astype(str)
    fare = pool["Fare"].round(2).astype(str)
    stem = ticket.str[:-1]
    uf = UnionFind(len(pool))
    link_keys([t or None for t in ticket], uf)
    family = surname + "|" + pclass + "|" + stem + "|" + fare + "|" + embarked
    link_keys([None if not s or not t else f for s, t, f in zip(surname, stem, family)], uf)
    if nearby:
        near = surname + "|" + pclass + "|" + embarked + "|" + stem
        link_keys([None if not s or not t else f for s, t, f in zip(surname, stem, near)], uf)
    return np.array([uf.find(i) for i in range(len(pool))])


def passenger_kinds(pool: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(is_woman, is_boy) boolean arrays. Boy = male with title Master."""
    woman = (pool["Sex"] == "female").to_numpy()
    master = pool["Name"].fillna("").map(extract_title) == "Master"
    boy = ((pool["Sex"] == "male") & master).to_numpy()
    return woman, boy


def gender_default(pool: pd.DataFrame) -> np.ndarray:
    return (pool["Sex"] == "female").to_numpy().astype(int)


def apply_rules(pool: pd.DataFrame, groups: np.ndarray, known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WCG predictions and reason codes for every pooled row.

    `known` holds the labels usable for prediction (NaN = hidden). A row's evidence is the known
    labels of the OTHER women/boys in its group, so its own label is excluded by construction.
    """
    woman, boy = passenger_kinds(pool)
    wcg = woman | boy
    known = np.asarray(known, dtype=float)
    usable = wcg & ~np.isnan(known)
    lab = np.where(usable, known, 0.0)
    n_groups = int(groups.max()) + 1
    mates = np.bincount(groups, weights=usable.astype(float), minlength=n_groups)[groups] - usable
    alive = np.bincount(groups, weights=lab, minlength=n_groups)[groups] - lab

    pred = gender_default(pool)
    reason = np.full(len(pool), DEFAULT, dtype=object)
    die = woman & (mates > 0) & (alive == 0)
    live = boy & (mates > 0) & (alive == mates)
    pred[die], reason[die] = 0, DIE
    pred[live], reason[live] = 1, LIVE
    return pred, reason


def combine(
    variant: str,
    pool: pd.DataFrame,
    rule_pred: np.ndarray,
    reason: np.ndarray,
    prob: np.ndarray | None = None,
    strong: tuple[float, float] = (STRONG_DIE, STRONG_LIVE),
) -> np.ndarray:
    """Final 0/1 predictions for one variant. `prob` (model survival probability) is needed for h2/h3."""
    default = gender_default(pool)
    overridden = reason != DEFAULT
    if variant == "gender":
        return default
    if variant in ("wcg", "h1"):  # the rules already are gender default + overrides
        return np.where(overridden, rule_pred, default)
    if prob is None:
        raise ValueError(f"variant {variant} needs model probabilities")
    if variant == "h2":
        return np.where(overridden, rule_pred, (prob >= 0.5).astype(int))
    if variant == "h3":
        woman, boy = passenger_kinds(pool)
        pred = np.where(overridden, rule_pred, default)
        adult_male = ~woman & ~boy
        pred = np.where(adult_male & ~overridden & (prob >= strong[1]), 1, pred)
        return np.where(woman & ~overridden & (prob <= strong[0]), 0, pred)
    raise ValueError(f"unknown variant {variant}")


def honest_cv(
    pool: pd.DataFrame,
    groups: np.ndarray,
    n_train: int,
    seeds: tuple[int, ...] = (42, 7, 123),
    folds: int = 5,
    oof: np.ndarray | None = None,
    oof_seed: int = 42,
) -> dict:
    """Repeated stratified CV of every variant. Groups use all pooled rows (label-free); in each fold
    the validation labels are hidden. h2/h3 (need `oof`, aligned to train rows) run on `oof_seed` only.

    Returns {variant: {"accuracy_by_seed", "mean", "overrides", "override_correct"}}; override counts
    (WCG rule only) are summed over seeds.
    """
    y = pool["Survived"].to_numpy()[:n_train].astype(int)
    labels = pool["Survived"].to_numpy(dtype=float)
    train_pool = pool.iloc[:n_train]
    result: dict = {v: {"accuracy_by_seed": {}, "overrides": 0, "override_correct": 0} for v in VARIANTS}
    hits: dict = {}
    for seed in seeds:
        for val in fold_splits(y, folds, seed):
            known = labels.copy()
            known[val] = np.nan
            known[n_train:] = np.nan
            rule_pred, reason = apply_rules(pool, groups, known)
            for v in VARIANTS:
                if v in ("h2", "h3") and (oof is None or seed != oof_seed):
                    continue
                prob = None if oof is None else oof[val]
                pred = combine(v, train_pool.iloc[val], rule_pred[val], reason[val], prob)
                cell = hits.setdefault((v, seed), [0, 0])
                cell[0] += int((pred == y[val]).sum())
                cell[1] += len(val)
                if v == "wcg":
                    ov = reason[val] != DEFAULT
                    result[v]["overrides"] += int(ov.sum())
                    result[v]["override_correct"] += int((rule_pred[val][ov] == y[val][ov]).sum())
    for (v, seed), (hit, total) in hits.items():
        result[v]["accuracy_by_seed"][seed] = hit / total
    for v in VARIANTS:
        accs = list(result[v]["accuracy_by_seed"].values())
        result[v]["mean"] = float(np.mean(accs)) if accs else None
    return result


def predict_test(pool: pd.DataFrame, groups: np.ndarray, n_train: int, prob: np.ndarray | None = None) -> dict:
    """Test predictions of every variant (all train labels known) plus change counts."""
    rule_pred, reason = apply_rules(pool, groups, pool["Survived"].to_numpy(dtype=float))
    test = pool.iloc[n_train:]
    rp, rs = rule_pred[n_train:], reason[n_train:]
    preds = {v: combine(v, test, rp, rs, prob) for v in VARIANTS if prob is not None or v not in ("h2", "h3")}
    default = gender_default(test)
    counts = {
        "wcg_overrides": int((rs != DEFAULT).sum()),
        "wcg_overridden_die": int((rs == DIE).sum()),
        "wcg_overridden_live": int((rs == LIVE).sum()),
    }
    for v, p in preds.items():
        counts[f"{v}_vs_gender"] = int((p != default).sum())
        if prob is not None:
            counts[f"{v}_vs_model"] = int((p != (prob >= 0.5).astype(int)).sum())
    return {"preds": preds, "counts": counts}
