"""Woman-Child-Group (WCG) rule model: the gender rule, adjusted by the fate of one's group.

Train and test passengers are pooled (labels exist only for train) and linked into groups by shared
ticket / family keys. Women live and boys and men die, unless every known woman-or-boy group-mate
went the other way. A passenger's prediction never uses their own label, and groups are built from
label-free columns only, so honest cross-validation is possible: hide the validation labels and
predict from the fit labels alone. All functions here are pure.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from titanic.features import extract_title
from titanic.tabular import fold_splits

DEFAULT, DIE, LIVE = "default", "overridden-die", "overridden-live"
CLASS3 = "overridden-die-class3"  # large 3rd-class family with no known outcomes
KEYS = ("a", "b", "c")
RATE_DIE, RATE_LIVE, RATE_MIN_KNOWN = 0.25, 0.75, 2
LARGE_GROUP = 5
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


def build_groups(pool: pd.DataFrame, nearby: bool = False, keys: str = "a") -> np.ndarray:
    """Group id per row: connected components (union-find) over ticket plus family-style keys.

    keys "a": ticket + Surname|Pclass|Ticket[:-1]|Fare|Embarked (needs a non-empty stem); `nearby`
        adds Surname|Pclass|Embarked|stem links (the current model).
    keys "b": Deotte "name only" style: the same family key without the stem requirement; ignores `nearby`.
    keys "c": ticket + Surname|Pclass|Embarked only; ignores `nearby`.
    """
    if keys not in KEYS:
        raise ValueError(f"unknown keys {keys!r}; choose from {KEYS}")
    ticket = pool["Ticket"].fillna("").astype(str).str.strip()
    surname = surnames(pool["Name"])
    pclass = pool["Pclass"].astype(str)
    embarked = pool["Embarked"].fillna("?").astype(str)
    fare = pool["Fare"].round(2).astype(str)
    stem = ticket.str[:-1]
    uf = UnionFind(len(pool))
    link_keys([t or None for t in ticket], uf)
    if keys == "c":
        fam = surname + "|" + pclass + "|" + embarked
        link_keys([None if not s else f for s, f in zip(surname, fam)], uf)
    else:
        family = surname + "|" + pclass + "|" + stem + "|" + fare + "|" + embarked
        need_stem = keys == "a"
        link_keys([None if not s or (need_stem and not t) else f for s, t, f in zip(surname, stem, family)], uf)
        if nearby and keys == "a":
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


def known_mates(pool: pd.DataFrame, groups: np.ndarray, known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(count, survivors) of the OTHER known women/boys in each row's group (own label excluded)."""
    woman, boy = passenger_kinds(pool)
    known = np.asarray(known, dtype=float)
    usable = (woman | boy) & ~np.isnan(known)
    lab = np.where(usable, known, 0.0)
    n_groups = int(groups.max()) + 1
    mates = np.bincount(groups, weights=usable.astype(float), minlength=n_groups)[groups] - usable
    alive = np.bincount(groups, weights=lab, minlength=n_groups)[groups] - lab
    return mates, alive


def apply_rules(
    pool: pd.DataFrame,
    groups: np.ndarray,
    known: np.ndarray,
    woman_rate: bool = False,
    boy_rate: bool = False,
    class3_large: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """WCG predictions and reason codes for every pooled row.

    `known` holds the labels usable for prediction (NaN = hidden). A row's evidence is the known
    labels of the OTHER women/boys in its group, so its own label is excluded by construction.
    Default: a woman dies only if every known mate died, a boy lives only if every known mate lived.
    woman_rate / boy_rate: use the mate survival rate (<= 0.25 / >= 0.75) with at least 2 known mates.
    class3_large: a 3rd-class woman/boy in a group of >= 5 with no known mates defaults to die.
    """
    woman, boy = passenger_kinds(pool)
    mates, alive = known_mates(pool, groups, known)
    rate = np.divide(alive, mates, out=np.zeros(len(pool)), where=mates > 0)

    pred = gender_default(pool)
    reason = np.full(len(pool), DEFAULT, dtype=object)
    if woman_rate:
        die = woman & (mates >= RATE_MIN_KNOWN) & (rate <= RATE_DIE)
    else:
        die = woman & (mates > 0) & (alive == 0)
    if boy_rate:
        live = boy & (mates >= RATE_MIN_KNOWN) & (rate >= RATE_LIVE)
    else:
        live = boy & (mates > 0) & (alive == mates)
    pred[die], reason[die] = 0, DIE
    pred[live], reason[live] = 1, LIVE
    if class3_large:
        size = np.bincount(groups)[groups]
        c3 = (woman | boy) & (pool["Pclass"].to_numpy() == 3) & (size >= LARGE_GROUP) & (mates == 0)
        pred[c3], reason[c3] = 0, CLASS3
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
    opts: dict | None = None,
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
            rule_pred, reason = apply_rules(pool, groups, known, **(opts or {}))
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


def predict_test(
    pool: pd.DataFrame, groups: np.ndarray, n_train: int, prob: np.ndarray | None = None, opts: dict | None = None
) -> dict:
    """Test predictions of every variant (all train labels known) plus change counts."""
    rule_pred, reason = apply_rules(pool, groups, pool["Survived"].to_numpy(dtype=float), **(opts or {}))
    test = pool.iloc[n_train:]
    rp, rs = rule_pred[n_train:], reason[n_train:]
    preds = {v: combine(v, test, rp, rs, prob) for v in VARIANTS if prob is not None or v not in ("h2", "h3")}
    default = gender_default(test)
    counts = {
        "wcg_overrides": int((rs != DEFAULT).sum()),
        "wcg_overridden_die": int((rs == DIE).sum()),
        "wcg_overridden_live": int((rs == LIVE).sum()),
        "class3_large_die": int((rs == CLASS3).sum()),
    }
    for v, p in preds.items():
        counts[f"{v}_vs_gender"] = int((p != default).sum())
        if prob is not None:
            counts[f"{v}_vs_model"] = int((p != (prob >= 0.5).astype(int)).sum())
    return {"preds": preds, "counts": counts}


@dataclass(frozen=True)
class Config:
    """One WCG variant. model: "none" (rules only), "h3" (flip any rule-default row when confident)
    or "seg" (flip only plain-default rows of three segments, see `segment_flips`)."""

    keys: str = "a"
    nearby: bool = False
    woman_rate: bool = False
    boy_rate: bool = False
    class3_large: bool = False
    model: str = "none"
    hi: float = STRONG_LIVE
    lo: float = STRONG_DIE

    @property
    def name(self) -> str:
        parts = [self.keys + ("-near" if self.nearby and self.keys == "a" else "")]
        parts += [t for t, on in (("wr", self.woman_rate), ("br", self.boy_rate), ("c3", self.class3_large)) if on]
        if self.model != "none":
            parts.append(self.model)
        return "+".join(parts)

    @property
    def rule_opts(self) -> dict:
        return {"woman_rate": self.woman_rate, "boy_rate": self.boy_rate, "class3_large": self.class3_large}


def segment_flips(
    pool: pd.DataFrame,
    groups: np.ndarray,
    known: np.ndarray,
    reason: np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray,
    hi: float,
    lo: float,
) -> np.ndarray:
    """Model flips only for plain-default rows (rule reason DEFAULT and no known WCG mates) in:
    adult men in 1st class (live if p >= hi), boys (live if p >= hi), 3rd-class women alone (die if p <= lo).
    NaN probabilities never flip."""
    woman, boy = passenger_kinds(pool)
    mates, _ = known_mates(pool, groups, known)
    plain = (reason == DEFAULT) & (mates == 0)
    pclass = pool["Pclass"].to_numpy()
    size = np.bincount(groups)[groups]
    out = pred.copy()
    with np.errstate(invalid="ignore"):
        live = plain & (prob >= hi) & ((~woman & ~boy & (pclass == 1)) | boy)
        die = plain & (prob <= lo) & woman & (pclass == 3) & (size == 1)
    out[live], out[die] = 1, 0
    return out


def predict_config(
    pool: pd.DataFrame, groups: np.ndarray, known: np.ndarray, cfg: Config, prob: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """(predictions, rule reasons) for every pooled row. `prob` is aligned to the pool (NaN = none)."""
    rule_pred, reason = apply_rules(pool, groups, known, **cfg.rule_opts)
    if cfg.model == "none":
        return rule_pred, reason
    if prob is None:
        raise ValueError(f"config {cfg.name} needs model probabilities")
    if cfg.model == "h3":
        return combine("h3", pool, rule_pred, reason, np.nan_to_num(prob, nan=0.5), (cfg.lo, cfg.hi)), reason
    if cfg.model == "seg":
        return segment_flips(pool, groups, known, reason, rule_pred, prob, cfg.hi, cfg.lo), reason
    raise ValueError(f"unknown model mode {cfg.model}")


def cv_config(
    pool: pd.DataFrame,
    groups: np.ndarray,
    n_train: int,
    cfg: Config,
    oof: np.ndarray | None,
    seeds: tuple[int, ...] = (42, 7, 123),
    folds: int = 5,
    oof_seed: int = 42,
) -> dict[int, float]:
    """Honest repeated CV accuracy per seed (val labels hidden, groups from fit labels only).
    Model configs run on `oof_seed` only: the OOF probabilities come from that one split."""
    y = pool["Survived"].to_numpy()[:n_train].astype(int)
    labels = pool["Survived"].to_numpy(dtype=float)
    prob = None
    if cfg.model != "none":
        if oof is None:
            raise ValueError(f"config {cfg.name} needs an OOF")
        prob = np.concatenate([oof, np.full(len(pool) - n_train, np.nan)])
        seeds = tuple(s for s in seeds if s == oof_seed)
    out: dict[int, float] = {}
    for seed in seeds:
        hit = 0
        for val in fold_splits(y, folds, seed):
            known = labels.copy()
            known[val] = np.nan
            known[n_train:] = np.nan
            pred, _ = predict_config(pool, groups, known, cfg, prob)
            hit += int((pred[val] == y[val]).sum())
        out[seed] = hit / n_train
    return out


def grid_configs(with_model: bool) -> list[Config]:
    """The pre-declared grid (at most 23 variants); nothing is tuned beyond it."""
    rules = [
        Config("a", True),
        Config("a"),
        Config("b"),
        Config("c"),
        Config("a", True, woman_rate=True),
        Config("a", True, boy_rate=True),
        Config("a", True, woman_rate=True, boy_rate=True),
        Config("a", True, class3_large=True),
        Config("a", True, woman_rate=True, boy_rate=True, class3_large=True),
        Config("b", class3_large=True),
        Config("c", class3_large=True),
    ]
    out = list(rules)
    if with_model:
        bases = [
            Config("a", True),
            Config("a", True, class3_large=True),
            Config("a", True, woman_rate=True, boy_rate=True),
            Config("a", True, woman_rate=True, boy_rate=True, class3_large=True),
            Config("b"),
            Config("c"),
        ]
        for b in bases:
            for mode in ("h3", "seg"):
                out.append(Config(b.keys, b.nearby, b.woman_rate, b.boy_rate, b.class3_large, mode))
    return out


def reference_config(with_model: bool) -> Config:
    """Config of the current best submission: nearby keys, plus h3 flips when a model is available."""
    return Config("a", True, model="h3" if with_model else "none")


def evaluate_grid(
    pool: pd.DataFrame,
    n_train: int,
    oof: np.ndarray | None,
    test_prob: np.ndarray | None,
    reference: np.ndarray | None,
    seeds: tuple[int, ...] = (42, 7, 123),
    folds: int = 5,
    min_diff: int = 2,
    max_diff: int = 15,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate the grid. Returns (summary sorted by CV mean, {name: test predictions}).

    `reference` = the reference submission's test predictions (aligned to test rows) or None. A variant is a
    "candidate" if its CV mean >= the reference config's mean over the SAME seeds (model variants only have
    seed 42, so they are compared on seed 42) and it differs from `reference` on min_diff..max_diff passengers.
    """
    with_model = oof is not None and test_prob is not None
    cfgs = grid_configs(with_model)
    ref_cfg = reference_config(with_model)
    pure = Config("a", True)
    pool_known = pool["Survived"].to_numpy(dtype=float)
    test_p = np.concatenate([np.full(n_train, np.nan), test_prob]) if with_model else None
    group_cache: dict = {}
    accs: dict[str, dict[int, float]] = {}
    preds: dict[str, np.ndarray] = {}
    n3: dict[str, int] = {}
    for cfg in cfgs:
        gk = (cfg.keys, cfg.nearby)
        if gk not in group_cache:
            group_cache[gk] = build_groups(pool, nearby=cfg.nearby, keys=cfg.keys)
        groups = group_cache[gk]
        accs[cfg.name] = cv_config(pool, groups, n_train, cfg, oof, seeds, folds)
        pred, reason = predict_config(pool, groups, pool_known, cfg, test_p)
        preds[cfg.name] = pred[n_train:]
        n3[cfg.name] = int((reason[n_train:] == CLASS3).sum())
    ref_accs = accs[ref_cfg.name]
    rows = []
    for cfg in cfgs:
        a = accs[cfg.name]
        common = [s for s in a if s in ref_accs]
        cmp_mean = float(np.mean([a[s] for s in common]))
        ref_mean = float(np.mean([ref_accs[s] for s in common]))
        n_ref = None if reference is None else int((preds[cfg.name] != reference).sum())
        cand = n_ref is not None and cmp_mean >= ref_mean and min_diff <= n_ref <= max_diff
        vals = list(a.values())
        rows.append(
            {
                "name": cfg.name,
                "cv_mean": float(np.mean(vals)),
                "cv_std": float(np.std(vals)),
                "cv_seed42": a.get(42, float("nan")),
                "n_diff_reference": float("nan") if n_ref is None else n_ref,
                "n_diff_wcg": int((preds[cfg.name] != preds[pure.name]).sum()),
                "n_seeds": len(vals),
                "n_class3_changes": n3[cfg.name],
                "candidate": "candidate" if cand else "",
            }
        )
    df = pd.DataFrame(rows).sort_values("cv_mean", ascending=False, kind="stable").reset_index(drop=True)
    return df, preds
