import re
from pathlib import Path

import numpy as np
import pandas as pd

TITLE_PATTERN = re.compile(r",\s*([^.]+)\.")
TITLE_ALIASES = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
COMMON_TITLES = frozenset({"Mr", "Mrs", "Miss", "Master"})

NUMERIC_COLS = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "TicketGroupSize",
    "FarePerPerson",
    "AgeMissing",
]
CATEGORICAL_COLS = ["Sex", "Embarked", "Title", "Deck"]

# Optional engineered columns, grouped so each group can be switched on or off (see baseline.py).
# group name -> (numeric columns, categorical columns). All are computed by engineer() from
# label-free inputs, except "wcg", which add_group_features() derives from fit labels only.
FEATURE_GROUPS: dict[str, tuple[list[str], list[str]]] = {
    "ticket": ([], ["TicketPrefix", "TicketNumBucket"]),
    "cabin": (["CabinNumber", "CabinCount"], ["CabinSide"]),
    "deckpclass": ([], ["DeckPclass"]),
    "farerel": (["FareRel"], []),
    "titlepclass": ([], ["TitlePclass"]),
    "wcg": (["WcgAllDied", "WcgAnySurvived"], []),
}
MIN_PREFIX_COUNT = 8  # rarer ticket prefixes collapse into "OTHER"


def load_raw(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "test.csv")


def extract_title(name: str) -> str:
    match = TITLE_PATTERN.search(name)
    if match is None:
        return "Rare"
    raw = match.group(1).strip()
    title = TITLE_ALIASES.get(raw, raw)
    return title if title in COMMON_TITLES else "Rare"


def _fill_with_group_median(values: pd.Series, groups: pd.Series) -> pd.Series:
    group_median = values.groupby(groups).transform("median")
    return values.fillna(group_median).fillna(values.median())


def _engineer_combined(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Title"] = df["Name"].fillna("").map(extract_title)
    out["AgeMissing"] = df["Age"].isna()
    out["Age"] = _fill_with_group_median(df["Age"].astype(float), out["Title"])
    out["Fare"] = _fill_with_group_median(df["Fare"].astype(float), df["Pclass"])
    embarked_mode = df["Embarked"].mode()
    out["Embarked"] = df["Embarked"].fillna(embarked_mode.iloc[0] if len(embarked_mode) else "S")
    out["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["Deck"] = df["Cabin"].fillna("U").str[0].str.upper()
    out["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("size")
    out["FarePerPerson"] = out["Fare"] / out["TicketGroupSize"]
    _add_optional_columns(df, out)
    return out


def _ticket_parts(ticket: str) -> tuple[str, str]:
    """(normalised letter prefix or "NUM", digit-count bucket of the trailing number)."""
    tokens = ticket.strip().split()
    number = tokens[-1] if tokens and tokens[-1].isdigit() else ""
    head = " ".join(tokens[:-1]) if number else " ".join(tokens)
    prefix = re.sub(r"[^A-Za-z]", "", head).upper() or "NUM"
    return prefix, f"d{len(number)}" if number else "none"


def _add_optional_columns(df: pd.DataFrame, out: pd.DataFrame) -> None:
    parts = df["Ticket"].fillna("").astype(str).map(_ticket_parts)
    prefix = parts.str[0]
    rare = prefix.map(prefix.value_counts()) < MIN_PREFIX_COUNT
    out["TicketPrefix"] = prefix.where(~rare, "OTHER")
    out["TicketNumBucket"] = parts.str[1]

    cabin = df["Cabin"].fillna("").astype(str).str.strip()
    out["CabinCount"] = cabin.map(lambda c: len(c.split()) if c else 0)
    number = cabin.str.extract(r"(\d+)", expand=False)
    out["CabinNumber"] = pd.to_numeric(number, errors="coerce").fillna(0).astype(float)
    side = np.where(number.isna(), "none", np.where(out["CabinNumber"] % 2 == 1, "odd", "even"))
    out["CabinSide"] = side

    out["DeckPclass"] = out["Deck"] + "_" + df["Pclass"].astype(int).astype(str)
    class_median = out.groupby(df["Pclass"])["FarePerPerson"].transform("median")
    out["FareRel"] = out["FarePerPerson"] / class_median.where(class_median > 0, 1.0)
    out["TitlePclass"] = out["Title"] + "_" + df["Pclass"].astype(int).astype(str)


def engineer(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = [c for c in test.columns if c in train.columns and c != "Survived"]
    combined = pd.concat([train[shared], test[shared]], ignore_index=True)
    features = _engineer_combined(combined)
    n_train = len(train)

    def attach(original: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
        part = part.set_axis(original.index)
        return pd.concat([original.drop(columns=part.columns, errors="ignore"), part], axis=1)

    return attach(train, features.iloc[:n_train]), attach(test, features.iloc[n_train:])


GROUP_COLS = [
    "GroupSize",
    "GroupKnownCount",
    "GroupKnownSurvived",
    "GroupSurvivalRate",
    "GroupNoKnown",
    "GroupWcKnown",
    "GroupWcSurvived",
    "WcgAllDied",
    "WcgAnySurvived",
]
WC_TITLES = frozenset({"Mrs", "Miss", "Master"})  # women and boys


def _group_keys(df: pd.DataFrame) -> pd.Series:
    """Label-free group id per row: the Ticket if shared, else Surname+Fare if shared, else unique."""
    ticket = df["Ticket"].astype(str)
    surname = df["Name"].fillna("").str.split(",").str[0].str.strip().str.lower()
    fare = df["Fare"].round(2).astype(str)
    family = surname + "|" + fare
    key = "t:" + ticket
    ticket_alone = ticket.groupby(ticket).transform("size") == 1
    family_shared = (family.groupby(family).transform("size") > 1) & (df["Fare"] > 0) & (surname != "")
    key = key.where(~(ticket_alone & family_shared), "f:" + family)
    return key


def add_group_features(
    fit: pd.DataFrame, others: list[pd.DataFrame]
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Ticket / family group survival features without label leakage.

    Groups are formed over fit + others (label-free). Survival statistics use only the
    labels in `fit`: a fit row sees its group mates' labels but never its own
    (leave-one-out); rows in `others` see every fit label in their group. Inputs are not
    mutated. `fit` must hold `Survived`; `others` may or may not.
    """
    frames = [fit, *others]
    combined = pd.concat([f[["Ticket", "Name", "Fare"]] for f in frames], ignore_index=True)
    keys = _group_keys(combined)
    n_fit = len(fit)
    size = keys.groupby(keys).transform("size").to_numpy()

    labels = fit["Survived"].to_numpy(dtype=float)
    fit_keys = keys.iloc[:n_fit]
    known = pd.Series(1.0, index=fit_keys.index).groupby(fit_keys.to_numpy()).sum()
    surv = pd.Series(labels, index=fit_keys.index).groupby(fit_keys.to_numpy()).sum()
    count = keys.map(known).fillna(0.0).to_numpy().copy()
    wins = keys.map(surv).fillna(0.0).to_numpy().copy()
    count[:n_fit] -= 1.0  # leave one out: a fit row never sees its own label
    wins[:n_fit] -= labels

    stats = pd.DataFrame(
        {
            "GroupSize": size.astype(int),
            "GroupKnownCount": count.astype(int),
            "GroupKnownSurvived": wins.astype(int),
        }
    )
    stats["GroupSurvivalRate"] = np.where(count > 0, wins / np.maximum(count, 1), np.nan)
    stats["GroupNoKnown"] = (count == 0).astype(int)

    # Women-and-children group (WCG) rule: among the other known women/boys of the group, did all
    # die (WcgAllDied) or did any survive (WcgAnySurvived)? Same leave-one-out treatment.
    is_wc = combined["Name"].fillna("").map(extract_title).isin(WC_TITLES).to_numpy(dtype=float)
    wc_labels = is_wc[:n_fit] * labels
    wc_known = pd.Series(is_wc[:n_fit], index=fit_keys.index).groupby(fit_keys.to_numpy()).sum()
    wc_surv = pd.Series(wc_labels, index=fit_keys.index).groupby(fit_keys.to_numpy()).sum()
    wc_count = keys.map(wc_known).fillna(0.0).to_numpy().copy()
    wc_wins = keys.map(wc_surv).fillna(0.0).to_numpy().copy()
    wc_count[:n_fit] -= is_wc[:n_fit]
    wc_wins[:n_fit] -= wc_labels
    stats["GroupWcKnown"] = wc_count.astype(int)
    stats["GroupWcSurvived"] = wc_wins.astype(int)
    stats["WcgAllDied"] = ((wc_count > 0) & (wc_wins == 0)).astype(int)
    stats["WcgAnySurvived"] = (wc_wins > 0).astype(int)

    outputs: list[pd.DataFrame] = []
    start = 0
    for frame in frames:
        part = stats.iloc[start : start + len(frame)].set_axis(frame.index)
        start += len(frame)
        outputs.append(pd.concat([frame.drop(columns=GROUP_COLS, errors="ignore"), part], axis=1))
    return outputs[0], outputs[1:]
