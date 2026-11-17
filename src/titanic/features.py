import re
from pathlib import Path

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
    return out


def engineer(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = [c for c in test.columns if c in train.columns and c != "Survived"]
    combined = pd.concat([train[shared], test[shared]], ignore_index=True)
    features = _engineer_combined(combined)
    n_train = len(train)

    def attach(original: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
        part = part.set_axis(original.index)
        return pd.concat([original.drop(columns=part.columns, errors="ignore"), part], axis=1)

    return attach(train, features.iloc[:n_train]), attach(test, features.iloc[n_train:])
