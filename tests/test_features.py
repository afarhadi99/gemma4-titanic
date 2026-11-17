from pathlib import Path

import pandas as pd
import pytest

from titanic.features import CATEGORICAL_COLS, NUMERIC_COLS, engineer, extract_title, load_raw

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("name", "title"),
    [
        ("Braund, Mr. Owen Harris", "Mr"),
        ("Cumings, Mrs. John Bradley (Florence Briggs Thayer)", "Mrs"),
        ("Heikkinen, Miss. Laina", "Miss"),
        ("Palsson, Master. Gosta Leonard", "Master"),
        ("Baxter, Mlle. Helene", "Miss"),
        ("Beesley, Ms. Lucy", "Miss"),
        ("Dupont, Mme. Marie", "Mrs"),
        ("Graham, Dr. William Thomas", "Rare"),
        ("Wheadon, Rev. Edward H", "Rare"),
        ("Weir, Col. John", "Rare"),
        ("No title here", "Rare"),
    ],
)
def test_extract_title(name: str, title: str) -> None:
    assert extract_title(name) == title


def test_engineered_columns_have_no_nans() -> None:
    train, test = engineer(*load_raw(FIXTURES))
    for frame in (train, test):
        assert not frame[NUMERIC_COLS + CATEGORICAL_COLS].isna().any().any()


def test_engineering_values() -> None:
    train, test = engineer(*load_raw(FIXTURES))
    assert set(train["Title"]) <= {"Mr", "Mrs", "Miss", "Master", "Rare"}
    assert train.loc[0, "FamilySize"] == 2
    assert train.loc[2, "IsAlone"] == 1
    assert train.loc[0, "Deck"] == "U"
    assert train.loc[1, "Deck"] == "C"
    assert bool(train.loc[5, "AgeMissing"]) is True
    assert train.loc[21, "Embarked"] == "S"
    assert test.loc[test["PassengerId"] == 900, "Fare"].iloc[0] > 0
    shared = test.loc[test["PassengerId"] == 898, "TicketGroupSize"].iloc[0]
    assert shared == 2
    assert "Survived" in train.columns
    assert "Survived" not in test.columns
    assert "Name" in train.columns


def test_inputs_are_not_mutated() -> None:
    train_raw, test_raw = load_raw(FIXTURES)
    train_copy, test_copy = train_raw.copy(), test_raw.copy()
    engineer(train_raw, test_raw)
    pd.testing.assert_frame_equal(train_raw, train_copy)
    pd.testing.assert_frame_equal(test_raw, test_copy)
