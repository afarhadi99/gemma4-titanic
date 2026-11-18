from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from titanic.features import (
    CATEGORICAL_COLS,
    GROUP_COLS,
    NUMERIC_COLS,
    add_group_features,
    engineer,
    extract_title,
    load_raw,
)

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


# --- group survival features -------------------------------------------------


def _group_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_raw, test_raw = load_raw(FIXTURES)
    train, test = engineer(train_raw, test_raw)
    fit = train.iloc[:16].reset_index(drop=True)
    val = train.iloc[16:].reset_index(drop=True)
    return fit, val, test


def _toy(names, tickets, fares, survived=None):
    df = pd.DataFrame({"Name": names, "Ticket": tickets, "Fare": fares})
    if survived is not None:
        df["Survived"] = survived
    return df


def test_group_stats_are_leave_one_out() -> None:
    fit = _toy(["A, Mr. x", "A, Mrs. y", "A, Miss. z", "B, Mr. q"], ["t1", "t1", "t1", "t2"], [10.0] * 4, [1, 0, 1, 1])
    out, _ = add_group_features(fit, [])
    assert out["GroupSize"].tolist() == [3, 3, 3, 1]
    assert out["GroupKnownCount"].tolist() == [2, 2, 2, 0]
    assert out["GroupKnownSurvived"].tolist() == [1, 2, 1, 0]
    assert out["GroupNoKnown"].tolist() == [0, 0, 0, 1]
    assert out["GroupSurvivalRate"].iloc[:3].tolist() == [0.5, 1.0, 0.5]
    assert np.isnan(out["GroupSurvivalRate"].iloc[3])


def test_own_label_never_affects_own_feature() -> None:
    fit, val, test = _group_frames()
    base, _ = add_group_features(fit, [val, test])
    for i in range(len(fit)):
        flipped = fit.copy()
        flipped.loc[i, "Survived"] = 1 - fit.loc[i, "Survived"]
        out, _ = add_group_features(flipped, [val, test])
        pd.testing.assert_series_equal(out.loc[i, GROUP_COLS], base.loc[i, GROUP_COLS])


def test_other_rows_depend_only_on_fit_labels() -> None:
    fit, val, test = _group_frames()
    base_fit, (base_val, base_test) = add_group_features(fit, [val, test])
    # Changing labels of the held-out rows must not change anything.
    val_flipped = val.assign(Survived=1 - val["Survived"])
    f2, (v2, t2) = add_group_features(fit, [val_flipped, test])
    for a, b in ((base_fit, f2), (base_val, v2), (base_test, t2)):
        pd.testing.assert_frame_equal(a[GROUP_COLS], b[GROUP_COLS])
    # Dropping the labels entirely from `others` also changes nothing.
    f3, (v3, _) = add_group_features(fit, [val.drop(columns="Survived"), test])
    pd.testing.assert_frame_equal(base_val[GROUP_COLS], v3[GROUP_COLS])
    # Flipping a fit label does change a held-out row that shares its group.
    toy_fit = _toy(["A, Mr. x", "A, Mrs. y"], ["t1", "t1"], [5.0, 5.0], [1, 1])
    toy_val = _toy(["A, Miss. z"], ["t1"], [5.0])
    _, (v_a,) = add_group_features(toy_fit, [toy_val])
    _, (v_b,) = add_group_features(toy_fit.assign(Survived=[0, 0]), [toy_val])
    assert v_a["GroupKnownSurvived"].iloc[0] == 2 and v_b["GroupKnownSurvived"].iloc[0] == 0
    assert v_a["GroupKnownCount"].iloc[0] == 2


def test_surname_fare_fallback_and_inputs_not_mutated() -> None:
    fit = _toy(["Smith, Mr. a", "Smith, Mrs. b"], ["t1", "t2"], [30.5, 30.5], [1, 0])
    other = _toy(["Smith, Miss. c"], ["t3"], [30.5])
    fit_copy, other_copy = fit.copy(), other.copy()
    out, (o,) = add_group_features(fit, [other])
    assert o["GroupSize"].iloc[0] == 3 and o["GroupKnownCount"].iloc[0] == 2
    assert out["GroupKnownCount"].tolist() == [1, 1]
    pd.testing.assert_frame_equal(fit, fit_copy)
    pd.testing.assert_frame_equal(other, other_copy)


def test_empty_others_frame_is_fine() -> None:
    fit, val, _ = _group_frames()
    _, (empty, test_out) = add_group_features(fit, [val.iloc[0:0], val])
    assert len(empty) == 0 and len(test_out) == len(val)
