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


# --- optional feature groups and the women-and-children group (WCG) rule ------


def test_optional_feature_columns_are_complete_and_sane() -> None:
    from titanic.features import FEATURE_GROUPS

    train, test = engineer(*load_raw(FIXTURES))
    for frame in (train, test):
        for numeric, categorical in FEATURE_GROUPS.values():
            cols = [c for c in numeric + categorical if c not in GROUP_COLS]
            assert not frame[cols].isna().any().any()
    assert (train["FareRel"] > 0).all()
    assert set(train["CabinSide"]) <= {"odd", "even", "none"}
    assert train.loc[1, "DeckPclass"].startswith("C_")
    assert train.loc[1, "TitlePclass"] == f"{train.loc[1, 'Title']}_{train.loc[1, 'Pclass']}"


def test_ticket_and_cabin_parsing() -> None:
    from titanic.features import _ticket_parts

    assert _ticket_parts("A/5 21171") == ("A", "d5")
    assert _ticket_parts("PC 17599") == ("PC", "d5")
    assert _ticket_parts("STON/O2. 3101282") == ("STONO", "d7")
    assert _ticket_parts("113803") == ("NUM", "d6")
    assert _ticket_parts("LINE") == ("LINE", "none")
    raw = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Pclass": [1, 1, 3],
            "Name": ["A, Mr. x", "B, Mrs. y", "C, Miss. z"],
            "Sex": ["male", "female", "female"],
            "Age": [30.0, 25.0, 20.0],
            "SibSp": [0, 0, 0],
            "Parch": [0, 0, 0],
            "Ticket": ["1", "2", "3"],
            "Fare": [10.0, 20.0, 5.0],
            "Cabin": ["C85", "B57 B59 B63", None],
            "Embarked": ["S", "C", "Q"],
        }
    )
    out, _ = engineer(raw.assign(Survived=[0, 1, 1]), raw)
    assert out["CabinNumber"].tolist() == [85.0, 57.0, 0.0]
    assert out["CabinCount"].tolist() == [1, 3, 0]
    assert out["CabinSide"].tolist() == ["odd", "odd", "none"]


def _wcg_toy(survived):
    names = ["A, Mr. a", "A, Mrs. b", "A, Master. c", "A, Mr. d", "Z, Mr. z"]
    return _toy(names, ["t1", "t1", "t1", "t1", "t2"], [10.0] * 5, survived)


def test_wcg_all_died_and_any_survived_are_leave_one_out() -> None:
    fit = _wcg_toy([0, 0, 0, 1, 1])
    out, _ = add_group_features(fit, [])
    # Mr. a sees the mother and the boy (both died); the mother sees only the boy; Z is alone.
    assert out["GroupWcKnown"].tolist() == [2, 1, 1, 2, 0]
    assert out["WcgAllDied"].tolist() == [1, 1, 1, 1, 0]
    assert out["WcgAnySurvived"].tolist() == [0, 0, 0, 0, 0]
    fit2 = _wcg_toy([0, 1, 0, 1, 1])  # the mother survived
    out2, _ = add_group_features(fit2, [])
    assert out2["WcgAnySurvived"].tolist() == [1, 0, 1, 1, 0]  # she does not see her own label
    assert out2["WcgAllDied"].tolist() == [0, 1, 0, 0, 0]


def test_wcg_for_unseen_rows_uses_fit_labels_only() -> None:
    fit = _wcg_toy([0, 1, 0, 1, 1])
    val = _toy(["A, Mr. e"], ["t1"], [10.0])
    _, (v,) = add_group_features(fit, [val])
    assert v["GroupWcKnown"].tolist() == [2] and v["WcgAnySurvived"].tolist() == [1]
    _, (v2,) = add_group_features(fit.assign(Survived=[0, 0, 0, 1, 1]), [val])
    assert v2["WcgAllDied"].tolist() == [1] and v2["WcgAnySurvived"].tolist() == [0]
