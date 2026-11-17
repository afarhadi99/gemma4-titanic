from pathlib import Path

import pandas as pd

from titanic.features import engineer, load_raw
from titanic.text import LABEL_WORDS, QUESTION, SYSTEM_PROMPT, describe

FIXTURES = Path(__file__).parent / "fixtures"


def engineered() -> tuple[pd.DataFrame, pd.DataFrame]:
    return engineer(*load_raw(FIXTURES))


def test_first_row_matches_plan_example() -> None:
    train, _ = engineered()
    assert describe(train.iloc[0]) == (
        "A 22-year-old man (title Mr) travelling in 3rd class, embarked at Southampton, "
        "paid 7.25 per person, with 1 sibling/spouse and 0 parents/children aboard "
        "(family of 2). Cabin deck unknown. Ticket shared by 1 passenger."
    )


def test_no_label_or_id_leakage() -> None:
    train, _ = engineered()
    for _, row in train.iterrows():
        text = describe(row).lower()
        assert "survived" not in text
        assert "passengerid" not in text
        assert str(row["Name"].split(",")[0]).lower() not in text


def test_deterministic() -> None:
    train, _ = engineered()
    again, _ = engineered()
    assert [describe(r) for _, r in train.iterrows()] == [describe(r) for _, r in again.iterrows()]


def test_label_does_not_change_description() -> None:
    train, _ = engineered()
    row = train.iloc[3]
    flipped = row.copy()
    flipped["Survived"] = 1 - row["Survived"]
    assert describe(row) == describe(flipped)


def test_age_unknown_phrasing() -> None:
    train, _ = engineered()
    text = describe(train.iloc[5])
    assert "of unknown age (estimated " in text
    assert "-year-old" not in text


def test_plural_and_singular() -> None:
    train, _ = engineered()
    assert "1 sibling/spouse and 0 parents/children" in describe(train.iloc[0])
    assert "3 siblings/spouses and 1 parent/child aboard" in describe(train.iloc[7])
    assert "Ticket shared by 1 passenger." in describe(train.iloc[0])
    _, test = engineered()
    row = test.loc[test["PassengerId"] == 898].iloc[0]
    assert "Ticket shared by 2 passengers." in describe(row)


def test_person_nouns_and_class_and_ports() -> None:
    train, _ = engineered()
    assert "boy" in describe(train.iloc[7])
    assert "girl" in describe(train.iloc[9])
    assert "woman" in describe(train.iloc[1])
    assert "1st class" in describe(train.iloc[1])
    assert "Cherbourg" in describe(train.iloc[1])
    assert "2nd class" in describe(train.iloc[9])
    assert "Queenstown" in describe(train.iloc[16])
    assert "Cabin deck C." in describe(train.iloc[1])


def test_constants() -> None:
    assert LABEL_WORDS == ("no", "yes")
    assert "yes or no" in QUESTION
    assert SYSTEM_PROMPT
