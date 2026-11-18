import pandas as pd

SYSTEM_PROMPT = (
    "You predict whether a Titanic passenger survived the sinking; answer with a single word, yes or no."
)
QUESTION = "Did this passenger survive the sinking? Answer yes or no."
LABEL_WORDS: tuple[str, str] = ("no", "yes")

PORTS = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
CLASS_WORDS = {1: "1st", 2: "2nd", 3: "3rd"}
CHILD_AGE_LIMIT = 16


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _person_noun(sex: str, age: float) -> str:
    child = age < CHILD_AGE_LIMIT
    if str(sex).lower() == "female":
        return "girl" if child else "woman"
    return "boy" if child else "man"


def _who(row: pd.Series) -> str:
    age = float(row["Age"])
    noun = _person_noun(row["Sex"], age)
    title = f"(title {row['Title']})"
    if bool(row["AgeMissing"]):
        return f"A {noun} {title} of unknown age (estimated {round(age)})"
    if age < 1:
        return f"An infant {noun} {title}"
    return f"A {int(age)}-year-old {noun} {title}"


def _cabin(deck: str) -> str:
    return "Cabin deck unknown." if deck == "U" else f"Cabin deck {deck}."


def _group_sentence(row: pd.Series) -> str:
    """Plain-English group signal; empty when group features are absent or the passenger is alone.

    Uses only counts derived from OTHER passengers' labels (see add_group_features)."""
    if "GroupKnownCount" not in row.index or pd.isna(row["GroupKnownCount"]):
        return ""
    size = int(row["GroupSize"])
    if size < 2:
        return ""
    known = int(row["GroupKnownCount"])
    intro = f"Travelling in a group of {size} sharing a ticket or family name and fare;"
    if known == 0:
        return f"{intro} no other members of this group have known outcomes."
    survived = int(row["GroupKnownSurvived"])
    members = _plural(known, "other member", "other members")
    outcomes = "a known outcome" if known == 1 else "known outcomes"
    return f"{intro} of the {members} with {outcomes}, {survived} survived."


def describe(row: pd.Series) -> str:
    pclass = int(row["Pclass"])
    port = PORTS.get(row["Embarked"], str(row["Embarked"]))
    siblings = _plural(int(row["SibSp"]), "sibling/spouse", "siblings/spouses")
    parents = _plural(int(row["Parch"]), "parent/child", "parents/children")
    group = _plural(int(row["TicketGroupSize"]), "passenger", "passengers")
    text = (
        f"{_who(row)} travelling in {CLASS_WORDS.get(pclass, f'{pclass}th')} class, "
        f"embarked at {port}, paid {float(row['FarePerPerson']):.2f} per person, "
        f"with {siblings} and {parents} aboard (family of {int(row['FamilySize'])}). "
        f"{_cabin(row['Deck'])} Ticket shared by {group}."
    )
    extra = _group_sentence(row)
    return f"{text} {extra}" if extra else text


def _age_field(row: pd.Series) -> str:
    age = float(row["Age"])
    if bool(row["AgeMissing"]):
        return f"{round(age)} (estimated)"
    return str(int(age)) if age == int(age) else f"{age:g}"


def describe_structured(row: pd.Series) -> str:
    """Key-value rendering of the same facts as `describe` (never the label or PassengerId)."""
    pclass = int(row["Pclass"])
    deck = "unknown" if row["Deck"] == "U" else row["Deck"]
    parts = [
        f"Sex: {str(row['Sex']).lower()}",
        f"Age: {_age_field(row)}",
        f"Class: {pclass}",
        f"Title: {row['Title']}",
        f"Fare per person: {float(row['FarePerPerson']):.2f}",
        f"Embarked: {PORTS.get(row['Embarked'], str(row['Embarked']))}",
        f"Family size: {int(row['FamilySize'])}",
        f"Deck: {deck}",
        f"Ticket group size: {int(row['TicketGroupSize'])}",
    ]
    if "GroupKnownCount" in row.index and not pd.isna(row["GroupKnownCount"]) and int(row["GroupSize"]) >= 2:
        known = int(row["GroupKnownCount"])
        outcomes = f"Group outcomes known: {known}"
        if known:
            outcomes += f", survived: {int(row['GroupKnownSurvived'])}"
        parts.append(outcomes)
    return " | ".join(parts)


PROMPT_STYLES = {"prose": describe, "structured": describe_structured}
