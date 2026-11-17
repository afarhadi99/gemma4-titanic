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


def describe(row: pd.Series) -> str:
    pclass = int(row["Pclass"])
    port = PORTS.get(row["Embarked"], str(row["Embarked"]))
    siblings = _plural(int(row["SibSp"]), "sibling/spouse", "siblings/spouses")
    parents = _plural(int(row["Parch"]), "parent/child", "parents/children")
    group = _plural(int(row["TicketGroupSize"]), "passenger", "passengers")
    return (
        f"{_who(row)} travelling in {CLASS_WORDS.get(pclass, f'{pclass}th')} class, "
        f"embarked at {port}, paid {float(row['FarePerPerson']):.2f} per person, "
        f"with {siblings} and {parents} aboard (family of {int(row['FamilySize'])}). "
        f"{_cabin(row['Deck'])} Ticket shared by {group}."
    )
