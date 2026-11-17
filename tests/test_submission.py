from pathlib import Path

import pandas as pd
import pytest

from titanic.submission import validate_submission, write_submission

FIXTURES = Path(__file__).parent / "fixtures"
TEST_CSV = FIXTURES / "test.csv"


@pytest.fixture
def ids() -> list[int]:
    return pd.read_csv(TEST_CSV)["PassengerId"].tolist()


def write_raw(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def test_valid_submission_passes(tmp_path: Path, ids: list[int]) -> None:
    path = write_submission(ids, [i % 2 for i in ids], tmp_path / "sub" / "s.csv")
    validate_submission(path, TEST_CSV)
    assert path.read_text().splitlines()[0] == "PassengerId,Survived"


def test_write_coerces_to_ints(tmp_path: Path, ids: list[int]) -> None:
    path = write_submission(ids, [1.0] * len(ids), tmp_path / "s.csv")
    assert path.read_text().splitlines()[1] == f"{ids[0]},1"


def test_wrong_header(tmp_path: Path, ids: list[int]) -> None:
    path = write_raw(tmp_path / "s.csv", pd.DataFrame({"id": ids, "Survived": 0}))
    with pytest.raises(ValueError, match="Header"):
        validate_submission(path, TEST_CSV)


def test_wrong_row_count(tmp_path: Path, ids: list[int]) -> None:
    path = write_submission(ids[:-1], [0] * (len(ids) - 1), tmp_path / "s.csv")
    with pytest.raises(ValueError, match="rows"):
        validate_submission(path, TEST_CSV)


def test_duplicate_ids(tmp_path: Path, ids: list[int]) -> None:
    dup = [*ids[:-1], ids[0]]
    path = write_submission(dup, [0] * len(dup), tmp_path / "s.csv")
    with pytest.raises(ValueError, match="Duplicate"):
        validate_submission(path, TEST_CSV)


def test_missing_and_extra_ids(tmp_path: Path, ids: list[int]) -> None:
    swapped = [*ids[:-1], 99999]
    path = write_submission(swapped, [0] * len(swapped), tmp_path / "s.csv")
    with pytest.raises(ValueError, match="Missing.*unexpected"):
        validate_submission(path, TEST_CSV)


def test_non_binary_values(tmp_path: Path, ids: list[int]) -> None:
    values = [0] * len(ids)
    values[0] = 2
    path = write_raw(tmp_path / "s.csv", pd.DataFrame({"PassengerId": ids, "Survived": values}))
    with pytest.raises(ValueError, match="0 or 1"):
        validate_submission(path, TEST_CSV)


def test_non_numeric_values(tmp_path: Path, ids: list[int]) -> None:
    values = ["yes"] * len(ids)
    path = write_raw(tmp_path / "s.csv", pd.DataFrame({"PassengerId": ids, "Survived": values}))
    with pytest.raises(ValueError, match="0 or 1"):
        validate_submission(path, TEST_CSV)
