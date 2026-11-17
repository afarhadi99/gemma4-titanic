from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

HEADER = ["PassengerId", "Survived"]


def write_submission(
    passenger_ids: Sequence[int] | np.ndarray | pd.Series,
    survived: Sequence[int] | np.ndarray | pd.Series,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "PassengerId": np.asarray(passenger_ids).astype(int),
            "Survived": np.asarray(survived).astype(int),
        }
    )
    frame.to_csv(path, index=False)
    return path


def validate_submission(path: str | Path, test_csv: str | Path) -> None:
    submission = pd.read_csv(path)
    expected = pd.read_csv(test_csv)["PassengerId"]

    if list(submission.columns) != HEADER:
        raise ValueError(f"Header must be {HEADER}, got {list(submission.columns)}")
    if len(submission) != len(expected):
        raise ValueError(f"Expected {len(expected)} rows, got {len(submission)}")

    ids = submission["PassengerId"]
    duplicated = sorted(ids[ids.duplicated()].unique().tolist())
    if duplicated:
        raise ValueError(f"Duplicate PassengerIds: {duplicated[:10]}")
    missing = sorted(set(expected) - set(ids))
    extra = sorted(set(ids) - set(expected))
    if missing or extra:
        raise ValueError(f"Missing PassengerIds: {missing[:10]}; unexpected PassengerIds: {extra[:10]}")

    values = pd.to_numeric(submission["Survived"], errors="coerce")
    if not values.isin([0, 1]).all():
        raise ValueError("Survived must contain only 0 or 1")
