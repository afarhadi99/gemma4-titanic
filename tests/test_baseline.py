from pathlib import Path

import numpy as np

from baseline import BASE_MODELS, cross_validate, run
from titanic.features import engineer, load_raw

FIXTURES = Path(__file__).parent / "fixtures"


def test_cross_validate_group_features_ignore_validation_labels() -> None:
    train, test = engineer(*load_raw(FIXTURES))
    y = train["Survived"].to_numpy().astype(int)
    oof_a, splits = cross_validate(["logreg", "rf"], train, y, 2, 0, test=test)
    # Corrupt the labels of fold 0's validation rows in the frame: that fold's predictions
    # use only fit labels, so they must not move.
    corrupted = train.copy()
    corrupted.loc[train.index[splits[0]], "Survived"] = 1 - y[splits[0]]
    oof_b, _ = cross_validate(["logreg", "rf"], corrupted, y, 2, 0, test=test)
    for name in oof_a:
        np.testing.assert_allclose(oof_a[name][splits[0]], oof_b[name][splits[0]])


def test_run_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    metrics = run(FIXTURES, tmp_path / "out", 2, 42, (1,))
    assert metrics["model"].startswith("baseline:")
    assert set(BASE_MODELS) <= set(metrics["candidate_oof_accuracy"])
    for name in ("oof.csv", "test_probs.csv", "metrics.json"):
        assert (tmp_path / "out" / name).is_file()
    assert (tmp_path / "submissions" / "baseline.csv").is_file()
