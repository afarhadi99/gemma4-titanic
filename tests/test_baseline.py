from pathlib import Path

import numpy as np
import pytest

from baseline import BASE_MODELS, cross_validate, resolve_models, run, tabpfn_version, unavailable_reason
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


def _fixture_oof(name: str) -> np.ndarray:
    train, test = engineer(*load_raw(FIXTURES))
    y = train["Survived"].to_numpy().astype(int)
    oof, _ = cross_validate([name], train, y, 2, 0, test=test)
    probs = oof[name]
    assert probs.shape == y.shape and np.all((probs >= 0) & (probs <= 1))
    return probs


def test_gradient_boosting_libraries_run_on_fixtures() -> None:
    pytest.importorskip("catboost")
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    for name in ("catboost", "xgboost", "lightgbm"):
        _fixture_oof(name)


@pytest.mark.parametrize("name", ["tabpfn", "tabicl"])
def test_pretrained_tabular_models_run_or_skip(name: str) -> None:
    pytest.importorskip(name)
    reason = unavailable_reason(name)
    if reason:
        pytest.skip(f"{name} unavailable: {reason}")
    _fixture_oof(name)


def test_resolve_models_restricts_and_rejects_unknown(monkeypatch) -> None:
    monkeypatch.setenv("BASELINE_MODELS", "gb,rf")
    assert resolve_models() == ["gb", "rf"]
    assert resolve_models(["svc"]) == ["svc"]
    with pytest.raises(SystemExit):
        resolve_models(["nope"])


def test_tabpfn_version_default_depends_on_token(monkeypatch) -> None:
    monkeypatch.delenv("TABPFN_VERSION", raising=False)
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    assert tabpfn_version() == "v2"
    monkeypatch.setenv("TABPFN_TOKEN", "x")
    assert tabpfn_version() == "v3.5-fast"
    monkeypatch.setenv("TABPFN_VERSION", "v3.5")
    assert tabpfn_version() == "v3.5"
