from pathlib import Path

import json

import numpy as np
import pandas as pd
import pytest

from baseline import (
    BASE_MODELS,
    cross_validate,
    load_embeddings,
    resolve_models,
    run,
    set_features,
    tabpfn_version,
    unavailable_reason,
)
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


def _fake_embeddings(tmp_path: Path, dim: int = 6) -> Path:
    train, test = load_raw(FIXTURES)
    rng = np.random.RandomState(0)
    run_dir = tmp_path / "emb_run"
    run_dir.mkdir()
    np.save(run_dir / "emb_oof.npy", rng.randn(len(train), dim).astype(np.float32))
    np.save(run_dir / "emb_test.npy", rng.randn(len(test), dim).astype(np.float32))
    ids = {"train_ids": train["PassengerId"].tolist(), "test_ids": test["PassengerId"].tolist()}
    (run_dir / "emb_ids.json").write_text(json.dumps(ids))
    return run_dir


def test_load_embeddings_aligns_by_passenger_id(tmp_path: Path) -> None:
    run_dir = _fake_embeddings(tmp_path)
    train, test = engineer(*load_raw(FIXTURES))
    stored = np.load(run_dir / "emb_oof.npy")
    ids = json.loads((run_dir / "emb_ids.json").read_text())
    ids["train_ids"] = ids["train_ids"][::-1]  # stored in reverse order
    (run_dir / "emb_ids.json").write_text(json.dumps(ids))
    np.save(run_dir / "emb_oof.npy", stored[::-1])
    emb_train, emb_test = load_embeddings(run_dir, train, test)
    np.testing.assert_allclose(emb_train, stored, rtol=1e-6)
    assert emb_test.shape == (len(test), 6)


def test_embedding_candidates_run_with_fold_local_pca(tmp_path: Path) -> None:
    train, test = engineer(*load_raw(FIXTURES))
    emb = load_embeddings(_fake_embeddings(tmp_path), train, test)
    y = train["Survived"].to_numpy().astype(int)
    oof, _ = cross_validate(["gb", "gb+emb", "svc+emb", "logreg+emb"], train, y, 2, 0, test=test, emb=emb)
    for probs in oof.values():
        assert probs.shape == y.shape and np.all((probs >= 0) & (probs <= 1))
    with pytest.raises(ValueError):
        cross_validate(["gb+emb"], train, y, 2, 0, test=test)


def test_run_with_embeddings_and_repeated_cv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    emb_dir = _fake_embeddings(tmp_path)
    metrics = run(FIXTURES, tmp_path / "out", 2, 42, (7,), ["gb", "svc", "gb+emb", "svc+emb"], emb_dir)
    assert metrics["cv_seeds"] == [42, 7] and metrics["split_seed"] == 42
    assert {"gb+emb", "svc+emb"} <= set(metrics["candidate_cv_mean"])
    assert set(metrics["candidate_cv_mean"]) == set(metrics["candidate_cv_std"])
    assert set(metrics["chosen_oof_by_seed"]) == {"42", "7"}
    assert "chosen_oof_std_over_seeds" in metrics
    assert len(pd.read_csv(tmp_path / "out" / "oof.csv")) == len(load_raw(FIXTURES)[0])


def test_emb_models_are_skipped_without_embeddings(capsys) -> None:
    assert resolve_models(["gb", "gb+emb"]) == ["gb"]
    assert "needs --embeddings" in capsys.readouterr().out


def test_feature_groups_can_be_switched_on() -> None:
    import baseline

    before = list(baseline.ACTIVE_FEATURES)
    try:
        set_features(["ticket", "cabin", "deckpclass", "farerel", "titlepclass", "wcg"])
        train, test = engineer(*load_raw(FIXTURES))
        y = train["Survived"].to_numpy().astype(int)
        oof, _ = cross_validate(["gb", "catboost"], train, y, 2, 0, test=test)
        assert all(np.all(np.isfinite(p)) for p in oof.values())
        with pytest.raises(ValueError):
            set_features(["nope"])
    finally:
        set_features(before)
