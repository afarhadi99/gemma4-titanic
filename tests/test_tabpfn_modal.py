import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import baseline
import tabpfn_modal
from titanic.features import engineer, load_raw
from titanic.tabular import cross_validate, fit_predict_full, fold_splits

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"


def _texts() -> tuple[str, str]:
    return (FIXTURES / "train.csv").read_text(), (FIXTURES / "test.csv").read_text()


def test_modules_compile(tmp_path: Path) -> None:
    for name in ("tabpfn_modal.py", "baseline.py", "stack.py", "src/titanic/tabular.py"):
        py_compile.compile(str(ROOT / name), doraise=True, cfile=str(tmp_path / (Path(name).name + "c")))


def test_importing_tabpfn_modal_needs_no_torch() -> None:
    code = "import sys, tabpfn_modal; bad = [m for m in ('torch', 'tabpfn', 'tabicl') if m in sys.modules]; assert not bad, bad"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "src")])}
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_folds_are_identical_for_baseline_and_modal_path() -> None:
    y = engineer(*load_raw(FIXTURES))[0]["Survived"].to_numpy().astype(int)
    for seed in (42, 7):
        expected = [v for _, v in StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)]
        for got, want in zip(fold_splits(y, 3, seed), expected):
            np.testing.assert_array_equal(got, want)
    assert baseline.cross_validate is cross_validate and baseline.fit_predict_full is fit_predict_full


def test_modal_seed_function_matches_baseline_features_and_folds() -> None:
    """run_seed (the body of the GPU function) and baseline.cross_validate give identical OOF
    probabilities and identical final-fit test probabilities for the same model."""
    train_csv, test_csv = _texts()
    out = tabpfn_modal.run_seed(["gb", "svc"], train_csv, test_csv, seed=3, folds=2, with_test=True)
    train, test = engineer(*load_raw(FIXTURES))
    y = train["Survived"].to_numpy().astype(int)
    oof, _ = baseline.cross_validate(["gb", "svc"], train, y, 2, 3, test=test)
    final = baseline.fit_predict_full(["gb", "svc"], 3, train, y, test)
    for name in ("gb", "svc"):
        np.testing.assert_allclose(out["oof"][name], oof[name])
        np.testing.assert_allclose(out["test"][name], final[name])
    assert tabpfn_modal.run_seed(["gb"], train_csv, test_csv, 3, 2)["test"] is None


def test_helpers_parse_and_guard() -> None:
    assert tabpfn_modal.parse_models("tabpfn,tabicl") == ["tabpfn", "tabicl"]
    assert tabpfn_modal.parse_seeds("7,123") == [42, 7, 123]  # split seed always included
    with pytest.raises(ValueError):
        tabpfn_modal.parse_models("catboost")
    assert tabpfn_modal.tabpfn_skip_reason("v2", "") is None
    assert "TABPFN_TOKEN" in tabpfn_modal.tabpfn_skip_reason("v3.5-fast", "")
    assert tabpfn_modal.tabpfn_skip_reason("v3.5-fast", "x") is None
    assert tabpfn_modal.tabpfn_skip_reason("v9", "x")


def _write_precomputed(tmp_path: Path, features: list[str] | None = None, seeds=(42, 7)) -> Path:
    train_csv, test_csv = _texts()
    train, test = engineer(*load_raw(FIXTURES))
    results = []
    for s in seeds:
        r = tabpfn_modal.run_seed(["gb"], train_csv, test_csv, s, 2, features, with_test=s == 42)
        r["oof"]["tabpfn"] = r["oof"].pop("gb")
        r["seconds"]["tabpfn"] = r["seconds"].pop("gb")
        if r["test"]:
            r["test"]["tabpfn"] = r["test"].pop("gb")
        results.append(r)
    outputs = tmp_path / "outputs"
    metrics = tabpfn_modal.assemble_run(results, "tabpfn", train, test, outputs / "tabpfn", 2, "v2", "L4", features or [], 1.0)
    assert metrics["oof_accuracy_by_seed"].keys() == {str(s) for s in seeds}
    return outputs


def test_modal_outputs_are_a_normal_run_dir(tmp_path: Path) -> None:
    outputs = _write_precomputed(tmp_path)
    run_dir = outputs / "tabpfn"
    for name in ("oof.csv", "oof_seed42.csv", "oof_seed7.csv", "test_probs.csv", "metrics.json"):
        assert (run_dir / name).is_file()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["split_seed"] == 42 and metrics["gpu"] == "L4" and metrics["version"] == "v2"
    assert metrics["oof_accuracy_mean"] == pytest.approx(np.mean(list(metrics["oof_accuracy_by_seed"].values())))
    pd.testing.assert_frame_equal(pd.read_csv(run_dir / "oof.csv"), pd.read_csv(run_dir / "oof_seed42.csv"))
    from ensemble import usable_runs

    usable, skipped = usable_runs([run_dir], FIXTURES)
    assert usable == [run_dir] and not skipped


def test_baseline_uses_precomputed_candidates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("BASELINE_MODELS", raising=False)
    monkeypatch.chdir(tmp_path)
    outputs = _write_precomputed(tmp_path)
    metrics = baseline.run(FIXTURES, tmp_path / "out", 2, 42, (7,), None, None, precomputed_dir=outputs)
    assert "tabpfn" in metrics["candidate_cv_mean"] and metrics["precomputed"] == {"tabpfn": "v2"}
    assert "tabicl" not in metrics["candidate_cv_mean"]  # not run yet: skipped, not fatal


def test_baseline_skips_precomputed_that_do_not_match(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("BASELINE_MODELS", raising=False)
    monkeypatch.chdir(tmp_path)
    train, test = engineer(*load_raw(FIXTURES))
    missing = baseline.load_precomputed(tmp_path / "nothing", [42, 7], train, test, 2)
    assert missing == {} and "make tabpfn" in capsys.readouterr().out
    outputs = _write_precomputed(tmp_path)
    assert baseline.load_precomputed(outputs, [42, 7, 123], train, test, 2) == {}  # seed 123 missing
    assert "seeds [123] missing" in capsys.readouterr().out
    assert baseline.load_precomputed(outputs, [42, 7], train, test, 5) == {}  # folds differ
    assert "folds" in capsys.readouterr().out
    assert set(baseline.load_precomputed(outputs, [42, 7], train, test, 2)) == {"tabpfn"}


def test_default_local_candidates_exclude_tabpfn_and_tabicl(monkeypatch) -> None:
    monkeypatch.delenv("BASELINE_MODELS", raising=False)
    assert "tabpfn" not in baseline.LOCAL_DEFAULT_MODELS and "tabicl" not in baseline.LOCAL_DEFAULT_MODELS
    assert {"tabpfn", "tabicl"} <= set(baseline.ALL_MODELS)
