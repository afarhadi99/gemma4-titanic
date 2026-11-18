import json

import pandas as pd

from compare import collect_rows, format_table


def _run(root, name, acc, folds, smoke=False):
    d = root / name
    d.mkdir()
    (d / "metrics.json").write_text(json.dumps({
        "run_name": name, "model_id": f"org/{name}", "oof_accuracy": acc, "fold_accuracies": folds,
        "epochs": 3, "lr": 1e-4, "lora_r": 16, "elapsed_seconds": 120.0, "smoke": smoke,
    }))
    pd.DataFrame({"PassengerId": [1, 2, 3, 4], "prob": [0.1, 0.9, 0.2, 0.8]}).to_csv(d / "oof.csv", index=False)


def test_collect_and_format(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    _run(out, "a", 0.80, [0.78, 0.82])
    _run(out, "b", 0.85, [0.85, 0.85])
    _run(out, "s", 0.99, [1.0], smoke=True)
    train = tmp_path / "train.csv"
    pd.DataFrame({"PassengerId": [1, 2, 3, 4], "Survived": [0, 1, 0, 1]}).to_csv(train, index=False)
    rows = collect_rows(out, train)
    assert [r["run"] for r in rows] == ["b", "a"]
    assert abs(rows[1]["fold_mean"] - 0.80) < 1e-9 and abs(rows[1]["fold_std"] - 0.02) < 1e-9
    assert rows[0]["auc"] == 1.0 and rows[0]["logloss"] < 0.3
    assert "org/b" in format_table(rows)
    assert collect_rows(out, None)[0]["auc"] is None
