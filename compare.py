"""Compare finished runs: python compare.py [--outputs outputs] [--train data/train.csv]

Reads outputs/*/metrics.json (smoke runs skipped) and prints one row per run, best OOF first.
Log-loss and AUC come from oof.csv against the train labels when both files exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _probability_metrics(oof_path: Path, train_csv: Path | None):
    if train_csv is None or not train_csv.exists() or not oof_path.exists():
        return None, None
    from sklearn.metrics import log_loss, roc_auc_score

    oof = pd.read_csv(oof_path)
    labels = pd.read_csv(train_csv).set_index("PassengerId")["Survived"]
    y = labels.reindex(oof["PassengerId"]).to_numpy()
    if np.isnan(y.astype(float)).any() or len(set(y)) < 2:
        return None, None
    p = oof["prob"].clip(1e-6, 1 - 1e-6).to_numpy()
    return float(log_loss(y, p)), float(roc_auc_score(y, p))


def collect_rows(outputs_dir: Path, train_csv: Path | None = None) -> list[dict]:
    rows = []
    for path in sorted(Path(outputs_dir).glob("*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        if m.get("smoke"):
            continue
        folds = [a for a in m.get("fold_accuracies", []) if a is not None]
        logloss, auc = _probability_metrics(path.parent / "oof.csv", train_csv)
        rows.append(
            {
                "run": m.get("run_name", path.parent.name),
                "model_id": m.get("model_id", m.get("model", "?")),
                "oof_acc": m.get("oof_accuracy"),
                "fold_mean": float(np.mean(folds)) if folds else None,
                "fold_std": float(np.std(folds)) if folds else None,
                "split_seed": m.get("split_seed"),
                "seed": m.get("seed"),
                "head": m.get("head"),
                "prompt_style": m.get("prompt_style"),
                "epochs": m.get("epochs"),
                "lr": m.get("lr"),
                "lora_r": m.get("lora_r"),
                "logloss": logloss,
                "auc": auc,
                "seconds": m.get("elapsed_seconds"),
            }
        )
    return sorted(rows, key=lambda r: -(r["oof_acc"] or 0))


OPTIONAL_COLS = ("split_seed", "seed", "head", "prompt_style")


def format_table(rows: list[dict]) -> str:
    # Optimisation columns only appear when at least one run recorded them.
    cols = ["run", "model_id", "oof_acc", "fold_mean", "fold_std"]
    cols += [c for c in OPTIONAL_COLS if any(r.get(c) is not None for r in rows)]
    cols += ["epochs", "lr", "lora_r", "logloss", "auc", "seconds"]

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.4g}" if abs(v) < 1 and v != 0 and v < 0.001 else f"{v:.4f}" if v < 100 else f"{v:.0f}"
        return str(v)

    table = [cols] + [[fmt(r.get(c)) for c in cols] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(cols))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--train", default="data/train.csv")
    args = parser.parse_args()
    rows = collect_rows(Path(args.outputs), Path(args.train))
    print(format_table(rows) if rows else f"No non-smoke runs with metrics.json under {args.outputs}/")


if __name__ == "__main__":
    main()
