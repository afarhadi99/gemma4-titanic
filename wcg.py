"""Woman-Child-Group rule model and hybrids with a model run -> submissions/wcg*.csv.

Usage: python wcg.py [--data-dir data] [--model-run outputs/baseline] [--nearby]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ensemble import load_probs
from titanic.features import load_raw
from titanic.submission import validate_submission, write_submission
from titanic.wcg import VARIANTS, build_groups, honest_cv, pool_passengers, predict_test

FILES = {"wcg": "wcg.csv", "h1": "wcg_h1.csv", "h2": "wcg_h2.csv", "h3": "wcg_h3.csv"}
NAMES = {
    "gender": "gender rule",
    "wcg": "WCG",
    "h1": "H1 WCG overrides, gender default",
    "h2": "H2 model + WCG overrides",
    "h3": "H3 H1 + confident model flips",
}


def run(
    data_dir: Path, model_run: Path | None, out_dir: Path, nearby: bool, seeds: tuple[int, ...], folds: int = 5
) -> dict:
    train, test = load_raw(data_dir)
    pool = pool_passengers(train, test)
    n_train = len(train)
    groups = build_groups(pool, nearby=nearby)

    oof = test_prob = None
    if model_run and (model_run / "oof.csv").is_file() and (model_run / "test_probs.csv").is_file():
        oof = load_probs(model_run / "oof.csv", train["PassengerId"])
        test_prob = load_probs(model_run / "test_probs.csv", test["PassengerId"])
    cv = honest_cv(pool, groups, n_train, seeds, folds, oof)
    out = predict_test(pool, groups, n_train, test_prob)

    written = {}
    for v, name in FILES.items():
        if v in out["preds"]:
            path = write_submission(test["PassengerId"], out["preds"][v], out_dir / name)
            validate_submission(path, Path(data_dir) / "test.csv")
            written[v] = str(path)
    result = {
        "nearby": nearby,
        "seeds": list(seeds),
        "model_run": str(model_run) if oof is not None else None,
        "n_groups": int(len(np.unique(groups))),
        "cv": cv,
        "test_counts": out["counts"],
        "files": written,
        "note": "h2/h3 CV use the model OOF (split seed 42 only); gender/wcg/h1 use every seed.",
    }
    (out_dir / "wcg.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def print_table(result: dict) -> None:
    print(f"groups: {result['n_groups']}  nearby-ticket links: {result['nearby']}  model: {result['model_run']}")
    print(f"{'variant':36s} {'CV mean':>8s}  per seed")
    for v in VARIANTS:
        c = result["cv"][v]
        if c["mean"] is None:
            continue
        per = " ".join(f"{s}:{a:.4f}" for s, a in c["accuracy_by_seed"].items())
        print(f"{NAMES[v]:36s} {c['mean']:8.4f}  {per}")
    w = result["cv"]["wcg"]
    if w["overrides"]:
        n = len(result["seeds"])
        print(f"WCG overrides in CV: {w['overrides'] / n:.1f} per seed, {w['override_correct'] / w['overrides']:.3f} correct")
    print("test:", result["test_counts"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-run", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--out-dir", type=Path, default=Path("submissions"))
    parser.add_argument("--nearby", action="store_true", help="also link same surname/class/port tickets differing in the last character")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    args = parser.parse_args()
    result = run(args.data_dir, args.model_run, args.out_dir, args.nearby, tuple(args.seeds))
    print_table(result)
    for path in result["files"].values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
