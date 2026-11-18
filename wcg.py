"""Woman-Child-Group rule model and hybrids with a model run -> submissions/wcg*.csv.

Usage: python wcg.py [--data-dir data] [--model-run outputs/baseline] [--nearby]
       python wcg.py --nearby --keys a --class3-large --woman-rate --boy-rate --model-segments
       python wcg.py --grid --reference submissions/wcg_h3.csv --model-run outputs/a --model-run outputs/b

--grid evaluates a small pre-declared set of refinements and writes submissions/refine/<name>.csv plus
summary.csv. --model-run may repeat; probabilities are averaged.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble import load_probs
from titanic.features import load_raw
from titanic.submission import validate_submission, write_submission
from titanic.wcg import (
    VARIANTS,
    Config,
    build_groups,
    cv_config,
    evaluate_grid,
    honest_cv,
    pool_passengers,
    predict_config,
    predict_test,
    reference_config,
)

FILES = {"wcg": "wcg.csv", "h1": "wcg_h1.csv", "h2": "wcg_h2.csv", "h3": "wcg_h3.csv"}
NAMES = {
    "gender": "gender rule",
    "wcg": "WCG",
    "h1": "H1 WCG overrides, gender default",
    "h2": "H2 model + WCG overrides",
    "h3": "H3 H1 + confident model flips",
}


def load_model_runs(model_runs, train, test):
    """Average (oof, test_prob) over the runs that have both files; (None, None) if none do."""
    oofs, tests = [], []
    for run_dir in model_runs or []:
        run_dir = Path(run_dir)
        if (run_dir / "oof.csv").is_file() and (run_dir / "test_probs.csv").is_file():
            oofs.append(load_probs(run_dir / "oof.csv", train["PassengerId"]))
            tests.append(load_probs(run_dir / "test_probs.csv", test["PassengerId"]))
    if not oofs:
        return None, None
    return np.mean(oofs, axis=0), np.mean(tests, axis=0)


def run(
    data_dir: Path,
    model_run,
    out_dir: Path,
    nearby: bool,
    seeds: tuple[int, ...],
    folds: int = 5,
    keys: str = "a",
    woman_rate: bool = False,
    boy_rate: bool = False,
    class3_large: bool = False,
    model_segments: bool = False,
    flip_hi: float = 0.8,
    flip_lo: float = 0.2,
) -> dict:
    train, test = load_raw(data_dir)
    pool = pool_passengers(train, test)
    n_train = len(train)
    groups = build_groups(pool, nearby=nearby, keys=keys)
    runs = [model_run] if isinstance(model_run, (str, Path)) else model_run
    oof, test_prob = load_model_runs(runs, train, test)
    opts = {"woman_rate": woman_rate, "boy_rate": boy_rate, "class3_large": class3_large}

    cv = honest_cv(pool, groups, n_train, seeds, folds, oof, opts=opts)
    out = predict_test(pool, groups, n_train, test_prob, opts=opts)

    written = {}
    for v, name in FILES.items():
        if v in out["preds"]:
            path = write_submission(test["PassengerId"], out["preds"][v], out_dir / name)
            validate_submission(path, Path(data_dir) / "test.csv")
            written[v] = str(path)
    result = {
        "nearby": nearby,
        "keys": keys,
        "rule_options": opts,
        "seeds": list(seeds),
        "model_run": [str(r) for r in runs] if oof is not None else None,
        "n_groups": int(len(np.unique(groups))),
        "cv": cv,
        "test_counts": out["counts"],
        "files": written,
        "note": "h2/h3/seg CV use the model OOF (split seed 42 only); gender/wcg/h1 use every seed.",
    }
    if model_segments and oof is not None:
        cfg = Config(keys, nearby, woman_rate, boy_rate, class3_large, "seg", flip_hi, flip_lo)
        accs = cv_config(pool, groups, n_train, cfg, oof, seeds, folds)
        known = pool["Survived"].to_numpy(dtype=float)
        pred, _ = predict_config(pool, groups, known, cfg, np.concatenate([np.full(n_train, np.nan), test_prob]))
        path = write_submission(test["PassengerId"], pred[n_train:], out_dir / "wcg_seg.csv")
        validate_submission(path, Path(data_dir) / "test.csv")
        base = out["preds"]["wcg"]
        result["segments"] = {
            "cv_by_seed": accs,
            "cv_mean": float(np.mean(list(accs.values()))),
            "test_vs_wcg": int((pred[n_train:] != base).sum()),
            "file": str(path),
        }
        written["seg"] = str(path)
    (out_dir / "wcg.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def run_grid(
    data_dir: Path,
    model_runs,
    reference: Path | None,
    out_dir: Path,
    seeds: tuple[int, ...],
    folds: int = 5,
) -> pd.DataFrame:
    """Evaluate the pre-declared grid; write validated <out_dir>/refine/<name>.csv and summary.csv."""
    train, test = load_raw(data_dir)
    pool = pool_passengers(train, test)
    oof, test_prob = load_model_runs(model_runs, train, test)
    ref_pred = None
    if reference is not None:
        ref = pd.read_csv(reference).set_index("PassengerId")["Survived"]
        ref_pred = ref.reindex(test["PassengerId"]).to_numpy()
        if np.isnan(ref_pred.astype(float)).any():
            raise ValueError(f"{reference} lacks test PassengerIds")
        ref_pred = ref_pred.astype(int)
    summary, preds = evaluate_grid(pool, len(train), oof, test_prob, ref_pred, seeds, folds)
    refine = Path(out_dir) / "refine"
    for name, pred in preds.items():
        path = write_submission(test["PassengerId"], pred, refine / f"{name}.csv")
        validate_submission(path, Path(data_dir) / "test.csv")
    summary.to_csv(refine / "summary.csv", index=False)
    summary.attrs["reference_config"] = reference_config(oof is not None).name
    summary.attrs["model"] = oof is not None
    return summary


def print_grid(summary: pd.DataFrame, has_reference: bool) -> None:
    print(f"reference config: {summary.attrs['reference_config']}  model: {summary.attrs['model']}")
    show = summary.copy()
    show["n_diff_reference"] = show["n_diff_reference"].map(lambda v: "-" if pd.isna(v) else str(int(v)))
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("candidate = CV mean >= reference config (same seeds) and 2-15 test passengers differ from --reference.")
    if not has_reference:
        print("No --reference given: no variant can be marked a candidate.")
    print("Model variants use the OOF of split seed 42 only (n_seeds=1); rule-only variants use every seed.")


def print_table(result: dict) -> None:
    print(
        f"groups: {result['n_groups']}  keys: {result['keys']}  nearby-ticket links: {result['nearby']}  "
        f"rule options: {result['rule_options']}  model: {result['model_run']}"
    )
    print(f"{'variant':36s} {'CV mean':>8s}  per seed")
    for v in VARIANTS:
        c = result["cv"][v]
        if c["mean"] is None:
            continue
        per = " ".join(f"{s}:{a:.4f}" for s, a in c["accuracy_by_seed"].items())
        print(f"{NAMES[v]:36s} {c['mean']:8.4f}  {per}")
    seg = result.get("segments")
    if seg:
        per = " ".join(f"{s}:{a:.4f}" for s, a in seg["cv_by_seed"].items())
        print(f"{'segment-restricted model flips':36s} {seg['cv_mean']:8.4f}  {per}  (test differs from WCG on {seg['test_vs_wcg']})")
    w = result["cv"]["wcg"]
    if w["overrides"]:
        n = len(result["seeds"])
        print(f"WCG overrides in CV: {w['overrides'] / n:.1f} per seed, {w['override_correct'] / w['overrides']:.3f} correct")
    print("test:", result["test_counts"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-run", type=Path, action="append", help="repeatable; probabilities are averaged")
    parser.add_argument("--out-dir", type=Path, default=Path("submissions"))
    parser.add_argument("--nearby", action="store_true", help="also link same surname/class/port tickets differing in the last character (keys a)")
    parser.add_argument("--keys", choices=["a", "b", "c"], default="a", help="a: current; b: Deotte name-only family key; c: surname+Pclass+Embarked")
    parser.add_argument("--woman-rate", action="store_true", help="woman dies if known-mate survival rate <= 0.25 (>= 2 known)")
    parser.add_argument("--boy-rate", action="store_true", help="boy lives if known-mate survival rate >= 0.75 (>= 2 known)")
    parser.add_argument("--class3-large", action="store_true", help="3rd-class woman/boy in a group of >= 5 with no known outcomes dies")
    parser.add_argument("--model-segments", action="store_true", help="model flips only for plain-default rows of three segments")
    parser.add_argument("--flip-hi", type=float, default=0.8)
    parser.add_argument("--flip-lo", type=float, default=0.2)
    parser.add_argument("--grid", action="store_true", help="evaluate the pre-declared variant grid into <out-dir>/refine/")
    parser.add_argument("--reference", type=Path, help="submission csv to count test differences against (--grid)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    args = parser.parse_args()
    if args.grid:
        summary = run_grid(args.data_dir, args.model_run, args.reference, args.out_dir, tuple(args.seeds))
        print_grid(summary, args.reference is not None)
        print(f"Wrote {args.out_dir / 'refine'}/<name>.csv and summary.csv")
        return
    runs = args.model_run or [Path("outputs/baseline")]
    result = run(
        args.data_dir, runs, args.out_dir, args.nearby, tuple(args.seeds), keys=args.keys,
        woman_rate=args.woman_rate, boy_rate=args.boy_rate, class3_large=args.class3_large,
        model_segments=args.model_segments, flip_hi=args.flip_hi, flip_lo=args.flip_lo,
    )
    print_table(result)
    for path in result["files"].values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
