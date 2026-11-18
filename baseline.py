import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA  # noqa: F401  (kept importable from baseline)

from titanic.env import load_env
from titanic.features import FEATURE_GROUPS, engineer, load_raw
from titanic.submission import validate_submission, write_submission
from titanic.tabular import (  # noqa: F401  (re-exported: tests and tools import these from baseline)
    ACTIVE_FEATURES,
    BASE_MODELS,
    DEFAULT_FEATURES,
    EXTRA_MODELS,
    FrameModel,
    TABPFN_VERSIONS,
    accuracy,
    cross_validate,
    emb_components,
    feature_cols,
    fit_predict_full,
    fold_matrices,
    make_model,
    model_matrix,
    set_features,
    tabpfn_version,
    with_emb,
)

ALL_MODELS = BASE_MODELS + EXTRA_MODELS
# TabPFN and TabICL are slow on CPU: by default they are read from outputs/<name>/ (written by
# `make tabpfn` on a Modal GPU). Add them to BASELINE_MODELS / --models to fit them locally instead.
PRECOMPUTED_MODELS = ["tabpfn", "tabicl"]
LOCAL_DEFAULT_MODELS = [m for m in ALL_MODELS if m not in PRECOMPUTED_MODELS]
EMB_MODELS = ["logreg+emb", "gb+emb", "svc+emb"]  # base model + PCA of LLM embeddings (--embeddings)
DEFAULT_CV_SEEDS = (42, 7, 123)


def unavailable_reason(name: str) -> str | None:
    """None if the model can run here, else a one-line reason (never includes secrets)."""
    if name not in EXTRA_MODELS:
        return None
    try:
        if name == "tabpfn":
            version = tabpfn_version()
            if version not in TABPFN_VERSIONS:
                return f"TABPFN_VERSION={version!r} is not one of {', '.join(TABPFN_VERSIONS)}"
            if version != "v2" and not os.environ.get("TABPFN_TOKEN"):
                return f"TabPFN {version} needs TABPFN_TOKEN (non-commercial license, ux.priorlabs.ai)"
            os.environ.setdefault("TABPFN_NO_BROWSER", "1")  # never block on a browser login
        # Tiny probe fit: checks the import and that the weights download without a login.
        rng = np.random.RandomState(0)
        probe = pd.DataFrame({"a": rng.rand(40), "Pclass": rng.randint(1, 4, 40).astype(float), "c": ["x", "y"] * 20})
        FrameModel(name, 0, ["a", "Pclass"], ["c"]).fit(probe, np.arange(40) % 2).predict_proba(probe)
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {' '.join(str(exc).split())[:120]}"
    return None


def resolve_models(requested: list[str] | None = None, with_emb: bool = False) -> list[str]:
    """Candidate list: `requested` (or BASELINE_MODELS, else all), minus models that cannot run here."""
    if not requested:
        env = os.environ.get("BASELINE_MODELS", "").replace(",", " ").split()
        requested = env or LOCAL_DEFAULT_MODELS + (EMB_MODELS if with_emb else [])
    known = ALL_MODELS + EMB_MODELS
    unknown = [n for n in requested if n not in known]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}; choose from {known}")
    names = []
    for name in requested:
        reason = "needs --embeddings" if name.endswith("+emb") and not with_emb else unavailable_reason(name)
        if reason:
            print(f"Skipping {name}: {reason}")
        else:
            names.append(name)
    if not names:
        raise SystemExit("No usable candidate models")
    return names


def load_precomputed(
    outputs_dir: Path, seeds: list[int], train: pd.DataFrame, test: pd.DataFrame, folds: int
) -> dict[str, dict]:
    """Candidates written by tabpfn_modal.py: {name: {"oof": {seed: probs}, "test": probs}}.

    Used only if outputs/<name>/ has every requested seed's oof_seed<k>.csv, the same folds and the
    same optional feature groups; otherwise one line says why and the candidate is left out."""
    found: dict[str, dict] = {}
    for name in PRECOMPUTED_MODELS:
        run_dir = Path(outputs_dir) / name
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            print(f"Skipping {name}: no {run_dir}; run `make tabpfn` first")
            continue
        meta = json.loads(metrics_path.read_text())
        have = {int(k) for k in meta.get("oof_accuracy_by_seed", {})}
        problem = None
        if not set(seeds) <= have:
            problem = f"seeds {sorted(set(seeds) - have)} missing"
        elif meta.get("folds") != folds:
            problem = f"folds {meta.get('folds')} != {folds}"
        elif list(meta.get("features", [])) != list(ACTIVE_FEATURES):
            problem = f"features {meta.get('features')} != {ACTIVE_FEATURES}"
        if problem:
            print(f"Skipping {name}: {run_dir} does not match this run ({problem}); re-run `make tabpfn`")
            continue

        def aligned(path: Path, ids: pd.Series) -> np.ndarray:
            frame = pd.read_csv(path).set_index("PassengerId")["prob"]
            if set(ids) - set(frame.index):
                raise ValueError(f"{path} lacks PassengerIds")
            return frame.loc[ids].to_numpy(dtype=float)

        found[name] = {
            "oof": {s: aligned(run_dir / f"oof_seed{s}.csv", train["PassengerId"]) for s in seeds},
            "test": aligned(run_dir / "test_probs.csv", test["PassengerId"]),
            "version": meta.get("version"),
        }
    return found


def load_embeddings(run_dir: Path, train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """emb_oof / emb_test of a modal_app run, reordered to the PassengerId order of train / test."""
    run_dir = Path(run_dir)
    ids = json.loads((run_dir / "emb_ids.json").read_text())
    out = []
    for file, key, frame in (("emb_oof.npy", "train_ids", train), ("emb_test.npy", "test_ids", test)):
        arr = np.load(run_dir / file).astype(float)
        order = {int(pid): i for i, pid in enumerate(ids[key])}
        missing = set(frame["PassengerId"]) - set(order)
        if missing or len(arr) != len(ids[key]):
            raise ValueError(f"{run_dir}/{file} does not cover the {key} PassengerIds")
        arr = arr[[order[int(pid)] for pid in frame["PassengerId"]]]
        if not np.isfinite(arr).all():
            raise ValueError(f"{run_dir}/{file} contains NaN/inf")
        out.append(arr)
    return out[0], out[1]


def pick_best(candidates: dict[str, np.ndarray], y: np.ndarray) -> str:
    return max(candidates, key=lambda name: accuracy(y, candidates[name]))


def vote_names(ranked: list[str], top: int = 3) -> list[str]:
    return ["avg:" + "+".join(ranked[:k]) for k in range(2, min(top, len(ranked)) + 1)]


def add_averages(probs: dict[str, np.ndarray], y: np.ndarray, top: int = 3) -> dict[str, np.ndarray]:
    """Add soft-vote averages of the best 2 and best 3 single models (ranked by OOF accuracy)."""
    ranked = sorted(probs, key=lambda n: -accuracy(y, probs[n]))
    out = dict(probs)
    for vote in vote_names(ranked, top):
        out[vote] = np.mean([probs[m] for m in average_members(vote)], axis=0)
    return out


def average_members(name: str) -> list[str]:
    return re.split(r"\+(?!emb)", name.removeprefix("avg:"))  # "gb+emb+svc" -> gb+emb, svc


def run(
    data_dir: Path,
    out_dir: Path,
    folds: int,
    seed: int,
    extra_seeds: tuple[int, ...] = (),
    models: list[str] | None = None,
    embeddings: Path | None = None,
    precomputed_dir: Path | None = None,
) -> dict:
    """Every candidate is cross-validated with split seed `seed` and each of `extra_seeds`
    (`folds` folds each). Selection uses the mean accuracy over all seeds; oof.csv and
    test_probs.csv come from split seed `seed`, which stays aligned with the LLM runs."""
    train_raw, test_raw = load_raw(data_dir)
    train, test = engineer(train_raw, test_raw)
    y = train["Survived"].to_numpy().astype(int)
    emb = load_embeddings(embeddings, train, test) if embeddings else None

    candidates = resolve_models(models, with_emb=emb is not None)
    seeds = [seed] + [s for s in extra_seeds if s != seed]
    use_pre = precomputed_dir is not None and models is None and not os.environ.get("BASELINE_MODELS", "").strip()
    pre = load_precomputed(precomputed_dir, seeds, train, test, folds) if use_pre else {}
    pre = {n: v for n, v in pre.items() if n not in candidates}  # a locally fitted model wins
    single: dict[int, dict[str, np.ndarray]] = {}
    splits: list[np.ndarray] = []
    for s in seeds:
        single[s], split = cross_validate(candidates, train, y, folds, s, test=test, emb=emb)
        if s == seed:
            splits = split

    def per_seed_acc(probs_by_seed: dict[int, np.ndarray]) -> list[float]:
        return [accuracy(y, probs_by_seed[s]) for s in seeds]

    probs: dict[str, dict[int, np.ndarray]] = {n: {s: single[s][n] for s in seeds} for n in candidates}
    for n, item in pre.items():
        probs[n] = dict(item["oof"])
        for s in seeds:
            single[s][n] = item["oof"][s]
    ranked = sorted(probs, key=lambda n: -np.mean(per_seed_acc(probs[n])))
    for vote in vote_names(ranked):
        members = average_members(vote)
        probs[vote] = {s: np.mean([single[s][m] for m in members], axis=0) for s in seeds}
    accs = {n: per_seed_acc(p) for n, p in probs.items()}
    chosen = max(accs, key=lambda n: np.mean(accs[n]))
    members = average_members(chosen) if chosen.startswith("avg:") else [chosen]
    oof_chosen = probs[chosen][seed]

    # Final fit: group statistics from all train labels, applied to train (leave-one-out) and test.
    # Precomputed members (tabpfn / tabicl from Modal) bring their own test probabilities.
    local_members = [n for n in members if n not in pre]
    fitted = fit_predict_full(local_members, seed, train, y, test, emb=emb) if local_members else {}
    member_probs = [fitted[n] if n in fitted else pre[n]["test"] for n in members]
    test_probs = np.mean(member_probs, axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PassengerId": train["PassengerId"], "prob": oof_chosen}).to_csv(out_dir / "oof.csv", index=False)
    pd.DataFrame({"PassengerId": test["PassengerId"], "prob": test_probs}).to_csv(out_dir / "test_probs.csv", index=False)
    metrics = {
        "model": f"baseline:{chosen}",
        "folds": folds,
        "seed": seed,
        "split_seed": seed,
        "cv_seeds": seeds,
        "features": list(ACTIVE_FEATURES),
        "embeddings": str(embeddings) if embeddings else None,
        "oof_accuracy": accuracy(y, oof_chosen),
        "fold_accuracies": [accuracy(y[idx], oof_chosen[idx]) for idx in splits],
        "candidate_oof_accuracy": {n: accs[n][0] for n in accs},
        "candidate_cv_mean": {n: float(np.mean(a)) for n, a in accs.items()},
        "candidate_cv_std": {n: float(np.std(a)) for n, a in accs.items()},
        "chosen_oof_by_seed": {str(s): a for s, a in zip(seeds, accs[chosen])},
        "chosen_oof_mean_over_seeds": float(np.mean(accs[chosen])),
        "chosen_oof_std_over_seeds": float(np.std(accs[chosen])),
    }
    if "tabpfn" in candidates:
        metrics["tabpfn_version"] = tabpfn_version()
    if pre:
        metrics["precomputed"] = {n: v["version"] for n, v in pre.items()}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    submission_path = write_submission(
        test["PassengerId"], (test_probs >= 0.5).astype(int), Path("submissions/baseline.csv")
    )
    validate_submission(submission_path, Path(data_dir) / "test.csv")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical Titanic baseline with out-of-fold probabilities.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cv-seeds", default=",".join(map(str, DEFAULT_CV_SEEDS)),
        help="comma-separated CV split seeds (each runs --folds folds); selection uses the mean, "
        "oof.csv comes from --seed (default: 42,7,123)",
    )
    parser.add_argument(
        "--precomputed-dir", type=Path, default=Path("outputs"),
        help="where `make tabpfn` wrote tabpfn/ and tabicl/ (default: outputs); used only with the default candidate list",
    )
    parser.add_argument("--embeddings", type=Path, default=None, help="outputs/<run> with emb_oof.npy, emb_test.npy, emb_ids.json")
    parser.add_argument(
        "--features", default=None,
        help=f"comma-separated optional feature groups ({','.join(FEATURE_GROUPS)}), 'none' or 'all'; default: {','.join(DEFAULT_FEATURES) or 'none'}",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"restrict candidates (default: BASELINE_MODELS env, else {' '.join(LOCAL_DEFAULT_MODELS)} plus any precomputed tabpfn/tabicl; "
        "add tabpfn/tabicl here to fit them locally)",
    )
    load_env()
    args = parser.parse_args()

    if args.features is not None:
        names = list(FEATURE_GROUPS) if args.features == "all" else [f for f in args.features.split(",") if f and f != "none"]
        set_features(names)
    cv_seeds = tuple(int(x) for x in args.cv_seeds.replace(",", " ").split())
    metrics = run(args.data_dir, args.out_dir, args.folds, args.seed, cv_seeds, args.models, args.embeddings, args.precomputed_dir)
    print(f"Chosen: {metrics['model']}  OOF accuracy (split seed {metrics['seed']}): {metrics['oof_accuracy']:.4f}")
    print(f"  {'candidate':22s} seed{metrics['seed']}  mean over {metrics['cv_seeds']}  std")
    for name, acc in metrics["candidate_oof_accuracy"].items():
        print(f"  {name:22s} {acc:.4f}  {metrics['candidate_cv_mean'][name]:.4f}  {metrics['candidate_cv_std'][name]:.4f}")
    print(f"Chosen model OOF by CV seed: {metrics['chosen_oof_by_seed']}  mean {metrics['chosen_oof_mean_over_seeds']:.4f}"
          f"  std {metrics['chosen_oof_std_over_seeds']:.4f}")
    print(f"Wrote {args.out_dir} and submissions/baseline.csv")


if __name__ == "__main__":
    main()
