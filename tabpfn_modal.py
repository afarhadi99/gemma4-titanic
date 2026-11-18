"""TabPFN and TabICL on a Modal GPU, with the exact per-fold pipeline of baseline.py.

Both models are slow on a 2-core codespace, so they run here instead: one GPU function per CV
seed, all seeds in parallel. Inside each call the code is `titanic.tabular` (the same engineer(),
fold-aware add_group_features, feature columns and StratifiedKFold split that baseline.py uses),
fitted on device="cuda". Results are written as ordinary run directories:

    outputs/tabpfn/  and  outputs/tabicl/
        oof.csv            OOF probabilities for split seed 42 (aligned with the LLM runs)
        oof_seed<k>.csv    OOF probabilities for every CV seed
        test_probs.csv     test probabilities from a final fit on all train rows (seed-42 call)
        metrics.json       per-seed OOF accuracy, mean/std, version, gpu, ...

so ensemble.py, compare.py, stack.py and baseline.py (precomputed candidates) pick them up.

Configuration comes from .env (see .env.example): MODAL_TOKEN_ID / MODAL_TOKEN_SECRET for the local
Modal client, TABPFN_TOKEN (forwarded to the container, never printed) and TABPFN_VERSION
(v2 | v3.5-fast | v3.5; default v3.5-fast with a token, else v2), optional TABPFN_GPU (default L4).
TabPFN v3.5 weights are under a NON-COMMERCIAL license (accept it at https://ux.priorlabs.ai).

Usage (repo root, data in ./data):
    python -m modal run tabpfn_modal.py                       # same as `make tabpfn`
    python -m modal run tabpfn_modal.py --models tabpfn --gpu A10G
"""

import io
import json
import os
import time
from pathlib import Path

import modal
import numpy as np
import pandas as pd

from titanic.env import load_env

if modal.is_local():
    load_env()

APP_NAME = "titanic-tabpfn"
CACHE_DIR = "/cache"
DEFAULT_GPU = os.environ.get("TABPFN_GPU") or "L4"
SPLIT_SEED = 42
DEFAULT_SEEDS = "42,7,123"
DEFAULT_FOLDS = 5
MODEL_CHOICES = ("tabpfn", "tabicl")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.15.0",  # default PyPI wheel (CUDA)
        "tabpfn==10.2.0",
        "tabicl==2.2.0",
        "pandas==3.1.1",
        "numpy==2.4.6",
        "scikit-learn==1.9.1",
    )
    .env(
        {
            "TABPFN_MODEL_CACHE_DIR": f"{CACHE_DIR}/tabpfn",
            "HF_HOME": f"{CACHE_DIR}/hf",
            "TABPFN_NO_BROWSER": "1",
        }
    )
    .add_local_python_source("titanic")
)

cache_volume = modal.Volume.from_name("tabpfn-cache", create_if_missing=True)
tabpfn_secret = modal.Secret.from_dict({"TABPFN_TOKEN": os.environ.get("TABPFN_TOKEN", "")})


# --------------------------------------------------------------------------- #
# Pure helpers (no GPU libraries; unit tested in tests/test_tabpfn_modal.py)
# --------------------------------------------------------------------------- #


def parse_models(text: str) -> list[str]:
    names = [m for m in text.replace(",", " ").split() if m]
    bad = [m for m in names if m not in MODEL_CHOICES]
    if bad or not names:
        raise ValueError(f"--models must be a subset of {MODEL_CHOICES}, got {text!r}")
    return list(dict.fromkeys(names))


def parse_seeds(text: str) -> list[int]:
    seeds = [int(s) for s in text.replace(",", " ").split()]
    if not seeds:
        raise ValueError("--seeds must not be empty")
    # The split seed is always included: it produces oof.csv and the test probabilities.
    return list(dict.fromkeys([SPLIT_SEED] + seeds))


def tabpfn_skip_reason(version: str, token: str) -> str | None:
    from titanic.tabular import TABPFN_VERSIONS

    if version not in TABPFN_VERSIONS:
        return f"TABPFN_VERSION={version!r} is not one of {', '.join(TABPFN_VERSIONS)}"
    if version != "v2" and not token:
        return f"TabPFN {version} needs TABPFN_TOKEN (non-commercial license, ux.priorlabs.ai)"
    return None


def run_seed(
    models: list[str],
    train_csv: str,
    test_csv: str,
    seed: int,
    folds: int = DEFAULT_FOLDS,
    features: list[str] | None = None,
    with_test: bool = False,
    device: str = "cpu",
) -> dict:
    """One CV seed: OOF probabilities per model and, if `with_test`, test probabilities from a
    final fit on all train rows. This is the same code path baseline.py runs
    (titanic.tabular.cross_validate / fit_predict_full)."""
    from titanic.features import engineer
    from titanic.tabular import cross_validate, fit_predict_full, set_features

    set_features(list(features or []))
    train, test = engineer(pd.read_csv(io.StringIO(train_csv)), pd.read_csv(io.StringIO(test_csv)))
    y = train["Survived"].to_numpy().astype(int)
    oof: dict[str, list[float]] = {}
    seconds: dict[str, float] = {}
    for name in models:
        start = time.time()
        probs, _ = cross_validate([name], train, y, folds, seed, test=test, device=device)
        oof[name] = probs[name].tolist()
        seconds[name] = round(time.time() - start, 1)
    test_probs = None
    if with_test:
        start = time.time()
        fitted = fit_predict_full(models, seed, train, y, test, device=device)
        test_probs = {n: p.tolist() for n, p in fitted.items()}
        seconds["final_fit"] = round(time.time() - start, 1)
    return {"seed": seed, "oof": oof, "test": test_probs, "seconds": seconds}


def assemble_run(
    results: list[dict],
    model: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
    folds: int,
    version: str | None,
    gpu: str,
    features: list[str],
    elapsed_seconds: float,
) -> dict:
    """Write outputs/<model>/ from the per-seed results and return its metrics."""
    y = train["Survived"].to_numpy().astype(int)
    by_seed = {r["seed"]: np.array(r["oof"][model]) for r in results}
    acc = {s: float(((p >= 0.5).astype(int) == y).mean()) for s, p in by_seed.items()}
    test_probs = next(np.array(r["test"][model]) for r in results if r["test"] and r["seed"] == SPLIT_SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = train["PassengerId"]
    for s, p in by_seed.items():
        pd.DataFrame({"PassengerId": ids, "prob": p}).to_csv(out_dir / f"oof_seed{s}.csv", index=False)
    pd.DataFrame({"PassengerId": ids, "prob": by_seed[SPLIT_SEED]}).to_csv(out_dir / "oof.csv", index=False)
    pd.DataFrame({"PassengerId": test["PassengerId"], "prob": test_probs}).to_csv(
        out_dir / "test_probs.csv", index=False
    )
    from titanic.tabular import fold_splits

    metrics = {
        "model": f"{model}:{version}" if version else model,
        "model_id": model,
        "run_name": model,
        "version": version,
        "gpu": gpu,
        "folds": folds,
        "seed": SPLIT_SEED,
        "split_seed": SPLIT_SEED,
        "cv_seeds": sorted(by_seed),
        "features": features,
        "smoke": False,
        "n_train": int(len(y)),
        "n_test": int(len(test)),
        "oof_accuracy": acc[SPLIT_SEED],
        "fold_accuracies": [
            float(((by_seed[SPLIT_SEED][i] >= 0.5).astype(int) == y[i]).mean())
            for i in fold_splits(y, folds, SPLIT_SEED)
        ],
        "oof_accuracy_by_seed": {str(s): a for s, a in sorted(acc.items())},
        "oof_accuracy_mean": float(np.mean(list(acc.values()))),
        "oof_accuracy_std": float(np.std(list(acc.values()))),
        "seconds_by_seed": {str(r["seed"]): r["seconds"].get(model) for r in results},
        "elapsed_seconds": round(elapsed_seconds, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# --------------------------------------------------------------------------- #
# Remote GPU functions
# --------------------------------------------------------------------------- #


@app.function(image=image, volumes={CACHE_DIR: cache_volume}, secrets=[tabpfn_secret], timeout=30 * 60)
def prefetch_weights(models: list[str], version: str) -> list[str]:
    """Download the weights once into the tabpfn-cache volume (CPU only), so the parallel GPU
    functions never race to download the same file."""
    from titanic.tabular import FrameModel

    os.environ["TABPFN_VERSION"] = version
    rng = np.random.RandomState(0)
    probe = pd.DataFrame({"a": rng.rand(40), "c": ["x", "y"] * 20})
    for name in models:
        FrameModel(name, 0, ["a"], ["c"], "cpu").fit(probe, np.arange(40) % 2)
    cache_volume.commit()
    return models


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=30 * 60,
    volumes={CACHE_DIR: cache_volume},
    secrets=[tabpfn_secret],
)
def tabpfn_seed(
    models: list[str],
    train_csv: str,
    test_csv: str,
    seed: int,
    folds: int,
    version: str,
    features: list[str],
    with_test: bool,
) -> dict:
    """All requested models for one CV seed on the GPU (test probabilities only when with_test)."""
    import torch

    cache_volume.reload()
    os.environ["TABPFN_VERSION"] = version
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"seed {seed}: models {models}, device {device}, version {version}")
    return run_seed(models, train_csv, test_csv, seed, folds, features, with_test, device)


@app.local_entrypoint()
def main(
    data_dir: str = "data",
    models: str = ",".join(MODEL_CHOICES),
    seeds: str = DEFAULT_SEEDS,
    folds: int = DEFAULT_FOLDS,
    gpu: str = DEFAULT_GPU,
    features: str = "",
    out_dir: str = "outputs",
):
    from titanic.tabular import ACTIVE_FEATURES, tabpfn_version

    start = time.time()
    names = parse_models(models)
    seed_list = parse_seeds(seeds)
    version = tabpfn_version()
    if "tabpfn" in names:
        reason = tabpfn_skip_reason(version, os.environ.get("TABPFN_TOKEN", ""))
        if reason:
            print(f"Skipping tabpfn: {reason}")
            names.remove("tabpfn")
    if not names:
        raise SystemExit("Nothing to run")
    feature_list = [f for f in features.replace(",", " ").split() if f != "none"] if features else list(ACTIVE_FEATURES)

    train_text = (Path(data_dir) / "train.csv").read_text()
    test_text = (Path(data_dir) / "test.csv").read_text()
    train = pd.read_csv(io.StringIO(train_text))
    test = pd.read_csv(io.StringIO(test_text))
    print(f"{len(train)} train rows, {len(test)} test rows; models {names}, seeds {seed_list}, gpu {gpu}, "
          f"tabpfn version {version if 'tabpfn' in names else '-'}")

    prefetch_weights.remote(names, version)
    arguments = [
        (names, train_text, test_text, seed, folds, version, feature_list, seed == SPLIT_SEED) for seed in seed_list
    ]
    results = list(tabpfn_seed.with_options(gpu=gpu).starmap(arguments))

    elapsed = time.time() - start
    for name in names:
        metrics = assemble_run(
            results, name, train, test, Path(out_dir) / name, folds,
            version if name == "tabpfn" else None, gpu, feature_list, elapsed,
        )
        print(f"{name}: OOF {metrics['oof_accuracy']:.4f} (split seed {SPLIT_SEED}), by seed "
              f"{metrics['oof_accuracy_by_seed']}, mean {metrics['oof_accuracy_mean']:.4f} "
              f"std {metrics['oof_accuracy_std']:.4f} -> {Path(out_dir) / name}")
    print(f"Done in {elapsed:.0f}s. Now run `make baseline` to include them in the vote.")
