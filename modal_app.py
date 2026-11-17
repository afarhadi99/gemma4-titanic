"""Fine-tune Gemma 4 with LoRA on Kaggle Titanic, on Modal GPUs.

Each passenger is described in English; the model answers "yes"/"no" and the
prediction is the softmax over those two label-token logits. K-fold training gives
out-of-fold (OOF) probabilities plus K test predictions that are averaged.

Configuration comes from a single .env file in the repo root (see .env.example):
MODAL_TOKEN_ID / MODAL_TOKEN_SECRET authenticate the local Modal client and HF_TOKEN is
forwarded to the GPU containers. Optional defaults: GEMMA_MODEL_ID, MODAL_GPU, FOLDS,
EPOCHS, LR, SEED. Command-line flags override them.

Usage (run from the repo root, data in ./data):
    uv run modal run modal_app.py --smoke
    uv run modal run modal_app.py
    uv run modal run modal_app.py --model-id google/gemma-4-E4B-it --gpu A100-80GB --epochs 2

Results land in outputs/<run-name>/{oof.csv,test_probs.csv,metrics.json}; adapters are
saved to the `titanic-gemma4-runs` Modal volume under /runs/<run-name>/fold<k>/.
"""

from __future__ import annotations

import io
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import modal
import numpy as np
import pandas as pd

from titanic.env import load_env

if modal.is_local():
    load_env()

APP_NAME = "titanic-gemma4"
HF_HOME = "/root/.cache/huggingface"
RUNS_DIR = "/runs"
DEFAULT_MODEL_ID = os.environ.get("GEMMA_MODEL_ID") or "google/gemma-4-E2B-it"
DEFAULT_GPU = os.environ.get("MODAL_GPU") or "L40S"
DEFAULT_FOLDS = int(os.environ.get("FOLDS") or 5)
DEFAULT_EPOCHS = int(os.environ.get("EPOCHS") or 3)
DEFAULT_LR = float(os.environ.get("LR") or 1e-4)
DEFAULT_SEED = int(os.environ.get("SEED") or 42)
SMOKE_ROWS = 64
SMOKE_FOLDS = 2
SMOKE_MAX_STEPS = 10

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch>=2.7",
        "transformers>=5.5.2,<6",
        "peft>=0.19.0",
        "accelerate>=1.12",
        "pandas",
        "numpy",
        "scikit-learn",
    )
    .env({"HF_HOME": HF_HOME})
    .add_local_python_source("titanic")
)

hf_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("titanic-gemma4-runs", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})


# --------------------------------------------------------------------------- #
# Pure helpers (no GPU libraries; unit tested in tests/test_modal_helpers.py)
# --------------------------------------------------------------------------- #


def short_model_name(model_id: str) -> str:
    """'google/gemma-4-E2B-it' -> 'e2b' (falls back to the lowercased last path segment)."""
    last = model_id.rsplit("/", 1)[-1].lower()
    parts = last.split("-")
    if len(parts) >= 3 and parts[0] == "gemma":
        core = [p for p in parts[2:] if p not in ("it", "pt")]
        if core:
            return "-".join(core)
    return last


def default_run_name(model_id: str, now: datetime | None = None, smoke: bool = False) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    suffix = "-smoke" if smoke else ""
    return f"gemma4-{short_model_name(model_id)}-{stamp}{suffix}"


def build_messages(description: str, system_prompt: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{description}\n\n{question}"},
    ]


def warmup_steps_for(n_rows: int, batch_size: int, epochs: int, max_steps: int, ratio: float = 0.05) -> int:
    total = max_steps if max_steps > 0 else math.ceil(n_rows / batch_size) * epochs
    return max(1, int(total * ratio))


def accuracy(y_true, probs, threshold: float = 0.5) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probs, dtype=float)
    if len(y) == 0:
        raise ValueError("Cannot compute accuracy of zero rows")
    return float(((p >= threshold).astype(int) == y).mean())


def assemble_oof(results: list[dict], train_ids) -> pd.DataFrame:
    """Combine per-fold validation predictions into one frame in `train_ids` order."""
    ids: list[int] = []
    probs: list[float] = []
    for result in results:
        ids.extend(int(i) for i in result["val_ids"])
        probs.extend(float(p) for p in result["val_probs"])
    if len(ids) != len(set(ids)):
        raise ValueError("A passenger appears in more than one validation fold")
    expected = [int(i) for i in train_ids]
    missing = set(expected) - set(ids)
    extra = set(ids) - set(expected)
    if missing or extra:
        raise ValueError(
            f"OOF ids do not match train ids ({len(missing)} missing, {len(extra)} unexpected)"
        )
    oof = pd.Series(probs, index=ids).reindex(expected)
    return pd.DataFrame({"PassengerId": expected, "prob": oof.to_numpy()})


def average_test_probs(results: list[dict], test_ids) -> pd.DataFrame:
    """Mean of the per-fold test probabilities, in `test_ids` order."""
    expected = [int(i) for i in test_ids]
    columns = []
    for result in results:
        ids = [int(i) for i in result["test_ids"]]
        if ids != expected:
            raise ValueError("Fold test ids do not match the test set order")
        columns.append(np.asarray(result["test_probs"], dtype=float))
    if not columns:
        raise ValueError("No fold results to average")
    return pd.DataFrame({"PassengerId": expected, "prob": np.mean(columns, axis=0)})


def build_metrics(
    *,
    run_name: str,
    model_id: str,
    gpu: str,
    folds: int,
    epochs: int,
    lr: float,
    lora_r: int,
    lora_alpha: int,
    seed: int,
    smoke: bool,
    oof: pd.DataFrame,
    y_true,
    results: list[dict],
    n_test: int,
    elapsed_seconds: float,
) -> dict:
    ordered = sorted(results, key=lambda r: r["fold"])
    return {
        "model": f"gemma4-lora:{model_id}",
        "run_name": run_name,
        "model_id": model_id,
        "gpu": gpu,
        "folds": folds,
        "epochs": epochs,
        "lr": lr,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "seed": seed,
        "smoke": smoke,
        "n_train": int(len(oof)),
        "n_test": int(n_test),
        "oof_accuracy": accuracy(y_true, oof["prob"]),
        "fold_accuracies": [r["val_accuracy"] for r in ordered],
        "elapsed_seconds": round(elapsed_seconds, 1),
    }


def make_cfg(
    *,
    model_id: str,
    run_name: str,
    epochs: int,
    lr: float,
    max_steps: int,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    batch_size: int = 16,
    eval_batch_size: int = 16,
) -> dict:
    return {
        "model_id": model_id,
        "run_name": run_name,
        "epochs": epochs,
        "lr": lr,
        "max_steps": max_steps,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
    }


# --------------------------------------------------------------------------- #
# Remote GPU function
# --------------------------------------------------------------------------- #


@app.function(
    image=image,
    volumes={HF_HOME: hf_volume},
    secrets=[hf_secret],
    timeout=30 * 60,
)
def prefetch_model(model_id: str) -> str:
    """Download the model weights once into the hf-cache volume (CPU only)."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(model_id, ignore_patterns=["*.gguf", "original/*"])
    hf_volume.commit()
    return path


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60,
    volumes={HF_HOME: hf_volume, RUNS_DIR: runs_volume},
    secrets=[hf_secret],
)
def train_fold(
    train_csv: str, test_csv: str, fold: int, folds: int, seed: int, cfg: dict
) -> dict:
    """Train one LoRA fold (fold=-1 trains on all rows) and score val + test rows."""
    import inspect

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from sklearn.model_selection import StratifiedKFold
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from titanic.features import engineer
    from titanic.lora import select_lora_targets
    from titanic.text import LABEL_WORDS, QUESTION, SYSTEM_PROMPT, describe

    hf_volume.reload()
    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN is empty (set it in .env); continuing without it")
    transformers.set_seed(seed + max(fold, 0))
    train_raw = pd.read_csv(io.StringIO(train_csv))
    test_raw = pd.read_csv(io.StringIO(test_csv))
    train_df, test_df = engineer(train_raw, test_raw)

    if fold >= 0:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        tr_idx, va_idx = list(splitter.split(train_df, train_df["Survived"]))[fold]
    else:
        tr_idx, va_idx = np.arange(len(train_df)), np.array([], dtype=int)
    fit_df = train_df.iloc[tr_idx]
    val_df = train_df.iloc[va_idx]
    print(f"[fold {fold}] fit={len(fit_df)} val={len(val_df)} test={len(test_df)}", flush=True)

    model_id = cfg["model_id"]
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        print(f"AutoTokenizer failed ({exc!r}); falling back to AutoProcessor.tokenizer")
        tokenizer = transformers.AutoProcessor.from_pretrained(model_id).tokenizer

    load_kwargs = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception as exc:
        fallback = getattr(transformers, "AutoModelForMultimodalLM", None)
        if fallback is None:
            raise
        print(f"AutoModelForCausalLM failed ({exc!r}); falling back to AutoModelForMultimodalLM")
        model = fallback.from_pretrained(model_id, **load_kwargs)
    model.to("cuda")
    print(f"Loaded {type(model).__name__}", flush=True)

    label_ids = []
    for word in LABEL_WORDS:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if len(encoded) != 1:
            print(f"WARNING: label word {word!r} encodes to {encoded}; using the first token")
        label_ids.append(encoded[0])
    assert label_ids[0] != label_ids[1], f"Label tokens collide: {label_ids}"
    print(f"Label token ids {dict(zip(LABEL_WORDS, label_ids))}", flush=True)

    def prompt_ids(row: pd.Series) -> list[int]:
        messages = build_messages(describe(row), SYSTEM_PROMPT, QUESTION)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def encode_frame(df: pd.DataFrame) -> list[list[int]]:
        return [prompt_ids(row) for _, row in df.iterrows()]

    fit_prompts = encode_frame(fit_df)
    print(f"Example prompt:\n{tokenizer.decode(fit_prompts[0])}", flush=True)
    fit_labels = fit_df["Survived"].astype(int).tolist()
    examples = [
        {
            "input_ids": ids + [label_ids[label]],
            "labels": [-100] * len(ids) + [label_ids[label]],
        }
        for ids, label in zip(fit_prompts, fit_labels)
    ]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    forward_params = inspect.signature(model.forward).parameters
    extra_zero_keys = [k for k in ("mm_token_type_ids", "token_type_ids") if k in forward_params]

    def pad_batch(sequences: list[list[int]], fill: int) -> torch.Tensor:
        width = max(len(s) for s in sequences)
        return torch.tensor([s + [fill] * (width - len(s)) for s in sequences], dtype=torch.long)

    def collate(batch: list[dict]) -> dict:
        input_ids = pad_batch([b["input_ids"] for b in batch], pad_id)
        out = {
            "input_ids": input_ids,
            "attention_mask": pad_batch([[1] * len(b["input_ids"]) for b in batch], 0),
            "labels": pad_batch([b["labels"] for b in batch], -100),
        }
        for key in extra_zero_keys:
            out[key] = torch.zeros_like(input_ids)
        return out

    targets = select_lora_targets([(n, type(m).__name__) for n, m in model.named_modules()])
    print(f"LoRA targets: {len(targets)}; examples: {targets[:3]}", flush=True)
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    batch_size = cfg["batch_size"]
    max_steps = cfg["max_steps"]
    args = TrainingArguments(
        output_dir=f"/tmp/trainer-fold{fold}",
        per_device_train_batch_size=batch_size,
        learning_rate=cfg["lr"],
        num_train_epochs=cfg["epochs"],
        max_steps=max_steps if max_steps > 0 else -1,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps_for(len(examples), batch_size, cfg["epochs"], max_steps),
        bf16=True,
        logging_steps=5,
        save_strategy="none",
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        seed=seed + max(fold, 0),
    )
    trainer = Trainer(model=model, args=args, train_dataset=examples, data_collator=collate)
    trainer.train()

    @torch.no_grad()
    def score(prompts: list[list[int]]) -> list[float]:
        """P(yes) from the logits at the last prompt token (right-padded, gathered by length)."""
        model.eval()
        probs: list[float] = []
        size = cfg["eval_batch_size"]
        for start in range(0, len(prompts), size):
            chunk = prompts[start : start + size]
            input_ids = pad_batch(chunk, pad_id).to("cuda")
            attention_mask = pad_batch([[1] * len(c) for c in chunk], 0).to("cuda")
            extra = {k: torch.zeros_like(input_ids) for k in extra_zero_keys}
            logits = model(input_ids=input_ids, attention_mask=attention_mask, **extra).logits
            last = torch.tensor([len(c) - 1 for c in chunk], device=logits.device)
            at_answer = logits[torch.arange(len(chunk), device=logits.device), last]
            pair = at_answer[:, label_ids].float()
            probs.extend(torch.softmax(pair, dim=-1)[:, 1].cpu().tolist())
        return probs

    val_ids = val_df["PassengerId"].astype(int).tolist()
    val_probs = score(encode_frame(val_df)) if len(val_df) else []
    val_accuracy = accuracy(val_df["Survived"], val_probs) if len(val_df) else None
    test_ids = test_df["PassengerId"].astype(int).tolist()
    test_probs = score(encode_frame(test_df))
    print(f"[fold {fold}] val_accuracy={val_accuracy}", flush=True)

    fold_dir = Path(RUNS_DIR) / cfg["run_name"] / ("full" if fold < 0 else f"fold{fold}")
    fold_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(fold_dir)
    tokenizer.save_pretrained(fold_dir)
    runs_volume.commit()

    return {
        "fold": fold,
        "val_ids": val_ids,
        "val_probs": val_probs,
        "test_ids": test_ids,
        "test_probs": test_probs,
        "val_accuracy": val_accuracy,
    }


# --------------------------------------------------------------------------- #
# Local entrypoint
# --------------------------------------------------------------------------- #


@app.local_entrypoint()
def main(
    data_dir: str = "data",
    model_id: str = DEFAULT_MODEL_ID,
    folds: int = DEFAULT_FOLDS,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    gpu: str = DEFAULT_GPU,
    run_name: str = "",
    smoke: bool = False,
    seed: int = DEFAULT_SEED,
):
    start = time.time()
    data_path = Path(data_dir)
    train_text = (data_path / "train.csv").read_text(encoding="utf-8")
    test_text = (data_path / "test.csv").read_text(encoding="utf-8")

    max_steps = -1
    if smoke:
        train_df = pd.read_csv(io.StringIO(train_text)).sample(n=SMOKE_ROWS, random_state=seed)
        train_text = train_df.to_csv(index=False)
        folds, epochs, max_steps = SMOKE_FOLDS, 1, SMOKE_MAX_STEPS
        print(f"SMOKE: {SMOKE_ROWS} rows, {folds} folds, {epochs} epoch, max {max_steps} steps")

    run_name = run_name or default_run_name(model_id, smoke=smoke)
    cfg = make_cfg(model_id=model_id, run_name=run_name, epochs=epochs, lr=lr, max_steps=max_steps)
    train_df = pd.read_csv(io.StringIO(train_text))
    test_df = pd.read_csv(io.StringIO(test_text))
    print(f"Run {run_name}: {model_id} on {gpu}, {folds} folds, {epochs} epochs, lr {lr}")

    weights_path = prefetch_model.remote(model_id)
    print(f"Model weights cached at {weights_path}")

    arguments = [(train_text, test_text, fold, folds, seed, cfg) for fold in range(folds)]
    results = list(train_fold.with_options(gpu=gpu).starmap(arguments))

    oof = assemble_oof(results, train_df["PassengerId"])
    test_probs = average_test_probs(results, test_df["PassengerId"])
    metrics = build_metrics(
        run_name=run_name,
        model_id=model_id,
        gpu=gpu,
        folds=folds,
        epochs=epochs,
        lr=lr,
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        seed=seed,
        smoke=smoke,
        oof=oof,
        y_true=train_df.set_index("PassengerId").loc[oof["PassengerId"], "Survived"],
        results=results,
        n_test=len(test_df),
        elapsed_seconds=time.time() - start,
    )

    out_dir = Path("outputs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out_dir / "oof.csv", index=False)
    test_probs.to_csv(out_dir / "test_probs.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nFold accuracies: {[round(a, 4) for a in metrics['fold_accuracies']]}")
    print(f"OOF accuracy: {metrics['oof_accuracy']:.4f}")
    print(f"Wrote {out_dir}/")
    print("\nNext:")
    print("  make ensemble          # or: uv run python ensemble.py")
    print("  make submit MSG=\"gemma4 lora\"   # or: bash scripts/submit.sh")
