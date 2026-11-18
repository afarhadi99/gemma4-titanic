"""Fine-tune a small open LLM (default Gemma 4) with LoRA on Kaggle Titanic, on Modal GPUs.

Each passenger is described in English; the model answers "yes"/"no" and the
prediction is the softmax over those two label-token logits. K-fold training gives
out-of-fold (OOF) probabilities plus K test predictions that are averaged.

Configuration comes from a single .env file in the repo root (see .env.example):
MODAL_TOKEN_ID / MODAL_TOKEN_SECRET authenticate the local Modal client and HF_TOKEN is
forwarded to the GPU containers. Optional defaults: GEMMA_MODEL_ID, MODAL_GPU, FOLDS,
EPOCHS, LR, SEED, TRUST_REMOTE_CODE. Command-line flags override them.

Usage (run from the repo root, data in ./data):
    python -m modal run modal_app.py --smoke
    python -m modal run modal_app.py
    python -m modal run modal_app.py --model-id google/gemma-4-E4B-it --gpu A100-80GB --epochs 2
    python -m modal run modal_app.py --model-id openbmb/MiniCPM5-2B

Optimisation flags (all recorded in metrics.json): --seed (training randomness) vs --split-seed
(StratifiedKFold seed, fixed at 42 so seed-averaged runs share folds), --head lm|classification,
--prompt-style prose|structured, --lora-r / --lora-alpha (alpha defaults to 2*r),
--save-embeddings / --no-save-embeddings.

Results land in outputs/<run-name>/{oof.csv,test_probs.csv,metrics.json}; adapters are
saved to the `titanic-gemma4-runs` Modal volume under /runs/<run-name>/fold<k>/.

Embeddings (default on): each fold model embeds its own VAL rows and all TEST rows (last-layer
hidden state at the final prompt token; the pooled last-token state for --head classification).
emb_oof.npy (891, H) holds each train row's embedding from the fold that never saw its label, in
train PassengerId order; emb_test.npy (418, H) is the mean over folds; emb_ids.json lists the
train/test id orders. They are an OOF-stacking input for the classical models: only use emb_oof
as features inside the same CV that produced it, never fit a model on emb_oof and score
emb_test from a different split of the labels.
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
from titanic.llm import (  # noqa: F401  (re-exported for tests)
    MODEL_CLASS_NAMES,
    attn_candidates,
    chat_prompt,
    resolve_trust_remote_code,
    is_attention_error,
    last_token_index,
    HEAD_NAME,
    load_causal_lm,
    load_sequence_classifier,
    pick_label_tokens,
)

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
DEFAULT_SPLIT_SEED = 42
HEADS = ("lm", "classification")
DEFAULT_TRUST_REMOTE_CODE = (os.environ.get("TRUST_REMOTE_CODE") or "").lower() in ("1", "true", "yes")
SMOKE_ROWS = 64
SMOKE_FOLDS = 2
SMOKE_MAX_STEPS = 10

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch>=2.7",
        "transformers>=5.6,<6",  # tested with 5.23.0; MiniCPM5 needs >=5.6
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
    """'google/gemma-4-E2B-it' -> 'e2b'; 'Qwen/Qwen3.5-2B' -> 'qwen3.5-2b';
    'openbmb/MiniCPM5-2B' -> 'minicpm5-2b'; 'HuggingFaceTB/SmolLM3-3B' -> 'smollm3-3b'."""
    last = model_id.rsplit("/", 1)[-1].lower()
    parts = last.split("-")
    if len(parts) >= 3 and parts[0] == "gemma":
        core = [p for p in parts[2:] if p not in ("it", "pt")]
        if core:
            return "-".join(core)
    cleaned = [p for p in parts if p not in ("it", "instruct", "chat")]
    return "-".join(cleaned) or last


def default_run_name(model_id: str, now: datetime | None = None, smoke: bool = False) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    suffix = "-smoke" if smoke else ""
    family = "gemma4" if "gemma-4" in model_id.lower() else "lora"
    return f"{family}-{short_model_name(model_id)}-{stamp}{suffix}"


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
    split_seed: int = DEFAULT_SPLIT_SEED,
    head: str = "lm",
    prompt_style: str = "prose",
    save_embeddings: bool = False,
    embedding_dim: int | None = None,
) -> dict:
    ordered = sorted(results, key=lambda r: r["fold"])
    return {
        "model": f"llm-lora:{model_id}",
        "run_name": run_name,
        "model_id": model_id,
        "gpu": gpu,
        "folds": folds,
        "epochs": epochs,
        "lr": lr,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "seed": seed,
        "split_seed": split_seed,
        "head": head,
        "prompt_style": prompt_style,
        "save_embeddings": save_embeddings,
        "embedding_dim": embedding_dim,
        "smoke": smoke,
        "n_train": int(len(oof)),
        "n_test": int(n_test),
        "oof_accuracy": accuracy(y_true, oof["prob"]),
        "fold_accuracies": [r["val_accuracy"] for r in ordered],
        "elapsed_seconds": round(elapsed_seconds, 1),
    }


def assemble_embeddings(results: list[dict], train_ids, test_ids) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build emb_oof (train order, each row from the fold that held it out) and emb_test
    (mean over folds) as float32 arrays, plus the id orders."""
    train_order = [int(i) for i in train_ids]
    test_order = [int(i) for i in test_ids]
    position = {pid: k for k, pid in enumerate(train_order)}
    emb_oof = None
    seen: set[int] = set()
    test_sum = None
    for result in results:
        val = np.asarray(result["val_emb"], dtype=np.float32)
        ids = [int(i) for i in result["val_ids"]]
        if len(val) != len(ids):
            raise ValueError("Fold val embeddings do not match val ids")
        if emb_oof is None:
            emb_oof = np.full((len(train_order), val.shape[1]), np.nan, dtype=np.float32)
        for pid, vec in zip(ids, val):
            if pid in seen or pid not in position:
                raise ValueError(f"Passenger {pid} is duplicated or not a train id")
            seen.add(pid)
            emb_oof[position[pid]] = vec
        if [int(i) for i in result["test_ids"]] != test_order:
            raise ValueError("Fold test ids do not match the test set order")
        test = np.asarray(result["test_emb"], dtype=np.float32)
        test_sum = test if test_sum is None else test_sum + test
    if emb_oof is None or len(seen) != len(train_order):
        raise ValueError("Embeddings do not cover every train row")
    return emb_oof, test_sum / len(results), {"train_ids": train_order, "test_ids": test_order}


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
    trust_remote_code: bool = False,
    split_seed: int = DEFAULT_SPLIT_SEED,
    head: str = "lm",
    prompt_style: str = "prose",
    save_embeddings: bool = True,
) -> dict:
    return {
        "split_seed": split_seed,
        "head": head,
        "prompt_style": prompt_style,
        "save_embeddings": save_embeddings,
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
        "trust_remote_code": trust_remote_code,
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


def fit_and_score(
    model,
    tokenizer,
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
    device: str,
    bf16: bool,
    fold: int = 0,
    seed: int = 42,
    save_dir: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """LoRA-fine-tune an already-loaded model on `fit_df`, then score `val_df` and `test_df`."""
    import inspect
    import tempfile

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    from titanic.lora import select_lora_targets
    from titanic.text import PROMPT_STYLES, QUESTION, SYSTEM_PROMPT

    describe = PROMPT_STYLES[cfg.get("prompt_style", "prose")]
    head = cfg.get("head", "lm")
    classify = head == "classification"
    want_emb = cfg.get("save_embeddings", True)

    output_dir = output_dir or tempfile.mkdtemp(prefix=f"trainer-fold{fold}-")

    label_words, label_ids = pick_label_tokens(lambda w: tokenizer.encode(w, add_special_tokens=False))
    print(f"Label tokens {dict(zip(label_words, label_ids))}", flush=True)

    def prompt_ids(row: pd.Series) -> list[int]:
        messages = build_messages(describe(row), SYSTEM_PROMPT, QUESTION)
        text = chat_prompt(tokenizer, messages)
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def encode_frame(df: pd.DataFrame) -> list[list[int]]:
        return [prompt_ids(row) for _, row in df.iterrows()]

    fit_prompts = encode_frame(fit_df)
    print(f"Example prompt:\n{tokenizer.decode(fit_prompts[0])}", flush=True)
    fit_labels = fit_df["Survived"].astype(int).tolist()
    if classify:  # plain cross-entropy over 2 classes at the last prompt token
        examples = [{"input_ids": ids, "labels": label} for ids, label in zip(fit_prompts, fit_labels)]
    else:
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
            "labels": (
                torch.tensor([b["labels"] for b in batch], dtype=torch.long)
                if classify
                else pad_batch([b["labels"] for b in batch], -100)
            ),
        }
        for key in extra_zero_keys:
            out[key] = torch.zeros_like(input_ids)
        return out

    targets = select_lora_targets([(n, type(m).__name__) for n, m in model.named_modules()])
    print(f"LoRA targets: {len(targets)}; examples: {targets[:3]}", flush=True)
    native_cls = classify and type(model).__name__.endswith("ForSequenceClassification")
    wrapper_cls = classify and not native_cls  # llm.make_last_token_classifier
    lora_kwargs = {}
    if classify:
        # Keep the 2-way head fully trainable and saved with the adapter.
        lora_kwargs["modules_to_save"] = [HEAD_NAME]
    if not classify:
        lora_kwargs["task_type"] = "CAUSAL_LM"
    elif native_cls:
        lora_kwargs["task_type"] = "SEQ_CLS"
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = pad_id
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=targets,
        bias="none",
        **lora_kwargs,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    batch_size = cfg["batch_size"]
    max_steps = cfg["max_steps"]
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=cfg["lr"],
        num_train_epochs=cfg["epochs"],
        max_steps=max_steps if max_steps > 0 else -1,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps_for(len(examples), batch_size, cfg["epochs"], max_steps),
        bf16=bf16,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        use_cpu=(device == "cpu"),
        remove_unused_columns=False,
        label_names=["labels"],
        seed=seed + max(fold, 0),
    )
    trainer = Trainer(model=model, args=args, train_dataset=examples, data_collator=collate)
    trainer.train()

    @torch.no_grad()
    def score(prompts: list[list[int]]) -> tuple[list[float], np.ndarray | None]:
        """P(yes) (or softmax[:, 1] for the classification head) at the last prompt token, plus
        that token's last-layer hidden state as float16 (right-padded, gathered by length)."""
        model.eval()
        probs: list[float] = []
        embs: list[np.ndarray] = []
        size = cfg["eval_batch_size"]
        for start in range(0, len(prompts), size):
            chunk = prompts[start : start + size]
            input_ids = pad_batch(chunk, pad_id).to(device)
            attention_mask = pad_batch([[1] * len(c) for c in chunk], 0).to(device)
            extra = {k: torch.zeros_like(input_ids) for k in extra_zero_keys}
            if wrapper_cls:
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                pair, hidden = out["logits"].float(), out["pooled"]
            else:
                out = model(
                    input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=want_emb, **extra
                )
                last = last_token_index(attention_mask)
                rows = torch.arange(len(chunk), device=out.logits.device)
                if native_cls:  # logits are already pooled at the last non-pad token: (B, 2)
                    pair = out.logits.float()
                else:
                    at_answer = out.logits[rows, last]
                    pair = at_answer[:, label_ids].float()
                hidden = out.hidden_states[-1][rows, last] if want_emb else None
            probs.extend(torch.softmax(pair, dim=-1)[:, 1].cpu().tolist())
            if want_emb:
                embs.append(hidden.float().cpu().numpy().astype(np.float16))
        return probs, (np.concatenate(embs) if embs else np.zeros((0, 0), dtype=np.float16))

    val_ids = val_df["PassengerId"].astype(int).tolist()
    val_probs, val_emb = score(encode_frame(val_df)) if len(val_df) else ([], None)
    val_accuracy = accuracy(val_df["Survived"], val_probs) if len(val_df) else None
    test_ids = test_df["PassengerId"].astype(int).tolist()
    test_probs, test_emb = score(encode_frame(test_df))
    print(f"[fold {fold}] val_accuracy={val_accuracy}", flush=True)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    return {
        "fold": fold,
        "val_ids": val_ids,
        "val_probs": val_probs,
        "test_ids": test_ids,
        "test_probs": test_probs,
        "val_accuracy": val_accuracy,
        "val_emb": val_emb,
        "test_emb": test_emb,
    }


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
    import torch
    import transformers
    from sklearn.model_selection import StratifiedKFold
    from transformers import AutoTokenizer

    from titanic.features import add_group_features, engineer
    hf_volume.reload()
    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN is empty (set it in .env); continuing without it")
    transformers.set_seed(seed + max(fold, 0))
    train_raw = pd.read_csv(io.StringIO(train_csv))
    test_raw = pd.read_csv(io.StringIO(test_csv))
    train_df, test_df = engineer(train_raw, test_raw)

    if fold >= 0:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.get("split_seed", DEFAULT_SPLIT_SEED))
        tr_idx, va_idx = list(splitter.split(train_df, train_df["Survived"]))[fold]
    else:
        tr_idx, va_idx = np.arange(len(train_df)), np.array([], dtype=int)
    # Group survival statistics come from the fit rows' labels only (leave-one-out for fit
    # rows), exactly as in baseline.py; fold -1 uses every train label for the test rows.
    fit_df, (val_df, test_df) = add_group_features(
        train_df.iloc[tr_idx], [train_df.iloc[va_idx], test_df]
    )
    print(f"[fold {fold}] fit={len(fit_df)} val={len(val_df)} test={len(test_df)}", flush=True)

    model_id = cfg["model_id"]
    trust = cfg.get("trust_remote_code", False)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
    except Exception as exc:
        print(f"AutoTokenizer failed ({exc!r}); falling back to AutoProcessor.tokenizer")
        tokenizer = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=trust).tokenizer

    if cfg.get("head", "lm") == "classification":
        model, _, _ = load_sequence_classifier(model_id, dtype=torch.bfloat16, trust_remote_code=trust)
    else:
        model, _ = load_causal_lm(model_id, dtype=torch.bfloat16, trust_remote_code=trust)
    model.to("cuda")
    print(f"Loaded {type(model).__name__}", flush=True)

    fold_dir = Path(RUNS_DIR) / cfg["run_name"] / ("full" if fold < 0 else f"fold{fold}")
    result = fit_and_score(
        model,
        tokenizer,
        fit_df,
        val_df,
        test_df,
        cfg,
        device="cuda",
        bf16=True,
        fold=fold,
        seed=seed,
        save_dir=str(fold_dir),
    )
    runs_volume.commit()
    return result


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
    trust_remote_code: bool = DEFAULT_TRUST_REMOTE_CODE,
    split_seed: int = DEFAULT_SPLIT_SEED,
    head: str = "lm",
    prompt_style: str = "prose",
    lora_r: int = 16,
    lora_alpha: int = 0,
    save_embeddings: bool = True,
):
    start = time.time()
    if head not in HEADS:
        raise SystemExit(f"--head must be one of {HEADS}, got {head!r}")
    from titanic.text import PROMPT_STYLES

    if prompt_style not in PROMPT_STYLES:
        raise SystemExit(f"--prompt-style must be one of {tuple(PROMPT_STYLES)}, got {prompt_style!r}")
    lora_alpha = lora_alpha or 2 * lora_r
    data_path = Path(data_dir)
    train_text = (data_path / "train.csv").read_text(encoding="utf-8")
    test_text = (data_path / "test.csv").read_text(encoding="utf-8")

    max_steps = -1
    if smoke:
        train_df = pd.read_csv(io.StringIO(train_text)).sample(n=SMOKE_ROWS, random_state=seed)
        train_text = train_df.to_csv(index=False)
        folds, epochs, max_steps = SMOKE_FOLDS, 1, SMOKE_MAX_STEPS
        print(f"SMOKE: {SMOKE_ROWS} rows, {folds} folds, {epochs} epoch, max {max_steps} steps")

    trust_remote_code = resolve_trust_remote_code(model_id, trust_remote_code)
    run_name = run_name or default_run_name(model_id, smoke=smoke)
    cfg = make_cfg(
        model_id=model_id,
        run_name=run_name,
        epochs=epochs,
        lr=lr,
        max_steps=max_steps,
        trust_remote_code=trust_remote_code,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        split_seed=split_seed,
        head=head,
        prompt_style=prompt_style,
        save_embeddings=save_embeddings,
    )
    train_df = pd.read_csv(io.StringIO(train_text))
    test_df = pd.read_csv(io.StringIO(test_text))
    print(f"Run {run_name}: {model_id} on {gpu}, {folds} folds, {epochs} epochs, lr {lr}, "
          f"lora r={lora_r}/alpha={lora_alpha}, head={head}, prompt={prompt_style}, seed={seed}, split_seed={split_seed}")

    weights_path = prefetch_model.remote(model_id)
    print(f"Model weights cached at {weights_path}")

    arguments = [(train_text, test_text, fold, folds, seed, cfg) for fold in range(folds)]
    results = list(train_fold.with_options(gpu=gpu).starmap(arguments))

    oof = assemble_oof(results, train_df["PassengerId"])
    test_probs = average_test_probs(results, test_df["PassengerId"])
    emb_arrays = None
    embedding_dim = None
    if save_embeddings:
        emb_arrays = assemble_embeddings(results, train_df["PassengerId"], test_df["PassengerId"])
        embedding_dim = int(emb_arrays[0].shape[1])
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
        split_seed=split_seed,
        head=head,
        prompt_style=prompt_style,
        save_embeddings=save_embeddings,
        embedding_dim=embedding_dim,
    )

    out_dir = Path("outputs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out_dir / "oof.csv", index=False)
    test_probs.to_csv(out_dir / "test_probs.csv", index=False)
    if emb_arrays is not None:
        np.save(out_dir / "emb_oof.npy", emb_arrays[0])
        np.save(out_dir / "emb_test.npy", emb_arrays[1])
        (out_dir / "emb_ids.json").write_text(json.dumps(emb_arrays[2]), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nFold accuracies: {[round(a, 4) for a in metrics['fold_accuracies']]}")
    print(f"OOF accuracy: {metrics['oof_accuracy']:.4f}")
    print(f"Wrote {out_dir}/")
    print("\nNext:")
    print("  make ensemble          # or: python ensemble.py")
    print("  make submit MSG=\"gemma4 lora\"   # or: bash scripts/submit.sh")
