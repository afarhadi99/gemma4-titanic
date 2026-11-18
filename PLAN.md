# Plan — Kaggle Titanic with Gemma 4 fine-tuned on Modal

Competition: https://www.kaggle.com/c/titanic — predict `Survived` (0/1) for 418 test passengers.
Metric: accuracy. Submission: CSV `PassengerId,Survived`, 418 rows + header.
Data: `train.csv` (891 rows, labelled), `test.csv` (418), `gender_submission.csv` (sample).

## Where it runs

Primary dev environment: **GitHub Codespaces** (Linux, bash, Python 3.12 devcontainer). The
codespace only orchestrates — all GPU work runs on Modal. All configuration comes from **one
`.env` file** in the repo root (copied from the committed `.env.example` by the post-create
script; gitignored, never committed): `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` (read lazily by the
Modal client), `HF_TOKEN` (forwarded to containers via `modal.Secret.from_dict`, the only value
that leaves the machine), `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` + `KAGGLE_KEY`), plus optional
training defaults (`GEMMA_MODEL_ID`, `MODAL_GPU`, `FOLDS`, `EPOCHS`, `LR`, `SEED`). No Codespaces
secrets, no Modal named secret, no `~/.kaggle` files. `modal_app.py` loads `.env` itself via
`titanic.env.load_env`; the Makefile and scripts also load it (`set -a; . ./.env` in bash,
`Import-DotEnv` in PowerShell). `make check` verifies keys and Modal/Kaggle auth. Bash scripts and a
`Makefile` are the primary interface; `.ps1` scripts remain for local Windows use.

## Strategy

Titanic is a tiny tabular problem. A fine-tuned LLM is not obviously better than gradient
boosting here, so the repo does three things and lets out-of-fold (OOF) accuracy decide:

1. **Classical baseline** (local, free, seconds): engineered features → logistic regression +
   gradient boosting, 5-fold CV. Expected public LB ≈ 0.77–0.80.
2. **Gemma 4 LoRA classifier** (Modal GPU): each passenger serialized as a short English
   description; the model is trained to answer `yes`/`no` to "Did this passenger survive?".
   Prediction = softmax over the logits of the `yes` / `no` tokens at the answer position
   (no free-text generation, so outputs are calibrated probabilities). K-fold training gives
   OOF probabilities (honest accuracy) and K test predictions that are averaged.
3. **Ensemble** (local): grid-search the blend weight and threshold on OOF probabilities,
   apply to test probabilities, write `submissions/submission.csv`, validate, submit.

## Key facts from research (2026-11-17)

- Gemma 4 on HF (Apache 2.0): `google/gemma-4-E2B-it` (2.3B eff.), `-E4B-it`, `-12B-it`,
  `-26B-A4B-it` (MoE), `-31B-it`. Default: **`google/gemma-4-E2B-it`**.
- Load with `AutoModelForCausalLM` first (text-only `Gemma4ForCausalLM`); fall back to
  `AutoModelForMultimodalLM` if that fails. `attn_implementation="sdpa"`, `dtype=torch.bfloat16`.
- Gotchas: PEFT rejects `Gemma4ClippableLinear` (vision/audio towers) when `target_modules` is
  a plain name list → build an explicit list of full module names: only `torch.nn.Linear`,
  only inside the language model, leaf name in {q,k,v,o,gate,up,down}_proj (or `.linear`
  child of those). TRL `SFTTrainer` forces `use_cache=False` and stalls → use plain
  `transformers.Trainer` with our own collator. Avoid DeepSpeed ZeRO-3.
- Library floors: transformers ≥ 5.5.2, peft ≥ 0.19.0 (needs torch ≥ 2.7), accelerate ≥ 1.12.
- Modal SDK 1.6.4. `modal.Image.debian_slim(python_version="3.12").uv_pip_install(...)`,
  `.add_local_python_source("titanic")`, `modal.Volume.from_name(..., create_if_missing=True)`,
  `vol.commit()`, `modal.Secret.from_dict({"HF_TOKEN": ...})`, `Function.with_options(gpu=...)`.
  $30/month free credit. Default GPU **L40S** (48 GB, ≈ $1.95/h); E2B bf16 LoRA fits easily.
- Kaggle CLI (current): competition is positional — `kaggle competitions download titanic`,
  `kaggle competitions submit titanic -f FILE -m MSG`. Must click "Join Competition" first.

## Repo layout and owners

```
README.md                    step-by-step directions                       (docs pass)
PLAN.md                      this file                                     (Opus)
pyproject.toml               pyproject.toml (hatchling) for `-e .`             (Agent A)
requirements.txt             pinned deps, installed with plain pip         (Agent A)
.gitignore                   data/, outputs/, submissions/, .venv, .env    (Agent A)
.env.example                 template for the single .env config file      (Agent A)
src/titanic/env.py           dependency-free .env loader                   (Agent A)
src/titanic/__init__.py                                                    (Agent A)
src/titanic/features.py      load + feature engineering                    (Agent A)
src/titanic/text.py          row → prompt text, prompt constants           (Agent A)
src/titanic/submission.py    write + validate submission                   (Agent A)
src/titanic/lora.py          pure LoRA target-module selection             (Agent B)
baseline.py                  classical CV models → outputs/baseline/       (Agent A)
ensemble.py                  blend outputs/*/ → submissions/submission.csv (Agent A)
modal_app.py                 Modal training + inference                    (Agent B)
scripts/download_data.sh    kaggle download + unzip into data/ (+ .ps1)   (Agent A)
scripts/submit.sh            validate + kaggle submit (+ .ps1)             (Agent A)
scripts/check_env.sh         check .env keys and Modal/Kaggle auth         (Agent A)
.devcontainer/               Codespaces: devcontainer.json + post-create.sh (Agent A)
Makefile                     setup/env/check/data/test/baseline/smoke/train/ensemble/submit (Agent A)
.gitattributes               LF endings for *.sh and Makefile              (Agent A)
tests/test_features.py, test_text.py, test_submission.py, test_ensemble.py (Agent A)
tests/test_lora.py                                                         (Agent B)
tests/fixtures/train.csv, test.csv  ~20 / ~10 synthetic rows, real schema  (Agent A)
```

## Interfaces (both agents code against these exactly)

`src/titanic/features.py`
- `load_raw(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]` — reads `train.csv`, `test.csv`.
- `engineer(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]` —
  fits imputations on train+test features jointly (no label use), returns copies with added
  columns: `Title` (Mr/Mrs/Miss/Master/Rare), `FamilySize`, `IsAlone`, `Deck` (letter or "U"),
  `TicketGroupSize`, `FarePerPerson`, `AgeMissing` (bool), `Age` imputed by Title median,
  `Fare` and `Embarked` filled. Original columns kept.
- `NUMERIC_COLS`, `CATEGORICAL_COLS` — lists used by the baseline.

`src/titanic/text.py`
- `SYSTEM_PROMPT: str` — one sentence: task is to predict survival, answer yes or no.
- `QUESTION: str` — "Did this passenger survive the sinking? Answer yes or no."
- `LABEL_WORDS: tuple[str, str] = ("no", "yes")` — index == label.
- `describe(row: pd.Series) -> str` — deterministic natural-language description from the
  *engineered* row, e.g. "A 22-year-old man (title Mr) travelling in 3rd class, embarked at
  Southampton, paid 7.25 per person, with 1 sibling/spouse and 0 parents/children aboard
  (family of 2). Cabin deck unknown. Ticket shared by 1 passenger." Never includes `Survived`
  or `PassengerId`. Says "age unknown (estimated N)" when `AgeMissing`.

`src/titanic/submission.py`
- `write_submission(passenger_ids, survived, path) -> Path` — ints 0/1, header `PassengerId,Survived`.
- `validate_submission(path, test_csv) -> None` — raises `ValueError` with a clear message on
  wrong header, row count ≠ test rows, missing/duplicate/extra ids, non-0/1 values.

`src/titanic/lora.py` (pure Python, no torch import)
- `LANGUAGE_PROJ_NAMES = ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")`
- `select_lora_targets(modules: Iterable[tuple[str, str]]) -> list[str]` — input pairs of
  (full module name, class name). Keep a name iff class name is `"Linear"`, the name has no
  segment matching vision/audio/multi_modal/embed (case-insensitive), and either the last
  segment is in `LANGUAGE_PROJ_NAMES` or the last segment is `linear` and the previous one
  is. Raises `ValueError` if the result is empty. Sorted, deduplicated.

Run artifacts — every model writes to `outputs/<run_name>/`:
- `oof.csv`: `PassengerId,prob` for all 891 train rows (out-of-fold P(survived)).
- `test_probs.csv`: `PassengerId,prob` for 418 test rows.
- `metrics.json`: `{"model": ..., "folds": K, "oof_accuracy": float, "fold_accuracies": [...], ...}`.

## modal_app.py design (Agent B)

- `app = modal.App("titanic-gemma4")`; image: debian_slim py3.12, `uv_pip_install` of
  torch≥2.7 (CUDA build), transformers≥5.5.2,<6, peft≥0.19.0, accelerate≥1.12, pandas,
  scikit-learn, numpy; `.add_local_python_source("titanic")`.
- Volumes: `hf-cache` at `/root/.cache/huggingface` (weights persist across runs),
  `titanic-gemma4-runs` at `/runs`. Secret: `modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})`.
- Data is read locally by the entrypoint and passed as CSV text arguments (tiny), so no upload step.
- `train_fold(train_csv, test_csv, fold, folds, seed, cfg) -> dict` (GPU fn): engineer
  features, split with `StratifiedKFold(folds, shuffle=True, random_state=seed)` (fold = -1
  means train on all rows, no OOF), build chat-template prompts
  `[system, user=describe(row)+"\n\n"+QUESTION]` with `add_generation_prompt=True`, append the
  label word token; loss only on the label token (labels = -100 elsewhere). Assert the label
  words each encode to exactly one token after the generation prompt (else use their first token
  and log a warning). LoRA r=16, alpha=32, dropout=0.05, targets from
  `select_lora_targets([(n, type(m).__name__) for n, m in model.named_modules()])`.
  Plain `Trainer`: bf16, lr 1e-4, epochs 3, batch 16, cosine, warmup 5%, gradient
  checkpointing off by default, `use_cache` untouched, `report_to=[]`. Scoring: one forward
  pass per batch, logits at the last prompt position, softmax over the two label token ids.
  Returns `{"fold", "val_ids", "val_probs", "test_ids", "test_probs", "val_accuracy"}` and saves the
  adapter to `/runs/<run_name>/fold<k>/` then `vol.commit()`.
- `@app.local_entrypoint() main(data_dir="data", model_id, folds, epochs, lr, gpu, run_name,
  smoke=False, seed)`: defaults come from `GEMMA_MODEL_ID` (google/gemma-4-E2B-it), `FOLDS` (5),
  `EPOCHS` (3), `LR` (1e-4), `MODAL_GPU` (L40S), `SEED` (42), overridable by CLI flags; reads CSVs, fans out
  folds in parallel with `train_fold.with_options(gpu=gpu).starmap(...)` / `.map`, assembles
  OOF + averaged test probs, writes `outputs/<run_name>/{oof.csv,test_probs.csv,metrics.json}`
  locally, prints OOF accuracy. `smoke=True`: 64 train rows, 2 folds, 1 epoch, ≤ 10 steps —
  a cheap end-to-end check (≈ $0.10).

## Verification

- Local: `python -m pytest` green; `python -m py_compile modal_app.py`;
  `python baseline.py --data-dir tests/fixtures` and `ensemble.py` run on fixtures.
- Cloud (needs owner's go-ahead, spends Modal credit): `modal run modal_app.py --smoke`
  then a full 5-fold run (≈ 10–20 min on L40S, well under $2).

## Risks

- Gemma 4 module paths/class names may differ from research → `select_lora_targets` raises
  loudly; smoke run catches it before a full run.
- LLM may not beat the baseline on 891 rows; the ensemble only uses it if OOF says it helps.
- Public LB is a 50% split of test; trust OOF over LB moves of ±1%.
