# gemma4-titanic

Kaggle [Titanic](https://www.kaggle.com/c/titanic) with a classical baseline and a Gemma 4 model fine-tuned with LoRA on [Modal](https://modal.com) GPUs. Out-of-fold (OOF) accuracy decides which models go into the final submission.

- **Baseline**: engineered features, logistic regression and gradient boosting, 5-fold CV. Runs locally in seconds.
- **Gemma 4 LoRA classifier**: each passenger becomes a short English description and the model answers `yes`/`no`. The prediction is a softmax over the two label-token logits, trained K-fold on Modal.
- **Ensemble**: grid-search blend weights and a threshold on OOF probabilities, then write and validate `submissions/submission.csv`.

Design details are in [PLAN.md](PLAN.md).

## Before you start

- [ ] **Kaggle**: account, then open https://www.kaggle.com/c/titanic and click **Join Competition** (accept the rules).
- [ ] **Kaggle API token**: kaggle.com > Settings > API > Create New Token.
- [ ] **Modal**: account at https://modal.com ($30/month free credit) and a token (`modal token new` locally, or modal.com > Settings > API Tokens). This gives you `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.
- [ ] **Hugging Face**: account and a read token at https://huggingface.co/settings/tokens. Open https://huggingface.co/google/gemma-4-E2B-it and accept the license if prompted.

## Step 1: add Codespaces secrets

GitHub > Settings > Codespaces > Secrets (or repo-level), and give this repo access to each:

| Secret | Value |
| --- | --- |
| `MODAL_TOKEN_ID` | Modal token ID |
| `MODAL_TOKEN_SECRET` | Modal token secret |
| `HF_TOKEN` | Hugging Face read token |
| `KAGGLE_API_TOKEN` | Kaggle API token (alternative: `KAGGLE_USERNAME` + `KAGGLE_KEY`) |

Secrets added after a codespace exists only appear after you restart the codespace.

## Step 2: create the codespace

Code > Codespaces > **Create codespace on main**. The post-create script installs uv, runs `uv sync`, activates the venv in new shells and lists which secrets are set (never their values). Success: the terminal shows `MODAL_TOKEN_ID: set` and so on, with nothing `missing`.

## Step 3: connect Modal and Kaggle

```bash
make secrets
```

Checks Modal auth, creates or updates the Modal secret `huggingface-secret` from `$HF_TOKEN`, and checks Kaggle auth. Success: `Modal: ok`, `huggingface-secret: created/updated`, `Kaggle: ok`.

## Step 4: download the data

```bash
make data
```

Downloads `titanic.zip` into `data/` and unzips it. Success: `data/train.csv: 891 rows` and `data/test.csv: 418 rows`.

## Step 5: tests and the free baseline

```bash
make test
make baseline
```

`make test` runs pytest (all green). `make baseline` writes `outputs/baseline/{oof.csv,test_probs.csv,metrics.json}` and `submissions/baseline.csv`, and prints OOF accuracy per candidate model. Expect about 0.80 OOF; it takes seconds.

## Step 6: smoke test on Modal

```bash
make smoke
```

Cheap end-to-end check (about $0.10): 64 rows, 2 folds, 1 epoch. The first run downloads the model weights, which takes several minutes. In the log look for:

- `Loaded <ModelClass>`: Gemma 4 loaded.
- `Label token ids {'no': ..., 'yes': ...}`: the two label words are distinct single tokens.
- `LoRA targets: N; examples: [...]`: N is greater than 0.
- `OOF accuracy: ...` at the end and `Wrote outputs/<run>-smoke/`.

The smoke output is marked `"smoke": true` and `make ensemble` skips it automatically.

## Step 7: full training run

```bash
make train
```

5 folds in parallel on an L40S, about 10-20 minutes and roughly $1-3. Writes `outputs/gemma4-<model>-<timestamp>/{oof.csv,test_probs.csv,metrics.json}` and prints fold and OOF accuracy. To override the model, GPU or epochs:

```bash
uv run modal run modal_app.py --model-id google/gemma-4-E4B-it --gpu A100-80GB --epochs 2
```

Other options: `--folds`, `--lr`, `--run-name`, `--seed`, `--data-dir`.

## Step 8: ensemble

```bash
make ensemble
```

Blends every complete run in `outputs/` (baseline plus Gemma runs) and writes `submissions/submission.csv` and `submissions/ensemble.json`. It prints each run's OOF accuracy, the chosen weights and threshold, and the blended OOF accuracy. Runs that are smoke tests or do not cover all train/test PassengerIds are skipped with a `skipping <dir>: ...` message. To pick runs explicitly:

```bash
uv run python ensemble.py --runs outputs/baseline outputs/<gemma-run>
```

## Step 9: submit

```bash
make submit MSG="baseline + gemma4 e2b blend"
```

Validates the file against `data/test.csv`, submits it, then lists your recent submissions. Success: a submission with status `complete` and a public score. Check the leaderboard at https://www.kaggle.com/c/titanic/leaderboard. Public LB uses half the test set, so trust OOF over LB moves of about 1%.

## Running locally instead of Codespaces

```bash
uv sync
uv run modal setup                      # or set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
# Kaggle: put the token in ~/.kaggle/access_token, or the legacy ~/.kaggle/kaggle.json
export HF_TOKEN=hf_xxx && bash scripts/setup_secrets.sh
bash scripts/download_data.sh
```

The Makefile targets work the same on Linux/macOS. On Windows PowerShell use `scripts/download_data.ps1` and `scripts/submit.ps1 -File submissions/submission.csv -Message "..."`, and run the `uv run ...` commands from the Makefile directly.

## Tuning ideas

- Bigger model: `--model-id google/gemma-4-E4B-it` (or `-12B-it`) with a larger `--gpu`.
- Epochs and learning rate: `--epochs`, `--lr`.
- Prompt wording: `src/titanic/text.py` (`SYSTEM_PROMPT`, `QUESTION`, `describe`).
- More folds or other seeds: `--folds`, `--seed`, then ensemble the extra runs.

## Costs

| GPU | Approx. $/hour | Notes |
| --- | --- | --- |
| L40S (default) | about 1.95 | 48 GB, E2B bf16 LoRA fits easily |
| A100-80GB | check modal.com/pricing | for larger models |

A smoke run is about $0.10 and a full 5-fold E2B run roughly $1-3, against the $30/month free credit.

## Troubleshooting

- **Modal auth errors**: check `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` are set (`make setup` prints status). Secrets added after the codespace was created need a restart.
- **Kaggle 403**: you have not joined the competition. Click **Join Competition** and accept the rules.
- **Hugging Face 401/403**: accept the Gemma license on the model page, check `HF_TOKEN`, then re-run `make secrets`.
- **`select_lora_targets` raises ValueError**: the Gemma 4 module names differ from what is expected. Paste the module names from the log and adjust `src/titanic/lora.py`.
- **CUDA out of memory**: use a bigger `--gpu`, or reduce the batch size in `make_cfg` in `modal_app.py`.

## Repo layout

```
baseline.py            classical CV models -> outputs/baseline/
ensemble.py            blend outputs/*/ -> submissions/submission.csv
modal_app.py           Modal LoRA training and inference
src/titanic/           features, text prompts, submission I/O, LoRA target selection
scripts/               download_data, submit, setup_secrets (.sh primary, .ps1 for Windows)
tests/                 pytest suite and synthetic fixtures
.devcontainer/         Codespaces config and post-create script
Makefile               shortcuts for every step above
PLAN.md                design and decisions
```
