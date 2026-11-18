# gemma4-titanic

Kaggle [Titanic](https://www.kaggle.com/c/titanic) with a classical baseline and a Gemma 4 model fine-tuned with LoRA on [Modal](https://modal.com) GPUs. Out-of-fold (OOF) accuracy decides which models go into the final submission.

- **Baseline**: engineered features plus leak-free family/ticket group features; logistic regression, HGB, gradient boosting, random forest and SVC, with the best single model or soft-vote average picked by OOF. 5-fold CV, runs locally in seconds.
- **Gemma 4 LoRA classifier**: each passenger becomes a short English description and the model answers `yes`/`no`. The prediction is a softmax over the two label-token logits, trained K-fold on Modal.
- **Ensemble**: grid-search blend weights (step 0.1, threshold fixed at 0.5 unless `--tune-threshold`) on OOF probabilities, then write and validate `submissions/submission.csv`.

Design details are in [PLAN.md](PLAN.md).

## Step 1: open the codespace

Code > Codespaces > **Create codespace on main** (or clone the repo locally and run `bash .devcontainer/post-create.sh`, i.e. `make setup`). The post-create script creates `.venv`, upgrades pip, runs `python -m pip install -r requirements.txt`, activates the venv in new shells and creates `.env` from `.env.example` if it is missing. It then prints which keys in `.env` are `set` or `missing` (never the values).

## Step 2: fill in `.env`

All configuration lives in one file, `.env` in the repo root. Open it and fill in:

| Variable | Where to get it |
| --- | --- |
| `KAGGLE_API_TOKEN` | kaggle.com > Settings > API > **Create New Token**. Alternative: `KAGGLE_USERNAME` + `KAGGLE_KEY`. Also open https://www.kaggle.com/c/titanic and click **Join Competition** (accept the rules). |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | Modal dashboard > Settings > API Tokens (account at https://modal.com, $30/month free credit). |
| `HF_TOKEN` | Read token from https://huggingface.co/settings/tokens. Open https://huggingface.co/google/gemma-4-E2B-it and accept the license. |

`HF_TOKEN` is the only value forwarded to the Modal GPU containers. The Modal and Kaggle tokens stay on the machine running `make`.

**Warning:** `.env` holds live tokens. It is gitignored: never commit it or paste it anywhere. It exists only in this codespace (or on your machine), so deleting the codespace deletes it.

The commented-out "training defaults" block in `.env.example` (`GEMMA_MODEL_ID`, `MODAL_GPU`, `FOLDS`, `EPOCHS`, `LR`, `SEED`) sets the defaults for `make smoke` / `make train`. Uncomment a line to change it; command-line flags still override it.

## Step 3: check the setup

```bash
make check
```

Prints `set`/`missing` for each key, then verifies Modal and Kaggle auth. Success: all lines `ok` (`Modal: ok`, `Kaggle: ok`). On failure it prints the real Modal/Kaggle error under `details:`, and every error points back at `.env`.

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

`make test` runs pytest (all green). `make baseline` writes `outputs/baseline/{oof.csv,test_probs.csv,metrics.json}` and `submissions/baseline.csv`, and prints OOF accuracy per candidate model. It takes seconds. On the real data with group features the chosen model was `avg:gb+svc+rf` with OOF 0.8563 (see Results so far).

## Step 6: smoke test on Modal

```bash
make smoke
```

Cheap end-to-end check (cents): 64 rows, 2 folds, 1 epoch. The first run downloads the Gemma 4 E2B weights, which takes a few minutes. In the log look for:

- `Loaded with AutoModelForCausalLM + attn_implementation=sdpa(text,vision)+eager(audio)`: Gemma 4 loaded.
- `Label token ids {'no': 1904, 'yes': 4443}`: the two label words are distinct single tokens.
- `LoRA targets: 205` (`; examples: [...]` follows): N is greater than 0.
- `OOF accuracy: ...` at the end and `Wrote outputs/<run>-smoke/`.

The smoke OOF (about 0.64 on 64 rows) is meaningless. The output is marked `"smoke": true` and `make ensemble` skips it automatically.

## Step 7: full training run

```bash
make train
```

5 folds in parallel (the real run used `MODAL_GPU=A100` in `.env`): about 6 minutes wall clock on an A100, well under $2. A harmless `[modal-client] Timed out waiting for final app logs` line can appear at the end; the results are still written. Writes `outputs/gemma4-<model>-<timestamp>/{oof.csv,test_probs.csv,metrics.json}` and prints fold and OOF accuracy. To override the model, GPU or epochs (flags beat the `.env` defaults):

```bash
python -m modal run modal_app.py --model-id google/gemma-4-E4B-it --gpu A100-80GB --epochs 2
```

Other options: `--folds`, `--lr`, `--run-name`, `--seed`, `--data-dir`.

## Step 8: ensemble

```bash
make ensemble
```

Threshold is fixed at 0.5 unless you pass `--tune-threshold`; the weight grid step is 0.1. Blends every complete run in `outputs/` (baseline plus Gemma runs) and writes `submissions/submission.csv` and `submissions/ensemble.json`. It prints each run's OOF accuracy, the chosen weights and threshold, and the blended OOF accuracy. Runs that are smoke tests or do not cover all train/test PassengerIds are skipped with a `skipping <dir>: ...` message. To pick runs explicitly:

```bash
python ensemble.py --runs outputs/baseline outputs/<gemma-run>
```

## Step 9: submit

```bash
make submit MSG="baseline + gemma4 e2b blend"
```

Validates the file against `data/test.csv`, submits it, then lists your recent submissions. Success: a submission with status `complete` and a public score. Check the leaderboard at https://www.kaggle.com/c/titanic/leaderboard. Trust OOF over small public LB moves (see the note under Results so far).

## Results so far

Real end-to-end run in GitHub Codespaces (2026-11-17), Modal A100, transformers 5.23.0. OOF accuracy on the 891 training rows:

| Model | OOF accuracy |
| --- | --- |
| Baseline before group features | 0.8418 |
| Baseline logreg / hgb / gb / rf / svc (with group features) | 0.8406 / 0.8440 / 0.8485 / 0.8451 / 0.8462 |
| Baseline chosen: `avg:gb+svc+rf` (3-seed mean 0.8515) | 0.8563 |
| Gemma 4 E2B LoRA before group features (folds 0.832-0.865) | 0.8373 |
| Gemma 4 E2B LoRA with group features | 0.8496 |
| Ensemble (baseline + both Gemma runs) | 0.8563 (weights: baseline 1.0, Gemma 0.0) |

**Kaggle public leaderboard: 0.80143** for `submissions/submission.csv` (the baseline `avg:gb+svc+rf`, since the ensemble gave Gemma zero weight).

Group features lifted Gemma from 0.8373 to 0.8496 OOF, but it still trails the classical vote, so the blend keeps the baseline alone. To make Gemma earn weight, average 2-3 Gemma runs with different `--seed` values or try `--model-id google/gemma-4-E4B-it`.

OOF is the guide. The public leaderboard scores only about 209 passengers, so expect roughly 0.78-0.81 there; an honest 0.80+ is a strong score. Perfect 1.0 scores on the leaderboard come from looking up the public passenger list, not from modelling.

## How accuracy was improved

- **Group features** (`add_group_features` in `src/titanic/features.py`): passengers are grouped by shared ticket, else by surname + fare. For each passenger the group size and survival statistics of the other group members are computed from training-fold labels only (leave-one-out for training rows, so a row never sees its own label). The same facts are written into Gemma's prompt, so the LLM sees them too.
- **Model soft-vote**: `baseline.py` also scores averages of the best 2 and best 3 single models and keeps whichever has the best OOF; `ensemble.py` then blends the baseline with the Gemma runs.

## Running locally instead of Codespaces

Needs Python 3.11-3.13. Create a venv and install everything from `requirements.txt` with plain pip.

```bash
python -m venv .venv
source .venv/bin/activate          # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env               # PowerShell: Copy-Item .env.example .env; then fill it in
make check
make data
```

`make install` does the venv and install steps in one go.

The Makefile targets work the same on Linux/macOS. On Windows PowerShell use `scripts/download_data.ps1` and `scripts/submit.ps1 -File submissions/submission.csv -Message "..."` (both read `.env`), and run the Makefile commands directly with the venv Python (for example `python -m modal run modal_app.py --smoke`, `python baseline.py`). `modal_app.py` also loads `.env` itself when run locally.

## Tuning ideas

- Bigger model: `--model-id google/gemma-4-E4B-it` (or `-12B-it`) with a larger `--gpu`.
- Epochs and learning rate: `--epochs`, `--lr`.
- Prompt wording: `src/titanic/text.py` (`SYSTEM_PROMPT`, `QUESTION`, `describe`).
- More folds or other seeds: `--folds`, `--seed`, then ensemble the extra runs.
- Average 2-3 Gemma runs with different `--seed`.
- Try `--model-id google/gemma-4-E4B-it`.
- Other models (same prompt, folds, epochs, lr and LoRA settings; only attention, label tokens, LoRA target names and chat-template kwargs adapt automatically):

```bash
make smoke MODEL=openbmb/MiniCPM5-2B
make train MODEL=openbmb/MiniCPM5-2B
make train-all MODELS="openbmb/MiniCPM5-2B ibm-granite/granite-4.2-3b LiquidAI/LFM3-1.7B"   # smoke + train each in turn, keeps going on failure
make compare                                   # table of outputs/*: OOF, fold mean/std, log-loss, AUC
```

`LiquidAI/LFM3-1.7B` needs custom model code (`trust_remote_code`; license `lfm1.0`, not gated). It is the only repo enabled automatically (with a warning); for any other repo opt in with `TRUST_REMOTE_CODE=1` (or `--trust-remote-code`). MiniCPM5 and Granite use native transformers classes.

## Costs

| GPU | Approx. $/hour | Notes |
| --- | --- | --- |
| L40S (code default) | check modal.com/pricing | 48 GB, E2B bf16 LoRA fits easily |
| A100 (used in the real run, `MODAL_GPU=A100`) | about 2.10 per GPU | check modal.com/pricing |

A full run is 5 folds x about 6 minutes on A100, well under $2; a smoke run is cents. Modal gives a $30/month free credit.

## Troubleshooting

- **Modal auth errors**: check `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in `.env` and run `make check`.
- **Can't install modal**: upgrade pip first (`python -m pip install --upgrade pip`), use Python 3.11-3.13, and on Windows try a fresh venv (`python -m venv .venv`, activate it, then `python -m pip install -r requirements.txt`). In Codespaces, `make install` re-runs the install.
- **Kaggle 403**: you have not joined the competition. Click **Join Competition** and accept the rules.
- **Hugging Face 401/403**: accept the Gemma license on the model page, check `HF_TOKEN` in `.env`.
- **`select_lora_targets` raises ValueError**: the Gemma 4 module names differ from what is expected. Paste the module names from the log and adjust `src/titanic/lora.py`.
- **ValueError about the Gemma 4 audio tower and SDPA**: fixed in the code (text/vision use SDPA, audio uses eager). Only seen if you pin an old commit; update.
- **`git pull` shows no updates**: if your codespace was created from a fork or copy of the repo, it does not track this repo. Run `git pull https://github.com/afarhadi99/gemma4-titanic.git main`.
- **`make check` fails**: read the real Modal/Kaggle error printed under `details:`.
- **Shared a screenshot of `.env`**: rotate the Modal, Hugging Face and Kaggle tokens.
- **CUDA out of memory**: use a bigger `--gpu`, or reduce the batch size in `make_cfg` in `modal_app.py`.

## Repo layout

```
baseline.py            classical CV models -> outputs/baseline/
ensemble.py            blend outputs/*/ -> submissions/submission.csv
modal_app.py           Modal LoRA training and inference
src/titanic/           features, text prompts, submission I/O, LoRA target selection, .env loader
scripts/               download_data, submit, check_env (.sh primary, .ps1 for Windows)
tests/                 pytest suite and synthetic fixtures
.env.example           template for the single .env config file (.env is gitignored)
.devcontainer/         Codespaces config and post-create script
Makefile               shortcuts for every step above
PLAN.md                design and decisions
```
