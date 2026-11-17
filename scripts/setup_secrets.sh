#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

fail() { echo "error: $1" >&2; exit 1; }

echo "Checking Modal auth..."
if ! uv run modal token info >/dev/null 2>&1; then
  fail "Modal is not authenticated. Set the Codespaces secrets MODAL_TOKEN_ID and MODAL_TOKEN_SECRET (or run: uv run modal setup)."
fi
echo "  Modal: ok"

echo "Syncing Hugging Face token to Modal secret 'huggingface-secret'..."
[[ -n "${HF_TOKEN:-}" ]] || fail "HF_TOKEN is not set. Add it as a Codespaces secret (https://huggingface.co/settings/tokens) and accept the Gemma license."
uv run modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN" --force >/dev/null
echo "  huggingface-secret: created/updated"

echo "Checking Kaggle auth..."
if ! uv run kaggle competitions list -s titanic >/dev/null 2>&1; then
  fail "Kaggle auth failed. Set KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY) as Codespaces secrets."
fi
echo "  Kaggle: ok"
