#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

has_auth=0
[[ -n "${KAGGLE_API_TOKEN:-}" ]] && has_auth=1
[[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]] && has_auth=1
[[ -f "$HOME/.kaggle/kaggle.json" || -f "$HOME/.kaggle/access_token" ]] && has_auth=1

if [[ $has_auth -eq 0 ]]; then
  echo "error: no Kaggle credentials found." >&2
  echo "Set the Codespaces secret KAGGLE_API_TOKEN (or legacy KAGGLE_USERNAME + KAGGLE_KEY)," >&2
  echo "or create ~/.kaggle/access_token / ~/.kaggle/kaggle.json." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

mkdir -p data
uv run kaggle competitions download titanic -p data
unzip -o data/titanic.zip -d data

for f in train test; do
  rows=$(( $(wc -l < "data/$f.csv") - 1 ))
  echo "data/$f.csv: $rows rows"
done
