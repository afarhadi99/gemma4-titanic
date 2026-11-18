#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then PY=python3
else PY=python; fi

if [[ ! -f .env ]]; then
  echo "error: no .env found. Run: cp .env.example .env and fill it in." >&2
  exit 1
fi
set -a
. ./.env
set +a

if [[ -z "${KAGGLE_API_TOKEN:-}" && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) ]]; then
  echo "error: no Kaggle credentials in .env. Set KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY)." >&2
  exit 1
fi

mkdir -p data
"$PY" -m kaggle competitions download titanic -p data
unzip -o data/titanic.zip -d data

for f in train test; do
  rows=$(( $(wc -l < "data/$f.csv") - 1 ))
  echo "data/$f.csv: $rows rows"
done
