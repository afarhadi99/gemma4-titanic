#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "error: no .env found. Run: cp .env.example .env and fill it in." >&2
  exit 1
fi
set -a
. ./.env
set +a

FILE="${1:-submissions/submission.csv}"
MESSAGE="${2:-gemma4-titanic submission}"

if [[ ! -f "$FILE" ]]; then
  echo "error: $FILE not found" >&2
  exit 1
fi

uv run python -c "
import sys
from titanic.submission import validate_submission
validate_submission(sys.argv[1], 'data/test.csv')
print('submission valid:', sys.argv[1])
" "$FILE"

uv run kaggle competitions submit titanic -f "$FILE" -m "$MESSAGE"
uv run kaggle competitions submissions titanic
