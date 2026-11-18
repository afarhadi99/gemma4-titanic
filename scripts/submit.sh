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

FILE="${1:-submissions/submission.csv}"
MESSAGE="${2:-gemma4-titanic submission}"

if [[ ! -f "$FILE" ]]; then
  echo "error: $FILE not found" >&2
  exit 1
fi

"$PY" -c "
import sys
from titanic.submission import validate_submission
validate_submission(sys.argv[1], 'data/test.csv')
print('submission valid:', sys.argv[1])
" "$FILE"

"$PY" -m kaggle competitions submit titanic -f "$FILE" -m "$MESSAGE"
"$PY" -m kaggle competitions submissions titanic
