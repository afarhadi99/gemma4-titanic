#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then PY=python3
else PY=python; fi

fail() { echo "error: $1" >&2; exit 1; }

# fail_with_output MESSAGE OUTPUT: print the error, then the last 15 lines of OUTPUT indented.
fail_with_output() {
  echo "error: $1" >&2
  if [[ -n "$2" ]]; then
    echo "  details:" >&2
    printf '%s
' "$2" | tail -n 15 | sed 's/^/    /' >&2
  fi
  exit 1
}

echo "Using $PY"

[[ -f .env ]] || fail "no .env found. Run: cp .env.example .env and fill it in."
set -a
. ./.env
set +a

echo "Keys in .env (values are never printed):"
for name in MODAL_TOKEN_ID MODAL_TOKEN_SECRET HF_TOKEN KAGGLE_API_TOKEN KAGGLE_USERNAME KAGGLE_KEY; do
  if [[ -n "${!name:-}" ]]; then echo "  $name: set"; else echo "  $name: missing"; fi
done

[[ -n "${MODAL_TOKEN_ID:-}" && -n "${MODAL_TOKEN_SECRET:-}" ]] || fail "set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env."
[[ -n "${HF_TOKEN:-}" ]] || fail "set HF_TOKEN in .env."
[[ -n "${KAGGLE_API_TOKEN:-}" || ( -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ) ]] \
  || fail "set KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY) in .env."

echo "Checking Modal auth..."
if ! out=$("$PY" -c "import modal" 2>&1); then
  fail_with_output "modal is not installed in $PY — run: make install" "$out"
fi
if ! out=$("$PY" -m modal token info 2>&1); then
  fail_with_output "Modal auth failed. Check MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env." "$out"
fi
echo "  Modal: ok"

echo "Checking Kaggle auth..."
if ! out=$("$PY" -c "import kaggle" 2>&1); then
  fail_with_output "kaggle is not installed in $PY — run: make install" "$out"
fi
if ! out=$("$PY" -m kaggle competitions list -s titanic 2>&1); then
  fail_with_output "Kaggle auth failed. Check KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY) in .env." "$out"
fi
echo "  Kaggle: ok"
