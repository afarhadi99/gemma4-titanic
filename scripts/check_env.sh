#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

fail() { echo "error: $1" >&2; exit 1; }

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
uv run modal token info >/dev/null 2>&1 || fail "Modal auth failed. Check MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env."
echo "  Modal: ok"

echo "Checking Kaggle auth..."
uv run kaggle competitions list -s titanic >/dev/null 2>&1 || fail "Kaggle auth failed. Check KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY) in .env."
echo "  Kaggle: ok"
