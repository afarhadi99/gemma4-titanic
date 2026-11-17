#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync

activate='source "$PWD/.venv/bin/activate"'
grep -qxF "$activate" ~/.bashrc || echo "$activate" >> ~/.bashrc
grep -q '.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

echo "Secret status (values are never printed):"
for name in MODAL_TOKEN_ID MODAL_TOKEN_SECRET HF_TOKEN KAGGLE_API_TOKEN KAGGLE_USERNAME KAGGLE_KEY; do
  if [[ -n "${!name:-}" ]]; then echo "  $name: set"; else echo "  $name: missing"; fi
done
