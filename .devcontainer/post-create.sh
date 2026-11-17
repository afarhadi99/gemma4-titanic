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

[[ -f .env ]] || cp .env.example .env

echo "Keys in .env (values are never printed):"
while IFS= read -r line; do
  line="${line#export }"
  [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
  name="${BASH_REMATCH[1]}"
  value="$(printf '%s' "${BASH_REMATCH[2]}" | tr -d "[:space:]\"'")"
  if [[ -n "$value" ]]; then echo "  $name: set"; else echo "  $name: missing"; fi
done < .env
echo "Fill in .env, then run: make check"
