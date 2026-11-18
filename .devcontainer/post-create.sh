#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

[[ -d .venv ]] || python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

activate='source "$PWD/.venv/bin/activate"'
grep -qxF "$activate" ~/.bashrc || echo "$activate" >> ~/.bashrc

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
