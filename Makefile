.DEFAULT_GOAL := help
MSG ?= gemma4-titanic submission
RUN := uv run --env-file .env

.PHONY: help setup env check data test baseline smoke train ensemble submit clean-outputs

help:
	@echo "Targets:"
	@echo "  setup          install uv, sync dependencies, create .env from .env.example"
	@echo "  env            fail unless .env exists"
	@echo "  check          verify .env keys and Modal/Kaggle auth"
	@echo "  data           download Titanic data from Kaggle into data/"
	@echo "  test           run pytest"
	@echo "  baseline       run the classical baseline into outputs/baseline"
	@echo "  smoke          cheap end-to-end Gemma run on Modal"
	@echo "  train          full Gemma LoRA run on Modal"
	@echo "  ensemble       blend outputs/* into submissions/submission.csv"
	@echo "  submit         validate and submit to Kaggle (MSG='message')"
	@echo "  clean-outputs  delete outputs/ and submissions/"

setup:
	bash .devcontainer/post-create.sh

env:
	@test -f .env || { echo "No .env found — run: cp .env.example .env and fill it in" >&2; exit 1; }

check: env
	bash scripts/check_env.sh

data: env
	bash scripts/download_data.sh

test:
	uv run pytest -q

baseline:
	uv run python baseline.py

smoke: env
	$(RUN) modal run modal_app.py --smoke

train: env
	$(RUN) modal run modal_app.py

ensemble:
	uv run python ensemble.py

submit: env
	bash scripts/submit.sh submissions/submission.csv "$(MSG)"

clean-outputs:
	rm -rf outputs submissions
