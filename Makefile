.DEFAULT_GOAL := help
MSG ?= gemma4-titanic submission

.PHONY: help setup secrets data test baseline smoke train ensemble submit clean-outputs

help:
	@echo "Targets:"
	@echo "  setup          install uv, sync dependencies, show secret status"
	@echo "  secrets        verify Modal/Kaggle auth and create the Modal huggingface-secret"
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

secrets:
	bash scripts/setup_secrets.sh

data:
	bash scripts/download_data.sh

test:
	uv run pytest -q

baseline:
	uv run python baseline.py

smoke:
	uv run modal run modal_app.py --smoke

train:
	uv run modal run modal_app.py

ensemble:
	uv run python ensemble.py

submit:
	bash scripts/submit.sh submissions/submission.csv "$(MSG)"

clean-outputs:
	rm -rf outputs submissions
