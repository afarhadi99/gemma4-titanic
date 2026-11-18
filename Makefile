.DEFAULT_GOAL := help
MSG ?= gemma4-titanic submission
MODEL ?=
MODELS ?=
TRUST_REMOTE_CODE ?=
GEMMA ?= google/gemma-4-E2B-it
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.PHONY: help setup install env check data test baseline smoke train train-all sweep seeds variants compare ensemble submit clean-outputs

help:
	@echo "Targets:"
	@echo "  setup          create .venv, install requirements.txt, create .env from .env.example"
	@echo "  install        create .venv and pip install -r requirements.txt"
	@echo "  env            fail unless .env exists"
	@echo "  check          verify .env keys and Modal/Kaggle auth"
	@echo "  data           download Titanic data from Kaggle into data/"
	@echo "  test           run pytest"
	@echo "  baseline       run the classical baseline into outputs/baseline"
	@echo "  smoke          cheap end-to-end run on Modal (MODEL=org/name to override)"
	@echo "  train          full LoRA run on Modal (MODEL=org/name to override)"
	@echo "  train-all      smoke then train each of MODELS=\"a b c\" in turn, continue past failures"
	@echo "  sweep          one-at-a-time Gemma E2B sweep (epochs 2/4/6, lr 5e-5/2e-4, lora-r 8/32), then compare"
	@echo "  seeds          Gemma E2B base settings with seeds 7 and 123 (same folds), then compare"
	@echo "  variants       Gemma E2B with --head classification and --prompt-style structured, then compare"
	@echo "  compare      table of finished runs in outputs/"
	@echo "  ensemble       blend outputs/* into submissions/submission.csv"
	@echo "  submit         validate and submit to Kaggle (MSG='message')"
	@echo "  clean-outputs  delete outputs/ and submissions/"

setup:
	bash .devcontainer/post-create.sh

install:
	python -m venv .venv
	.venv/bin/python -m pip install -r requirements.txt

env:
	@test -f .env || { echo "No .env found — run: cp .env.example .env and fill it in" >&2; exit 1; }

check: env
	bash scripts/check_env.sh

data: env
	bash scripts/download_data.sh

test:
	$(PY) -m pytest -q

baseline:
	$(PY) baseline.py

smoke: env
	$(PY) -m modal run modal_app.py --smoke $(if $(MODEL),--model-id $(MODEL)) $(if $(TRUST_REMOTE_CODE),--trust-remote-code)

train: env
	$(PY) -m modal run modal_app.py $(if $(MODEL),--model-id $(MODEL)) $(if $(TRUST_REMOTE_CODE),--trust-remote-code)

train-all: env
	@test -n "$(MODELS)" || { echo 'Usage: make train-all MODELS="org/a org/b"' >&2; exit 1; }
	@failed=""; ok=""; \
	for m in $(MODELS); do \
	  echo "=== $$m: smoke ==="; \
	  if $(PY) -m modal run modal_app.py --smoke --model-id $$m $(if $(TRUST_REMOTE_CODE),--trust-remote-code); then \
	    echo "=== $$m: train ==="; \
	    if $(PY) -m modal run modal_app.py --model-id $$m $(if $(TRUST_REMOTE_CODE),--trust-remote-code); then ok="$$ok $$m"; else failed="$$failed $$m(train)"; fi; \
	  else failed="$$failed $$m(smoke)"; fi; \
	done; \
	echo "=== train-all summary ==="; echo "succeeded:$${ok:- none}"; echo "failed:$${failed:- none}"; \
	$(PY) compare.py; \
	test -z "$$failed"

# Gemma 4 E2B optimisation runs: each spec is tag=arg+value; every run gets a distinct name
# (gemma4-e2b-<tag>-<timestamp>) and a failure does not stop the remaining runs.
define run_gemma_variants
	@ts=$$(date +%Y%m%d-%H%M%S); failed=""; \
	for spec in $(1); do \
	  tag=$${spec%%=*}; args=$$(echo "$${spec#*=}" | tr '+' ' '); \
	  echo "=== $$tag: $$args ==="; \
	  $(PY) -m modal run modal_app.py --model-id $(GEMMA) --run-name gemma4-e2b-$$tag-$$ts $$args || failed="$$failed $$tag"; \
	done; \
	echo "failed:$${failed:- none}"; \
	$(PY) compare.py; \
	test -z "$$failed"
endef

sweep: env
	$(call run_gemma_variants,ep2=--epochs+2 ep4=--epochs+4 ep6=--epochs+6 lr5e-5=--lr+5e-5 lr2e-4=--lr+2e-4 r8=--lora-r+8 r32=--lora-r+32)

seeds: env
	$(call run_gemma_variants,seed7=--seed+7 seed123=--seed+123)

variants: env
	$(call run_gemma_variants,cls=--head+classification structured=--prompt-style+structured)

compare:
	$(PY) compare.py

ensemble:
	$(PY) ensemble.py

submit: env
	bash scripts/submit.sh submissions/submission.csv "$(MSG)"

clean-outputs:
	rm -rf outputs submissions
