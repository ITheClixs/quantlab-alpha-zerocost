PY := PYTHONPATH=src uv run
EXTRACT := scripts/alpha_extract_meta_features.py
TRAIN := scripts/alpha_train_s1.py
OPTUNA := scripts/alpha_optuna_search.py

.PHONY: test lint type extract train optuna full-retrain-s1 clean-experiments

test:
	$(PY) pytest -q

lint:
	uv run ruff check src scripts tests

type:
	uv run mypy src

extract:
	$(PY) python $(EXTRACT)

train:
	$(PY) python $(TRAIN)

optuna:
	$(PY) python $(OPTUNA) --n-trials 200

full-retrain-s1: test lint extract train optuna
	@echo "S1 full retrain complete. See experiments/alpha_s1/<latest>/metrics.json"

clean-experiments:
	rm -rf experiments/alpha_s1/*
