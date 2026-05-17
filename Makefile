PY := PYTHONPATH=src uv run
EXTRACT := scripts/alpha_extract_meta_features.py
TRAIN := scripts/alpha_train_s1.py
OPTUNA := scripts/alpha_optuna_search.py
OPTUNA_ARGS ?= --n-trials 200

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
	$(PY) python $(OPTUNA) $(OPTUNA_ARGS)

full-retrain-s1: test lint extract train optuna
	@echo "S1 full retrain complete. See experiments/alpha_s1/<latest>/metrics.json"

clean-experiments:
	rm -rf experiments/alpha_s1/*

GOVERNOR_BUILD_INDEXES := scripts/governor_build_indexes.py
GOVERNOR_LORA_DATASET := scripts/governor_lora_dataset.py
GOVERNOR_TRAIN_LORA := scripts/governor_train_lora.py
GOVERNOR_SMOKE := scripts/s2_smoke.py
GOVERNOR_DAEMON := scripts/s2_govern.py

.PHONY: governor-build-indexes governor-lora-dataset governor-train-lora governor-smoke governor-up governor-down

governor-build-indexes:
	$(PY) python $(GOVERNOR_BUILD_INDEXES)

governor-lora-dataset:
	$(PY) python $(GOVERNOR_LORA_DATASET)

governor-train-lora: governor-lora-dataset
	$(PY) python $(GOVERNOR_TRAIN_LORA)

governor-smoke:
	$(PY) python $(GOVERNOR_SMOKE)

governor-up:
	@echo "Run: PYTHONPATH=src uv run python $(GOVERNOR_DAEMON) --predictions <path-to-S1-predictions.parquet>"

governor-down:
	@pkill -f "python.*s2_govern.py" || true
