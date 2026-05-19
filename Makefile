PY := PYTHONPATH=src uv run
EXTRACT := scripts/alpha_extract_meta_features.py
TRAIN := scripts/alpha_train_s1.py
TRAIN_STREAMING := scripts/alpha_train_s1_streaming.py
TRAIN_STREAMING_CONFIG ?= configs/alpha_5m.yaml
TRAIN_STREAMING_ROWS ?= 5000000
OPTUNA := scripts/alpha_optuna_search.py
OPTUNA_ARGS ?= --n-trials 200

.PHONY: test lint type extract train train-streaming optuna full-retrain-s1 clean-experiments

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

train-streaming:
	$(PY) python $(TRAIN_STREAMING) --config $(TRAIN_STREAMING_CONFIG) --max-rows $(TRAIN_STREAMING_ROWS)

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

S3_RECORD := scripts/s3_record.py
BACKTEST_RUN := scripts/backtest_run.py
BACKTEST_CONFIG ?= configs/backtests/smoke.yaml

.PHONY: s3-record s3-parity backtest backtest-smoke

s3-record:
	$(PY) python $(S3_RECORD) --config configs/feeds.yaml

s3-parity:
	$(PY) pytest tests/integration/test_record_replay_parity.py -v -m s3_integration

backtest:
	$(PY) python $(BACKTEST_RUN) --config $(BACKTEST_CONFIG)

backtest-smoke:
	$(PY) python $(BACKTEST_RUN) --config configs/backtests/smoke.yaml

S4_EXECUTE := scripts/s4_execute.py
PROMOTION_REPORT := scripts/generate_promotion_report.py
AUDIT_REPLAY := scripts/audit_replay_check.py
S4_STAGE ?= paper
S4_ASSET ?= crypto
S4_EQUITY ?= 100000

.PHONY: s4-execute s4-promotion-report s4-audit-replay s4-smoke

s4-execute:
	QUANTLAB_STAGE=$(S4_STAGE) $(PY) python $(S4_EXECUTE) \
	  --risk-config configs/risk.yaml --exec-config configs/exec.yaml \
	  --brokers-config configs/brokers.yaml --asset-class $(S4_ASSET) \
	  --starting-equity $(S4_EQUITY)

s4-promotion-report:
	$(PY) python $(PROMOTION_REPORT) --from-stage paper --to-stage live_shadow \
	  --audit-root logs/audit/s4/paper

s4-audit-replay:
	$(PY) python $(AUDIT_REPLAY) --audit-dir logs/audit/s4/paper --stage paper

s4-smoke:
	$(PY) pytest tests/integration/test_s4_*.py -v -m s4_integration
