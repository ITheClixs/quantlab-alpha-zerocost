# Runbook: Governor LoRA retrain

## When to run
- Veto precision on the 200-pair backtest fixture drops below 60 % (master spec
  criterion 6).
- Held-out perplexity vs base regresses by > 5 %.
- The synthetic-label rule changes (regenerate `lora_governor.jsonl`).
- Quarterly hygiene retrain.

## Steps
1. Stop `s2_govern.py` daemon.
2. Regenerate the LoRA dataset:
   `PYTHONPATH=src uv run python scripts/governor_lora_dataset.py`
3. Train the adapter (up to 8 h wall-clock; checkpoints every 500 steps):
   `PYTHONPATH=src uv run python scripts/governor_train_lora.py`
4. Inspect `models/trained/governor_lora_qwen05b/<run_id>/metrics.json`.
   Required: `held_out_perplexity` < base − 10 %, `veto_precision_200pair` ≥ 0.60.
5. If criteria pass, point `configs/governor.yaml::tiers.tier1.adapter_dir` at the
   new run dir or update `models/trained/governor_lora_qwen05b/latest` symlink.
6. If criteria fail, run the fallback adapter on Qwen 1.5B Coder:
   `PYTHONPATH=src uv run python scripts/governor_train_lora.py --base Qwen/Qwen2.5-Coder-1.5B`
7. Restart `s2_govern.py`.

## Failure modes
- MPS OOM during training: drop `batch_size` from 4 to 2 in `configs/governor.yaml`,
  bump `gradient_accumulation_steps` to 8 to keep effective batch size 16.
- Training stalls: kill, examine the loss curve in `metrics.json`, reduce
  `learning_rate` from 2e-4 to 1e-4.
