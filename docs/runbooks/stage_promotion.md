# Runbook: Stage promotion

## Stages
paper -> live_shadow -> live.

## Promotion checklist (each stage)

1. Confirm gates from `configs/promotion.yaml` are met for the current stage.
2. Generate the promotion report:
   `PYTHONPATH=src uv run python scripts/generate_promotion_report.py --from <stage>`.
3. Commit the report under `docs/runbooks/<from>_to_<to>.md` with operator signature line.
4. Edit `.env` to set `QUANTLAB_STAGE=<next>`. Verify with `grep QUANTLAB_STAGE .env`.
5. Stop the running process (SIGINT). Confirm clean shutdown.
6. Start the process with the new stage env var.
7. Confirm `quantlab status` reports the new stage.
8. For `live`, confirm `configs/risk.yaml` caps are cut to 50% for the first 30 days.

## Demotion / rollback
Any kill-switch trigger automatically demotes to `live_shadow` for 7 days before the
next promotion attempt is permitted.
