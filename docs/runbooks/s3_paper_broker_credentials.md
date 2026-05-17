# Runbook: S3 paper broker credentials

## Purpose
Place free paper-trading API credentials so the AlpacaPaper and BinanceTestnet
adapters can authenticate.

## Alpaca paper
1. Sign up at https://alpaca.markets (free).
2. Switch the dashboard to Paper Trading mode.
3. Generate an API key + secret.
4. Place the credentials at ~/.alpaca/paper_keys.json:
   ```json
   { "api_key": "PK...", "api_secret": "..." }
   ```
5. chmod 600 ~/.alpaca/paper_keys.json
6. Verify: `PYTHONPATH=src uv run pytest tests/integration/test_alpaca_paper_roundtrip.py -m s3_integration`

## Binance testnet
1. Sign up at https://testnet.binance.vision (free, GitHub login).
2. Generate HMAC SHA256 key + secret.
3. Place at ~/.binance/testnet_keys.json:
   ```json
   { "api_key": "...", "api_secret": "..." }
   ```
4. chmod 600 ~/.binance/testnet_keys.json
5. Verify: `PYTHONPATH=src uv run pytest tests/integration/test_binance_testnet_roundtrip.py -m s3_integration`

## File permissions
Both files MUST be chmod 600 (owner read-only). The adapters refuse to load with
permissive permissions to avoid leaking credentials into shared snapshots.

## Rotation
- Alpaca: rotate by generating a new key on the dashboard and replacing the JSON.
  Old key is revoked immediately.
- Binance testnet: same procedure on the testnet dashboard.

## Live credentials
NEVER put live credentials in these paths. Live keys belong at
~/.alpaca/live_keys.json and ~/.binance/live_keys.json respectively and are only
read by *_live.py adapters (which don't exist yet — S4 will add them).
