# Social9 Forecasting

Authenticated forecasting and optional Gemini content-generation service for Social9.

The development model trains on a CC BY 4.0 Zenodo release containing 34,811
real Instagram posts from 309 anonymized accounts. Social9 converts those posts
into daily account histories while preserving attribution in
`data/external/README.md`. Synthetic data is retained only as an optional,
deterministic edge-case generator and is not part of normal training.

The public data represents music-artist accounts and does not contain private
Instagram reach or impression insights. Forecasts must remain experimental until
they are validated and retrained with consented, anonymized Social9 account data.

## Run locally

```bash
pip install -r requirements.txt
python -m src.import_public_data
python -m src.train
set FORECASTING_API_KEY=local-development-key
uvicorn src.api:app --reload
```

`POST /forecast` accepts `forecast_days` and an optional account-specific `history`
array containing at least eight daily metric records. Production requests must send
the shared key in `X-Social9-Forecasting-Key`.


## Connect to Social9

Set `GEMINI_API_KEY` and `FORECASTING_API_KEY` on this service. All POST endpoints,
including `/ai-text`, `/ai-caption` and `/ai-calendar`, require the
`X-Social9-Forecasting-Key` header. Health endpoints remain public.

On `social9-backend`, set `AI_PROVIDER=forecasting`, `FORECASTING_API_URL` to this
service's base URL, and `FORECASTING_API_KEY` to the same shared key. Deploy this
service before activating that backend configuration. The backend calls
`/ai-text` for the Create post assistant and `/forecast` for engagement forecasts.
Gemini credentials remain exclusively on this service. The Social9 backend
handles user authentication, subscription checks and assistant rate limits.

Run `python -m pytest -q tests/test_ai_integration.py` to verify AI endpoint
authentication and safe error responses without calling Gemini or spending credits.


Deployment compatibility: the API accepts both `history` (Social9 backend) and
`historical_data` (new client spelling). The validated ratio-based model and
licensed training-data loader are retained: the alternative upstream training
pipeline referenced a removed DATA_FILE setting and changed the feature schema.
Re-training and the forecast regression tests must pass before deployment.
