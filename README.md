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
