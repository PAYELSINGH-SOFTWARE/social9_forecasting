# Social9 Forecasting

Authenticated forecasting and optional Gemini content-generation service for Social9.

The development dataset contains 7,200 deterministic synthetic daily records across
16 account profiles. It is suitable for integration and behavior testing, not for
making guaranteed performance claims. Replace or supplement it with consented,
anonymized production histories before treating forecasts as production-grade.

## Run locally

```bash
pip install -r requirements.txt
python -m src.generate_training_data
python -m src.train
set FORECASTING_API_KEY=local-development-key
uvicorn src.api:app --reload
```

`POST /forecast` accepts `forecast_days` and an optional account-specific `history`
array containing at least eight daily metric records. Production requests must send
the shared key in `X-Social9-Forecasting-Key`.
