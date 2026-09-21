from fastapi.testclient import TestClient

from src.api import app
from src.config import MODEL_FILE
from src.data_loader import load_data
from src.preprocessing import clean_data
from src.features import create_features
from src.predict import forecast
from src.train import train_model


if not MODEL_FILE.exists():
    train_model()


def test_data_loading():

    df = load_data()

    assert not df.empty

    assert "date" in df.columns


def test_preprocessing():

    df = load_data()

    df = clean_data(df)

    assert "engagement" in df.columns

    assert "engagement_rate" in df.columns


def test_features():

    df = load_data()

    df = clean_data(df)

    df = create_features(df)

    assert "lag_1" in df.columns

    assert "lag_7" in df.columns

    assert "rolling_7" in df.columns


def test_forecast_accepts_account_history():
    df = clean_data(load_data()).tail(10)
    history = df[
        ["date", "likes", "comments", "shares", "reach", "impressions", "followers", "posts_count"]
    ].copy()
    history["date"] = history["date"].dt.strftime("%Y-%m-%d")

    predictions = forecast(3, history.to_dict(orient="records"))

    assert len(predictions) == 3
    assert all(item["predicted_engagement"] >= 0 for item in predictions)


def test_forecast_api_requires_service_key(monkeypatch):
    monkeypatch.setenv("FORECASTING_API_KEY", "test-service-key")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        unauthorized = client.post("/forecast", json={"forecast_days": 3})
        authorized = client.post(
            "/forecast",
            json={"forecast_days": 3},
            headers={"X-Social9-Forecasting-Key": "test-service-key"},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert len(authorized.json()["predictions"]) == 3
