import pytest
from fastapi.testclient import TestClient
from src import api


@pytest.mark.parametrize("path,payload", [
    ("/ai-text", {"prompt": "Write a caption"}),
    ("/ai-caption", {"topic": "Weekend pastries"}),
    ("/ai-calendar", {"topic": "Weekend pastries"}),
])
def test_content_endpoints_require_service_auth(monkeypatch, path, payload):
    monkeypatch.setenv("FORECASTING_API_KEY", "test-service-key")
    monkeypatch.setattr(api, "generate_ai_text", lambda prompt: "generated")
    monkeypatch.setattr(api, "generate_content_calendar", lambda **kwargs: "calendar")
    with TestClient(api.app) as client:
        assert client.post(path, json=payload).status_code == 401
        assert client.post(path, json=payload, headers={"X-Social9-Forecasting-Key": "wrong"}).status_code == 401
        assert client.post(path, json=payload, headers={"X-Social9-Forecasting-Key": "test-service-key"}).status_code == 200


def test_calendar_requests_json_from_gemini(monkeypatch):
    from google import genai
    from src.ai_calendar import generate_content_calendar

    captured = {}
    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"text": '{"items": []}'})()
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(genai, "Client", lambda **kwargs: type("Client", (), {"models": FakeModels()})())
    assert generate_content_calendar("instagram", 30, "bakery", "Custom cakes for local families", "2026-10-01") == '{"items": []}'
    assert captured["config"].response_mime_type == "application/json"
    assert "exactly 30 entries" in captured["contents"]
    assert "Custom cakes for local families" in captured["contents"]
    assert "2026-10-01" in captured["contents"]


def test_ai_error_does_not_expose_provider_details(monkeypatch):
    monkeypatch.setenv("FORECASTING_API_KEY", "test-service-key")
    def fail(prompt):
        raise RuntimeError("private provider details")
    monkeypatch.setattr(api, "generate_ai_text", fail)
    with TestClient(api.app) as client:
        response = client.post("/ai-text", json={"prompt": "Write a caption"},
            headers={"X-Social9-Forecasting-Key": "test-service-key"})
    assert response.status_code == 500
    assert "private" not in response.text


def test_new_history_name_remains_compatible(monkeypatch):
    monkeypatch.setenv("FORECASTING_API_KEY", "test-service-key")
    history = [{"date": f"2026-09-{day:02d}", "reach": 10,
        "impressions": 20, "followers": 30} for day in range(1, 9)]
    captured = []
    def fake_forecast(days, records):
        captured.extend(records)
        return [{"date": "2026-09-09", "predicted_engagement": 1}]
    monkeypatch.setattr(api, "forecast", fake_forecast)
    with TestClient(api.app) as client:
        response = client.post("/forecast", json={"forecast_days": 1,
            "historical_data": history, "instagram_id": "test-account"},
            headers={"X-Social9-Forecasting-Key": "test-service-key"})
    assert response.status_code == 200
    assert len(captured) == 8
