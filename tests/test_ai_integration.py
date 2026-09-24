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
