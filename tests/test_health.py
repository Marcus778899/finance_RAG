import httpx
import pytest
from app.main import create_app


@pytest.fixture
def client(settings):
    transport = httpx.ASGITransport(app=create_app(settings))
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_health_reports_configured_models(client):
    async with client as c:
        response = await c.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["embed_dim"] == 1024
    assert body["generator_model"] == "qwen3:4b"


async def test_health_timezone_is_taipei(client):
    async with client as c:
        response = await c.get("/api/health")

    assert response.json()["timezone"] == "Asia/Taipei"


async def test_health_never_exposes_secrets(client, settings):
    async with client as c:
        response = await c.get("/api/health")

    assert settings.postgres_password.get_secret_value() not in response.text
    assert "postgres" not in response.text
