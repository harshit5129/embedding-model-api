import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock


@pytest_asyncio.fixture
async def client():
    with patch("app.core.models.load_models") as mock_load:
        mock_load.return_value = None
        with patch("app.core.models.is_models_loaded", return_value=True):
            with patch("app.core.models.embed_texts_sync") as mock_embed:
                mock_embed.return_value = [[0.1, 0.2, 0.3] * 16]
                from app.main import app
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as ac:
                    yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    response = await client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "text_model" in data
    assert "image_model" in data


@pytest.mark.asyncio
async def test_embed_text(client):
    response = await client.post(
        "/embed/text", json={"text": "Hello world"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data


@pytest.mark.asyncio
async def test_embed_text_batch(client):
    response = await client.post(
        "/embed/text/batch", json={"texts": ["Hello", "World"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data


@pytest.mark.asyncio
async def test_similarity(client):
    response = await client.post(
        "/similarity",
        json={
            "query_embedding": [0.1] * 512,
            "embeddings": [[0.2] * 512, [0.3] * 512]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "similarities" in data
    assert len(data["similarities"]) == 2


@pytest.mark.asyncio
async def test_status_page(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "Embedding API" in response.text
