"""Integration tests for the FastAPI server."""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from dbie import config
from dbie.server import app, lifespan


@pytest.fixture
async def client():
    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["device"] == config.INFERENCE_DEVICE


@pytest.mark.asyncio
async def test_infer_success(client):
    payload = {"data": [1.0] * config.INPUT_DIM}
    resp = await client.post("/infer", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "request_id" in body
    assert "result" in body
    assert len(body["result"]) == config.OUTPUT_DIM
    assert body["latency_ms"] > 0


@pytest.mark.asyncio
async def test_infer_wrong_dimension(client):
    payload = {"data": [1.0, 2.0, 3.0]}  # Wrong size
    resp = await client.post("/infer", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "throughput_rps" in body
    assert "total_requests" in body


@pytest.mark.asyncio
async def test_concurrent_requests(client):
    """Send multiple requests concurrently and verify all succeed."""
    payloads = [{"data": [float(i)] * config.INPUT_DIM} for i in range(10)]
    tasks = [client.post("/infer", json=p) for p in payloads]
    responses = await asyncio.gather(*tasks)

    for resp in responses:
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["result"]) == config.OUTPUT_DIM
