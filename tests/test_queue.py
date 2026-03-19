"""Tests for async request queue."""

import asyncio

import pytest
import torch

from dbie.models import Request
from dbie.queue import QueueFullError, RequestQueue


def _make_request(loop: asyncio.AbstractEventLoop) -> Request:
    return Request(payload=torch.randn(8), future=loop.create_future())


@pytest.mark.asyncio
async def test_put_and_drain():
    q = RequestQueue(maxsize=10)
    loop = asyncio.get_running_loop()

    for _ in range(5):
        q.put(_make_request(loop))

    assert q.size == 5
    items = q.drain(3)
    assert len(items) == 3
    assert q.size == 2


@pytest.mark.asyncio
async def test_backpressure():
    q = RequestQueue(maxsize=2)
    loop = asyncio.get_running_loop()

    q.put(_make_request(loop))
    q.put(_make_request(loop))
    assert q.size == 2

    with pytest.raises(QueueFullError):
        q.put(_make_request(loop))


@pytest.mark.asyncio
async def test_get_async():
    q = RequestQueue(maxsize=10)
    loop = asyncio.get_running_loop()

    req = _make_request(loop)
    q.put(req)

    got = await q.get_async()
    assert got.request_id == req.request_id


@pytest.mark.asyncio
async def test_drain_empty():
    q = RequestQueue(maxsize=10)
    items = q.drain(5)
    assert items == []
