"""Tests for scheduler strategies."""

import asyncio
import time

import pytest
import torch

from dbie.models import Request
from dbie.queue import RequestQueue
from dbie.scheduler import AdaptiveScheduler, FIFOScheduler, create_scheduler


def _make_request(loop: asyncio.AbstractEventLoop, dim: int = 128) -> Request:
    return Request(payload=torch.randn(dim), future=loop.create_future())


@pytest.mark.asyncio
async def test_fifo_single_request():
    q = RequestQueue(maxsize=100)
    sched = FIFOScheduler(q, max_batch_size=8, max_wait_ms=10)
    loop = asyncio.get_running_loop()

    q.put(_make_request(loop))
    batch = await asyncio.wait_for(sched.next_batch(), timeout=2.0)
    assert batch is not None
    assert batch.size >= 1


@pytest.mark.asyncio
async def test_fifo_fills_batch():
    q = RequestQueue(maxsize=100)
    sched = FIFOScheduler(q, max_batch_size=4, max_wait_ms=50)
    loop = asyncio.get_running_loop()

    # Add requests before calling next_batch
    for _ in range(4):
        q.put(_make_request(loop))

    batch = await asyncio.wait_for(sched.next_batch(), timeout=2.0)
    assert batch is not None
    assert batch.size == 4


@pytest.mark.asyncio
async def test_adaptive_creates_batch():
    q = RequestQueue(maxsize=100)
    sched = AdaptiveScheduler(q, max_batch_size=8, max_wait_ms=20, min_wait_ms=5)
    loop = asyncio.get_running_loop()

    for _ in range(3):
        q.put(_make_request(loop))
        sched.record_enqueue()

    batch = await asyncio.wait_for(sched.next_batch(), timeout=2.0)
    assert batch is not None
    assert batch.size >= 1


@pytest.mark.asyncio
async def test_adaptive_records_latency():
    q = RequestQueue(maxsize=100)
    sched = AdaptiveScheduler(q, max_batch_size=8)

    sched.record_latency(5.0)
    sched.record_latency(10.0)
    assert len(sched._recent_latencies_ms) == 2


def test_create_scheduler_fifo():
    q = RequestQueue(maxsize=100)
    sched = create_scheduler(q, "fifo")
    assert isinstance(sched, FIFOScheduler)


def test_create_scheduler_adaptive():
    q = RequestQueue(maxsize=100)
    sched = create_scheduler(q, "adaptive")
    assert isinstance(sched, AdaptiveScheduler)


def test_create_scheduler_unknown():
    q = RequestQueue(maxsize=100)
    with pytest.raises(ValueError, match="Unknown"):
        create_scheduler(q, "invalid")
