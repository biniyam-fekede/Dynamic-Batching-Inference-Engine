"""Tests for model runner."""

import asyncio

import pytest
import torch

from dbie.models import Batch, Request
from dbie.queue import RequestQueue
from dbie.runner import MockLinearModel, ModelRunner
from dbie.scheduler import FIFOScheduler


def test_mock_model_forward():
    model = MockLinearModel(input_dim=32, output_dim=16)
    x = torch.randn(4, 32)
    out = model(x)
    assert out.shape == (4, 16)


def test_warmup_runs():
    model = MockLinearModel(input_dim=32, output_dim=16)
    q = RequestQueue(maxsize=10)
    sched = FIFOScheduler(q, max_batch_size=4, max_wait_ms=10)
    runner = ModelRunner(model=model, scheduler=sched, device="cpu", warmup_batches=2)
    runner.warmup()  # Should not raise


@pytest.mark.asyncio
async def test_runner_processes_batch():
    model = MockLinearModel(input_dim=32, output_dim=16)
    q = RequestQueue(maxsize=100)
    sched = FIFOScheduler(q, max_batch_size=4, max_wait_ms=20)
    runner = ModelRunner(model=model, scheduler=sched, device="cpu", warmup_batches=0)

    loop = asyncio.get_running_loop()

    # Enqueue requests
    requests = []
    for _ in range(3):
        req = Request(payload=torch.randn(32), future=loop.create_future())
        q.put(req)
        requests.append(req)

    # Start runner in background
    runner_task = asyncio.create_task(runner.run())

    # Wait for all futures to resolve
    results = await asyncio.wait_for(
        asyncio.gather(*[r.future for r in requests]),
        timeout=5.0,
    )

    assert len(results) == 3
    for r in results:
        assert r.shape == (16,)

    # Cleanup
    runner.stop()
    runner_task.cancel()
    try:
        await runner_task
    except asyncio.CancelledError:
        pass
