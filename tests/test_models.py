"""Tests for core data structures."""

import asyncio

import pytest
import torch

from dbie.models import Batch, Request


def _make_request(dim: int = 128, loop=None) -> Request:
    if loop is None:
        loop = asyncio.new_event_loop()
    future = loop.create_future()
    return Request(payload=torch.randn(dim), future=future)


def test_request_defaults():
    loop = asyncio.new_event_loop()
    try:
        req = _make_request(loop=loop)
        assert len(req.request_id) == 32  # hex UUID
        assert req.arrival_time > 0
        assert not req.future.done()
    finally:
        loop.close()


def test_batch_from_requests():
    loop = asyncio.new_event_loop()
    try:
        requests = [_make_request(dim=64, loop=loop) for _ in range(4)]
        batch = Batch.from_requests(requests)
        assert batch.tensor.shape == (4, 64)
        assert batch.size == 4
    finally:
        loop.close()


def test_batch_shape_mismatch_resolves_futures():
    loop = asyncio.new_event_loop()
    try:
        r1 = Request(payload=torch.randn(64), future=loop.create_future())
        r2 = Request(payload=torch.randn(128), future=loop.create_future())

        with pytest.raises(ValueError, match="shape mismatch"):
            Batch.from_requests([r1, r2])

        # Both futures should have the exception set
        assert r1.future.done()
        assert r2.future.done()
        with pytest.raises(ValueError):
            r1.future.result()
    finally:
        loop.close()
