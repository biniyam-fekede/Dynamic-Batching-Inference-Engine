"""Async request queue with backpressure support."""

from __future__ import annotations

import asyncio
from typing import List

from dbie.models import Request


class QueueFullError(Exception):
    """Raised when the queue is at capacity — signals HTTP 429."""
    pass


class RequestQueue:
    """Bounded async queue wrapping asyncio.Queue.

    - put() is non-blocking: raises QueueFullError immediately if full.
    - drain() is non-blocking: pulls up to max_size items without waiting.
    - get_async() blocks until at least one item is available.
    """

    def __init__(self, maxsize: int) -> None:
        self._queue: asyncio.Queue[Request] = asyncio.Queue(maxsize=maxsize)

    def put(self, request: Request) -> None:
        """Enqueue a request. Raises QueueFullError if at capacity."""
        try:
            self._queue.put_nowait(request)
        except asyncio.QueueFull:
            raise QueueFullError(
                f"Queue at capacity ({self._queue.maxsize}). Try again later."
            )

    async def get_async(self) -> Request:
        """Block until a request is available, then return it."""
        return await self._queue.get()

    def drain(self, max_size: int) -> List[Request]:
        """Non-blocking drain of up to max_size items from the queue."""
        items: List[Request] = []
        while len(items) < max_size:
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def maxsize(self) -> int:
        return self._queue.maxsize
