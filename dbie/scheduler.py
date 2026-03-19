"""Batch schedulers: FIFO and Adaptive strategies."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Optional

from dbie import config
from dbie.models import Batch, Request
from dbie.queue import RequestQueue


class BaseScheduler(ABC):
    """Common interface for all schedulers."""

    def __init__(self, queue: RequestQueue) -> None:
        self._queue = queue

    @abstractmethod
    async def next_batch(self) -> Optional[Batch]:
        """Wait for and return the next batch, or None on cancellation."""
        ...


class FIFOScheduler(BaseScheduler):
    """Fixed-size batching with timeout from head request arrival time.

    Waits until at least one request is available, then waits up to
    MAX_WAIT_MS (measured from the *head request's* arrival time) to
    fill the batch up to MAX_BATCH_SIZE.
    """

    def __init__(
        self,
        queue: RequestQueue,
        max_batch_size: int = config.MAX_BATCH_SIZE,
        max_wait_ms: float = config.MAX_WAIT_MS,
    ) -> None:
        super().__init__(queue)
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms

    async def next_batch(self) -> Optional[Batch]:
        # Step 1: block until at least one request arrives
        try:
            head: Request = await self._queue.get_async()
        except asyncio.CancelledError:
            return None

        requests = [head]

        # Step 2: compute deadline from the head request's arrival time
        deadline = head.arrival_time + self._max_wait_ms / 1000.0
        remaining = deadline - time.monotonic()

        if remaining > 0:
            await asyncio.sleep(remaining)

        # Step 3: drain up to max_batch_size (including the head we already have)
        extra = self._queue.drain(self._max_batch_size - 1)
        requests.extend(extra)

        return Batch.from_requests(requests)


class AdaptiveScheduler(BaseScheduler):
    """Dynamically adjusts batch size and wait time based on load signals.

    Uses exponential moving averages of queue depth and inter-arrival time
    to derive target batch size and wait. Includes hysteresis to prevent
    oscillation.
    """

    def __init__(
        self,
        queue: RequestQueue,
        max_batch_size: int = config.MAX_BATCH_SIZE,
        max_wait_ms: float = config.MAX_WAIT_MS,
        target_latency_ms: float = config.ADAPTIVE_TARGET_LATENCY_MS,
        ema_alpha: float = config.ADAPTIVE_EMA_ALPHA,
        fill_factor: float = config.ADAPTIVE_FILL_FACTOR,
        min_batch_size: int = config.ADAPTIVE_MIN_BATCH_SIZE,
        min_wait_ms: float = config.ADAPTIVE_MIN_WAIT_MS,
        hysteresis_n: int = config.ADAPTIVE_HYSTERESIS_N,
    ) -> None:
        super().__init__(queue)
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._target_latency_ms = target_latency_ms
        self._alpha = ema_alpha
        self._fill_factor = fill_factor
        self._min_batch_size = min_batch_size
        self._min_wait_ms = min_wait_ms
        self._hysteresis_n = hysteresis_n

        # EMA state
        self._ema_queue_depth: float = 1.0
        self._ema_inter_arrival_ms: float = 10.0
        self._last_enqueue_time: Optional[float] = None

        # Derived targets
        self._target_batch_size: int = min_batch_size
        self._target_wait_ms: float = max_wait_ms

        # Hysteresis counters: track consecutive signals to increase/decrease
        self._consecutive_increase: int = 0
        self._consecutive_decrease: int = 0

        # Latency tracking (updated externally via record_latency)
        self._recent_latencies_ms: list[float] = []

    def record_enqueue(self) -> None:
        """Call when a request is enqueued to update inter-arrival EMA."""
        now = time.monotonic()
        if self._last_enqueue_time is not None:
            delta_ms = (now - self._last_enqueue_time) * 1000.0
            self._ema_inter_arrival_ms = (
                self._alpha * delta_ms
                + (1 - self._alpha) * self._ema_inter_arrival_ms
            )
        self._last_enqueue_time = now

    def record_latency(self, latency_ms: float) -> None:
        """Record an observed request latency for P99 tracking."""
        self._recent_latencies_ms.append(latency_ms)
        # Keep a bounded window
        if len(self._recent_latencies_ms) > 200:
            self._recent_latencies_ms = self._recent_latencies_ms[-200:]

    def _update_targets(self) -> None:
        """Recompute target batch size and wait time from EMA signals."""
        # Update queue depth EMA
        self._ema_queue_depth = (
            self._alpha * self._queue.size
            + (1 - self._alpha) * self._ema_queue_depth
        )

        # Derive raw target batch size
        raw_target = self._ema_queue_depth * self._fill_factor
        proposed = int(max(self._min_batch_size, min(raw_target, self._max_batch_size)))

        # Check P99 latency pressure
        p99_exceeded = False
        if len(self._recent_latencies_ms) >= 20:
            sorted_lat = sorted(self._recent_latencies_ms)
            p99_idx = int(len(sorted_lat) * 0.99)
            p99 = sorted_lat[min(p99_idx, len(sorted_lat) - 1)]
            if p99 > self._target_latency_ms:
                p99_exceeded = True

        # Apply hysteresis
        if p99_exceeded or proposed < self._target_batch_size:
            self._consecutive_decrease += 1
            self._consecutive_increase = 0
            if self._consecutive_decrease >= self._hysteresis_n:
                self._target_batch_size = max(
                    self._min_batch_size, self._target_batch_size - 1
                )
                self._consecutive_decrease = 0
        elif proposed > self._target_batch_size:
            self._consecutive_increase += 1
            self._consecutive_decrease = 0
            if self._consecutive_increase >= self._hysteresis_n:
                self._target_batch_size = min(
                    self._max_batch_size, self._target_batch_size + 1
                )
                self._consecutive_increase = 0
        else:
            self._consecutive_increase = 0
            self._consecutive_decrease = 0

        # Derive wait time
        self._target_wait_ms = max(
            self._min_wait_ms,
            min(
                self._ema_inter_arrival_ms * self._target_batch_size,
                self._max_wait_ms,
            ),
        )

    async def next_batch(self) -> Optional[Batch]:
        # Block until at least one request
        try:
            head: Request = await self._queue.get_async()
        except asyncio.CancelledError:
            return None

        self._update_targets()

        requests = [head]
        deadline = head.arrival_time + self._target_wait_ms / 1000.0
        remaining = deadline - time.monotonic()

        if remaining > 0:
            await asyncio.sleep(remaining)

        extra = self._queue.drain(self._target_batch_size - 1)
        requests.extend(extra)

        return Batch.from_requests(requests)

    @property
    def target_batch_size(self) -> int:
        return self._target_batch_size

    @property
    def target_wait_ms(self) -> float:
        return self._target_wait_ms


def create_scheduler(
    queue: RequestQueue,
    strategy: str = config.SCHEDULER_STRATEGY,
) -> BaseScheduler:
    """Factory: create the configured scheduler."""
    if strategy == "fifo":
        return FIFOScheduler(queue)
    elif strategy == "adaptive":
        return AdaptiveScheduler(queue)
    else:
        raise ValueError(f"Unknown scheduler strategy: {strategy!r}")
