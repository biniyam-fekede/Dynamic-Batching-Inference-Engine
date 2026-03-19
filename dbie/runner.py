"""Model runner — executes inference in a ThreadPoolExecutor."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
import torch.nn as nn

from dbie import config
from dbie.models import Batch
from dbie.scheduler import BaseScheduler

logger = logging.getLogger(__name__)


class MockLinearModel(nn.Module):
    """Simple linear model for reproducible benchmarking."""

    def __init__(self, input_dim: int = config.INPUT_DIM, output_dim: int = config.OUTPUT_DIM) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ModelRunner:
    """Runs inference batches off the scheduler in a background loop.

    All model forward passes happen inside a ThreadPoolExecutor to avoid
    blocking the asyncio event loop. Futures are resolved via
    loop.call_soon_threadsafe to be safe across threads.
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler: BaseScheduler,
        device: str = config.INFERENCE_DEVICE,
        warmup_batches: int = config.WARMUP_BATCHES,
    ) -> None:
        self._model = model
        self._scheduler = scheduler
        self._device = torch.device(device)
        self._warmup_batches = warmup_batches
        # Single worker to avoid OpenMP/MKL over-subscription on CPU
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._model.to(self._device)
        self._model.eval()
        self._running = False
        self._batches_processed: int = 0

        # Metrics callback — set externally by the server
        self.on_batch_done: Optional[callable] = None

    def warmup(self) -> None:
        """Run warmup batches to eliminate JIT overhead."""
        logger.info(
            "Warming up model with %d batches on %s", self._warmup_batches, self._device
        )
        input_dim = getattr(self._model, "input_dim", config.INPUT_DIM)
        dummy = torch.randn(config.MAX_BATCH_SIZE, input_dim, device=self._device)
        with torch.no_grad():
            for _ in range(self._warmup_batches):
                self._model(dummy)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        logger.info("Warmup complete.")

    def _execute_batch(self, batch: Batch, loop: asyncio.AbstractEventLoop) -> float:
        """Synchronous batch execution — runs in the executor thread.

        Returns inference time in milliseconds.
        """
        tensor = batch.tensor.to(self._device)
        start = time.monotonic()
        with torch.no_grad():
            outputs = self._model(tensor)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.monotonic() - start) * 1000.0

        # Move outputs back to CPU for result delivery
        outputs = outputs.cpu()

        # Resolve each request's future from the event loop thread
        for i, req in enumerate(batch.requests):
            result = outputs[i]
            loop.call_soon_threadsafe(self._resolve_future, req.future, result)

        return inference_ms

    @staticmethod
    def _resolve_future(future: asyncio.Future, result: torch.Tensor) -> None:
        if not future.done():
            future.set_result(result)

    async def run(self) -> None:
        """Main loop: pull batches from scheduler and run inference."""
        loop = asyncio.get_running_loop()
        self._running = True
        logger.info("ModelRunner started.")

        try:
            while self._running:
                batch = await self._scheduler.next_batch()
                if batch is None:
                    continue

                inference_ms = await loop.run_in_executor(
                    self._executor, self._execute_batch, batch, loop
                )

                self._batches_processed += 1

                # Periodic CUDA cache clear to avoid allocator fragmentation
                if (
                    self._device.type == "cuda"
                    and self._batches_processed % 100 == 0
                ):
                    torch.cuda.empty_cache()

                if self.on_batch_done:
                    self.on_batch_done(batch, inference_ms)

        except asyncio.CancelledError:
            logger.info("ModelRunner cancelled.")
        finally:
            self._running = False
            self._executor.shutdown(wait=False)

    def stop(self) -> None:
        self._running = False

    @property
    def batches_processed(self) -> int:
        return self._batches_processed
