"""FastAPI server — the HTTP interface for DBIE."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dbie import config
from dbie.metrics import MetricsCollector
from dbie.models import Batch, Request
from dbie.queue import QueueFullError, RequestQueue
from dbie.runner import MockLinearModel, ModelRunner
from dbie.scheduler import AdaptiveScheduler, BaseScheduler, create_scheduler

logger = logging.getLogger(__name__)

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class InferRequest(BaseModel):
    data: List[float]

class InferResponse(BaseModel):
    request_id: str
    result: List[float]
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    device: str
    scheduler: str
    queue_size: int
    batches_processed: int

# ── Shared state ──────────────────────────────────────────────────────────────

_queue: RequestQueue
_scheduler: BaseScheduler
_runner: ModelRunner
_metrics: MetricsCollector
_runner_task: asyncio.Task
_metrics_task: asyncio.Task


async def _metrics_sampler() -> None:
    """Background task: samples queue depth at a fixed interval."""
    interval = config.METRICS_SAMPLE_INTERVAL_MS / 1000.0
    try:
        while True:
            _metrics.record_queue_depth(_queue.size)
            _metrics.record_gpu_memory()
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


def _on_batch_done(batch: Batch, inference_ms: float) -> None:
    """Callback invoked by the runner after each batch."""
    _metrics.record_batch(batch.size, inference_ms)
    # Feed latency to adaptive scheduler if applicable
    if isinstance(_scheduler, AdaptiveScheduler):
        _scheduler.record_latency(inference_ms)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _queue, _scheduler, _runner, _metrics, _runner_task, _metrics_task

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Initialize components
    _metrics = MetricsCollector()
    _queue = RequestQueue(maxsize=config.QUEUE_MAX_SIZE)
    _scheduler = create_scheduler(_queue, config.SCHEDULER_STRATEGY)
    model = MockLinearModel()
    _runner = ModelRunner(model=model, scheduler=_scheduler)
    _runner.on_batch_done = _on_batch_done

    # Warmup
    _runner.warmup()

    # Start background tasks
    _runner_task = asyncio.create_task(_runner.run())
    _metrics_task = asyncio.create_task(_metrics_sampler())

    logger.info(
        "DBIE ready — device=%s, scheduler=%s, max_batch=%d, max_wait=%gms",
        config.INFERENCE_DEVICE,
        config.SCHEDULER_STRATEGY,
        config.MAX_BATCH_SIZE,
        config.MAX_WAIT_MS,
    )

    yield

    # Shutdown: stop runner, cancel tasks, drain queue
    _runner.stop()
    _runner_task.cancel()
    _metrics_task.cancel()
    try:
        await _runner_task
    except asyncio.CancelledError:
        pass
    try:
        await _metrics_task
    except asyncio.CancelledError:
        pass
    logger.info("DBIE shutdown complete.")


app = FastAPI(title="DBIE", version="0.1.0", lifespan=lifespan)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/infer", response_model=InferResponse)
async def infer(body: InferRequest) -> InferResponse:
    arrival = time.monotonic()

    # Validate payload dimension
    if len(body.data) != config.INPUT_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {config.INPUT_DIM} features, got {len(body.data)}",
        )

    tensor = torch.tensor(body.data, dtype=torch.float32)
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    request = Request(payload=tensor, future=future)

    # Enqueue — 429 on backpressure
    try:
        _queue.put(request)
    except QueueFullError:
        _metrics.record_rejection()
        raise HTTPException(status_code=429, detail="Server overloaded. Try again later.")

    # Notify adaptive scheduler of enqueue
    if isinstance(_scheduler, AdaptiveScheduler):
        _scheduler.record_enqueue()

    # Await result with timeout to prevent memory leaks from disconnected clients
    try:
        result: torch.Tensor = await asyncio.wait_for(
            future, timeout=config.REQUEST_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timed out.")

    total_ms = (time.monotonic() - arrival) * 1000.0
    wait_ms = (request.arrival_time - arrival) * 1000.0  # ~0 but keeps the pattern
    # The real wait is from arrival to when the batch picked it up;
    # approximate as total - inference
    _metrics.record_request_complete(total_ms, total_ms)

    return InferResponse(
        request_id=request.request_id,
        result=result.tolist(),
        latency_ms=round(total_ms, 3),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=config.INFERENCE_DEVICE,
        scheduler=config.SCHEDULER_STRATEGY,
        queue_size=_queue.size,
        batches_processed=_runner.batches_processed,
    )


@app.get("/metrics")
async def metrics():
    return _metrics.snapshot()
