"""Metrics collection with bounded rolling windows."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np

from dbie import config


class RollingMetric:
    """A single metric backed by a bounded deque."""

    def __init__(self, maxlen: int = config.METRICS_WINDOW_SIZE) -> None:
        self._values: Deque[float] = deque(maxlen=maxlen)

    def record(self, value: float) -> None:
        self._values.append(value)

    @property
    def count(self) -> int:
        return len(self._values)

    def percentiles(self) -> Dict[str, float]:
        if not self._values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "count": 0}
        arr = np.array(self._values)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(np.mean(arr)),
            "count": len(arr),
        }

    def last(self) -> Optional[float]:
        return self._values[-1] if self._values else None


@dataclass
class MetricsCollector:
    """Central metrics store for the inference engine."""

    queue_depth: RollingMetric = field(default_factory=RollingMetric)
    queue_wait_ms: RollingMetric = field(default_factory=RollingMetric)
    batch_size: RollingMetric = field(default_factory=RollingMetric)
    inference_ms: RollingMetric = field(default_factory=RollingMetric)
    total_request_ms: RollingMetric = field(default_factory=RollingMetric)
    gpu_memory_mb: RollingMetric = field(default_factory=RollingMetric)

    # Counters
    total_requests: int = 0
    total_batches: int = 0
    rejected_requests: int = 0
    _start_time: float = field(default_factory=time.monotonic)

    def record_batch(self, batch_size: int, inference_ms: float) -> None:
        self.batch_size.record(batch_size)
        self.inference_ms.record(inference_ms)
        self.total_batches += 1

    def record_request_complete(self, total_ms: float, wait_ms: float) -> None:
        self.total_request_ms.record(total_ms)
        self.queue_wait_ms.record(wait_ms)
        self.total_requests += 1

    def record_rejection(self) -> None:
        self.rejected_requests += 1

    def record_queue_depth(self, depth: int) -> None:
        self.queue_depth.record(float(depth))

    def record_gpu_memory(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                mem_bytes = torch.cuda.memory_allocated()
                self.gpu_memory_mb.record(mem_bytes / (1024 * 1024))
        except Exception:
            pass

    def snapshot(self) -> Dict:
        uptime = time.monotonic() - self._start_time
        throughput = self.total_requests / uptime if uptime > 0 else 0.0
        return {
            "uptime_s": round(uptime, 2),
            "throughput_rps": round(throughput, 2),
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "rejected_requests": self.rejected_requests,
            "queue_depth": self.queue_depth.percentiles(),
            "queue_wait_ms": self.queue_wait_ms.percentiles(),
            "batch_size": self.batch_size.percentiles(),
            "inference_ms": self.inference_ms.percentiles(),
            "total_request_ms": self.total_request_ms.percentiles(),
            "gpu_memory_mb": self.gpu_memory_mb.percentiles(),
        }
