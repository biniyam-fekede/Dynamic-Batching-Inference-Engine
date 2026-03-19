"""Open-loop load generator for DBIE benchmarking.

Open-loop means requests are sent on a schedule independent of response
arrival — this avoids coordinated omission where slow responses reduce
offered load and under-measure true latency.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

from dbie import config


@dataclass
class RequestResult:
    sent_at: float
    received_at: float
    latency_ms: float
    status_code: int
    success: bool


@dataclass
class BenchmarkResult:
    name: str
    config_desc: str
    target_rps: float
    duration_s: float
    results: List[RequestResult] = field(default_factory=list)

    @property
    def successful(self) -> List[RequestResult]:
        return [r for r in self.results if r.success]

    @property
    def failed(self) -> List[RequestResult]:
        return [r for r in self.results if not r.success]

    def summary(self) -> Dict:
        latencies = [r.latency_ms for r in self.successful]
        if not latencies:
            return {"error": "No successful requests"}
        latencies.sort()
        n = len(latencies)
        return {
            "name": self.name,
            "config": self.config_desc,
            "target_rps": self.target_rps,
            "duration_s": self.duration_s,
            "total_requests": len(self.results),
            "successful": len(self.successful),
            "failed": len(self.failed),
            "actual_rps": round(len(self.successful) / self.duration_s, 2),
            "latency_ms": {
                "p50": latencies[int(n * 0.50)],
                "p95": latencies[int(n * 0.95)],
                "p99": latencies[min(int(n * 0.99), n - 1)],
                "mean": round(sum(latencies) / n, 3),
                "min": latencies[0],
                "max": latencies[-1],
            },
        }


class LoadGenerator:
    """Sends requests at a controlled rate, independent of response time."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        input_dim: int = config.INPUT_DIM,
    ) -> None:
        self._base_url = base_url
        self._input_dim = input_dim

    def _random_payload(self) -> Dict:
        return {"data": [random.gauss(0, 1) for _ in range(self._input_dim)]}

    async def _send_request(
        self, client: httpx.AsyncClient, results: List[RequestResult]
    ) -> None:
        payload = self._random_payload()
        sent_at = time.monotonic()
        try:
            resp = await client.post(f"{self._base_url}/infer", json=payload)
            received_at = time.monotonic()
            results.append(
                RequestResult(
                    sent_at=sent_at,
                    received_at=received_at,
                    latency_ms=(received_at - sent_at) * 1000,
                    status_code=resp.status_code,
                    success=resp.status_code == 200,
                )
            )
        except Exception:
            received_at = time.monotonic()
            results.append(
                RequestResult(
                    sent_at=sent_at,
                    received_at=received_at,
                    latency_ms=(received_at - sent_at) * 1000,
                    status_code=0,
                    success=False,
                )
            )

    async def constant_rate(
        self, rps: float, duration_s: float, name: str = "constant"
    ) -> BenchmarkResult:
        """Send requests at a fixed rate for the given duration."""
        interval = 1.0 / rps
        results: List[RequestResult] = []
        end_time = time.monotonic() + duration_s
        tasks: List[asyncio.Task] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.monotonic() < end_time:
                task = asyncio.create_task(self._send_request(client, results))
                tasks.append(task)
                await asyncio.sleep(interval)
            # Wait for all in-flight requests to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        return BenchmarkResult(
            name=name,
            config_desc=f"constant@{rps}rps",
            target_rps=rps,
            duration_s=duration_s,
            results=results,
        )

    async def ramp(
        self,
        start_rps: float,
        end_rps: float,
        duration_s: float,
        name: str = "ramp",
    ) -> BenchmarkResult:
        """Linearly ramp request rate from start_rps to end_rps."""
        results: List[RequestResult] = []
        tasks: List[asyncio.Task] = []
        start_time = time.monotonic()
        end_time = start_time + duration_s

        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.monotonic() < end_time:
                elapsed = time.monotonic() - start_time
                progress = elapsed / duration_s
                current_rps = start_rps + (end_rps - start_rps) * progress
                interval = 1.0 / max(current_rps, 0.1)

                task = asyncio.create_task(self._send_request(client, results))
                tasks.append(task)
                await asyncio.sleep(interval)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        return BenchmarkResult(
            name=name,
            config_desc=f"ramp@{start_rps}->{end_rps}rps/{duration_s}s",
            target_rps=end_rps,
            duration_s=duration_s,
            results=results,
        )

    async def burst(
        self,
        burst_size: int,
        pause_s: float,
        num_bursts: int,
        name: str = "burst",
    ) -> BenchmarkResult:
        """Send bursts of requests with pauses between them."""
        results: List[RequestResult] = []
        start = time.monotonic()

        async with httpx.AsyncClient(timeout=30.0) as client:
            for _ in range(num_bursts):
                tasks = [
                    asyncio.create_task(self._send_request(client, results))
                    for _ in range(burst_size)
                ]
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(pause_s)

        duration = time.monotonic() - start
        return BenchmarkResult(
            name=name,
            config_desc=f"burst@{burst_size}x{num_bursts}",
            target_rps=burst_size / pause_s if pause_s > 0 else float("inf"),
            duration_s=duration,
            results=results,
        )

    async def step(
        self,
        rps_levels: List[float],
        step_duration_s: float,
        name: str = "step",
    ) -> BenchmarkResult:
        """Step through different request rates."""
        results: List[RequestResult] = []
        total_duration = 0.0

        async with httpx.AsyncClient(timeout=30.0) as client:
            for rps in rps_levels:
                interval = 1.0 / max(rps, 0.1)
                end_time = time.monotonic() + step_duration_s
                tasks: List[asyncio.Task] = []
                while time.monotonic() < end_time:
                    task = asyncio.create_task(self._send_request(client, results))
                    tasks.append(task)
                    await asyncio.sleep(interval)
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                total_duration += step_duration_s

        return BenchmarkResult(
            name=name,
            config_desc=f"step@{rps_levels}",
            target_rps=max(rps_levels),
            duration_s=total_duration,
            results=results,
        )


async def run_benchmark_suite(
    base_url: str = "http://localhost:8000",
    output_file: Optional[str] = None,
) -> List[Dict]:
    """Run the standard benchmark suite and return summaries."""
    gen = LoadGenerator(base_url=base_url)
    all_results: List[Dict] = []

    print("=" * 60)
    print("DBIE Benchmark Suite")
    print("=" * 60)

    # A — Constant rate baseline
    print("\n[A] Constant rate @ 50 rps for 15s ...")
    result = await gen.constant_rate(rps=50, duration_s=15, name="A_constant")
    summary = result.summary()
    all_results.append(summary)
    print(json.dumps(summary, indent=2))

    # B — Burst test
    print("\n[B] Burst: 64 requests x 5 bursts, 1s pause ...")
    result = await gen.burst(burst_size=64, pause_s=1.0, num_bursts=5, name="B_burst")
    summary = result.summary()
    all_results.append(summary)
    print(json.dumps(summary, indent=2))

    # C — Ramp test
    print("\n[C] Ramp: 10 -> 200 rps over 20s ...")
    result = await gen.ramp(start_rps=10, end_rps=200, duration_s=20, name="C_ramp")
    summary = result.summary()
    all_results.append(summary)
    print(json.dumps(summary, indent=2))

    # D — Step function
    print("\n[D] Step: [20, 50, 100, 50, 20] rps, 5s each ...")
    result = await gen.step(
        rps_levels=[20, 50, 100, 50, 20], step_duration_s=5, name="D_step"
    )
    summary = result.summary()
    all_results.append(summary)
    print(json.dumps(summary, indent=2))

    if output_file:
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return all_results


if __name__ == "__main__":
    asyncio.run(run_benchmark_suite(output_file="benchmarks/results.json"))
