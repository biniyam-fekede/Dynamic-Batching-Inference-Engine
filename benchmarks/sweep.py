"""Batch size sweep benchmark — runs the server with different MAX_BATCH_SIZE values.

Usage: python -m benchmarks.sweep
Requires the server to NOT be running; this script starts/stops it for each config.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List

import httpx

from benchmarks.load_generator import LoadGenerator


async def wait_for_health(base_url: str, timeout: float = 30.0) -> bool:
    """Poll /health until the server is ready."""
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
    return False


async def run_sweep(
    batch_sizes: List[int] | None = None,
    rps: float = 100,
    duration_s: float = 15,
    warmup_s: float = 10,
    base_url: str = "http://localhost:8000",
    output_file: str = "benchmarks/sweep_results.json",
) -> List[Dict]:
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32]

    all_results: List[Dict] = []

    for bs in batch_sizes:
        print(f"\n{'='*50}")
        print(f"Batch size = {bs}")
        print(f"{'='*50}")

        env = os.environ.copy()
        env["MAX_BATCH_SIZE"] = str(bs)
        env["WARMUP_BATCHES"] = "5"

        # Start server
        proc = subprocess.Popen(
            [sys.executable, "-m", "dbie"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            if not await wait_for_health(base_url):
                print(f"  Server failed to start for batch_size={bs}")
                proc.kill()
                continue

            # Discard warmup period
            print(f"  Warming up for {warmup_s}s ...")
            gen = LoadGenerator(base_url=base_url)
            await gen.constant_rate(rps=rps * 0.5, duration_s=warmup_s, name="warmup")

            # Actual benchmark
            print(f"  Benchmarking @ {rps} rps for {duration_s}s ...")
            result = await gen.constant_rate(
                rps=rps, duration_s=duration_s, name=f"batch_{bs}"
            )

            # Fetch server-side metrics
            async with httpx.AsyncClient() as client:
                metrics_resp = await client.get(f"{base_url}/metrics")
                server_metrics = metrics_resp.json() if metrics_resp.status_code == 200 else {}

            summary = result.summary()
            summary["batch_size_config"] = bs
            summary["server_metrics"] = server_metrics
            all_results.append(summary)
            print(json.dumps(summary, indent=2))

        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSweep results saved to {output_file}")

    return all_results


if __name__ == "__main__":
    asyncio.run(run_sweep())
