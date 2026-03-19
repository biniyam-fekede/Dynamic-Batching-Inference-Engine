"""Standard analysis plots for DBIE benchmark results.

Generates the six plots specified in the design:
1. Throughput vs. Batch Size
2. Latency Percentile Fan
3. Time Series Under Ramp (FIFO vs. Adaptive P99)
4. Queue Depth Over Time
5. CPU vs GPU Throughput
6. Memory Scaling
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _load_results(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def plot_throughput_vs_batch_size(
    sweep_results: List[Dict],
    output_path: str = "analysis/throughput_vs_batch_size.png",
) -> None:
    """Plot 1: Throughput vs. Batch Size — shows diminishing returns."""
    batch_sizes = [r["batch_size_config"] for r in sweep_results]
    throughputs = [r["actual_rps"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(batch_sizes, throughputs, "o-", linewidth=2, markersize=8, color="#2196F3")
    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Throughput (requests/sec)", fontsize=12)
    ax.set_title("Throughput vs. Batch Size", fontsize=14)
    ax.set_xticks(batch_sizes)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_latency_percentile_fan(
    sweep_results: List[Dict],
    output_path: str = "analysis/latency_fan.png",
) -> None:
    """Plot 2: Latency percentiles (P50/P95/P99) vs. batch size."""
    batch_sizes = [r["batch_size_config"] for r in sweep_results]
    p50 = [r["latency_ms"]["p50"] for r in sweep_results]
    p95 = [r["latency_ms"]["p95"] for r in sweep_results]
    p99 = [r["latency_ms"]["p99"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(batch_sizes, p50, p99, alpha=0.15, color="#FF5722")
    ax.fill_between(batch_sizes, p50, p95, alpha=0.25, color="#FF9800")
    ax.plot(batch_sizes, p50, "o-", label="P50", linewidth=2, color="#4CAF50")
    ax.plot(batch_sizes, p95, "s--", label="P95", linewidth=2, color="#FF9800")
    ax.plot(batch_sizes, p99, "^:", label="P99", linewidth=2, color="#FF5722")
    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Latency Percentile Fan by Batch Size", fontsize=14)
    ax.set_xticks(batch_sizes)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_time_series_ramp(
    fifo_results: List[Dict[str, Any]],
    adaptive_results: List[Dict[str, Any]],
    output_path: str = "analysis/ramp_comparison.png",
) -> None:
    """Plot 3: FIFO vs. Adaptive P99 latency over time under ramp load.

    Expects lists of per-second summary dicts with keys: 'time_s', 'p99_ms'.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    if fifo_results:
        t = [r["time_s"] for r in fifo_results]
        p99 = [r["p99_ms"] for r in fifo_results]
        ax.plot(t, p99, label="FIFO P99", linewidth=2, color="#2196F3")

    if adaptive_results:
        t = [r["time_s"] for r in adaptive_results]
        p99 = [r["p99_ms"] for r in adaptive_results]
        ax.plot(t, p99, label="Adaptive P99", linewidth=2, color="#4CAF50")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("P99 Latency (ms)", fontsize=12)
    ax.set_title("FIFO vs. Adaptive Under Ramp Load", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_queue_depth_over_time(
    timestamps: List[float],
    depths: List[int],
    output_path: str = "analysis/queue_depth.png",
) -> None:
    """Plot 4: Queue depth over time — visualizes backpressure."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(timestamps, depths, alpha=0.3, color="#9C27B0")
    ax.plot(timestamps, depths, linewidth=1.5, color="#9C27B0")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Queue Depth", fontsize=12)
    ax.set_title("Queue Depth Over Time", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_cpu_vs_gpu_throughput(
    cpu_rps: Dict[int, float],
    gpu_rps: Dict[int, float],
    output_path: str = "analysis/cpu_vs_gpu.png",
) -> None:
    """Plot 5: CPU vs GPU throughput side-by-side bars with speedup annotation."""
    batch_sizes = sorted(set(cpu_rps.keys()) | set(gpu_rps.keys()))
    cpu_vals = [cpu_rps.get(bs, 0) for bs in batch_sizes]
    gpu_vals = [gpu_rps.get(bs, 0) for bs in batch_sizes]

    x = np.arange(len(batch_sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_cpu = ax.bar(x - width / 2, cpu_vals, width, label="CPU", color="#2196F3")
    bars_gpu = ax.bar(x + width / 2, gpu_vals, width, label="GPU", color="#4CAF50")

    # Annotate speedup
    for i, bs in enumerate(batch_sizes):
        cpu_v = cpu_rps.get(bs, 0)
        gpu_v = gpu_rps.get(bs, 0)
        if cpu_v > 0 and gpu_v > 0:
            speedup = gpu_v / cpu_v
            ax.annotate(
                f"{speedup:.1f}x",
                xy=(x[i] + width / 2, gpu_v),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Throughput (requests/sec)", fontsize=12)
    ax.set_title("CPU vs GPU Throughput", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_memory_scaling(
    input_dims: List[int],
    allocated_mb: List[float],
    reserved_mb: List[float],
    output_path: str = "analysis/memory_scaling.png",
) -> None:
    """Plot 6: GPU memory allocated vs reserved by input dimension."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(input_dims, allocated_mb, "o-", label="Allocated", linewidth=2, color="#FF5722")
    ax.plot(input_dims, reserved_mb, "s--", label="Reserved", linewidth=2, color="#9C27B0")
    ax.set_xlabel("Input Dimension", fontsize=12)
    ax.set_ylabel("GPU Memory (MB)", fontsize=12)
    ax.set_title("GPU Memory Scaling by Input Dimension", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_all_from_sweep(sweep_file: str = "benchmarks/sweep_results.json") -> None:
    """Generate plots 1 and 2 from sweep results."""
    results = _load_results(sweep_file)
    Path("analysis").mkdir(exist_ok=True)
    plot_throughput_vs_batch_size(results)
    plot_latency_percentile_fan(results)
    print("\nAll sweep-based plots generated.")


if __name__ == "__main__":
    generate_all_from_sweep()
