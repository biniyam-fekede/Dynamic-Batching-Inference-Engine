"""Configuration for DBIE — all tunable parameters via environment variables."""

import os
import platform


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


# --- Batching ---
MAX_BATCH_SIZE: int = _env_int("MAX_BATCH_SIZE", 32)
MAX_WAIT_MS: float = _env_float("MAX_WAIT_MS", 50)
QUEUE_MAX_SIZE: int = _env_int("QUEUE_MAX_SIZE", 1000)

# --- Inference ---
INFERENCE_DEVICE: str = _env_str("INFERENCE_DEVICE", "cpu")

# --- Scheduler ---
SCHEDULER_STRATEGY: str = _env_str("SCHEDULER_STRATEGY", "fifo")

# --- Adaptive scheduler ---
ADAPTIVE_TARGET_LATENCY_MS: float = _env_float("ADAPTIVE_TARGET_LATENCY_MS", 100)
ADAPTIVE_EMA_ALPHA: float = _env_float("ADAPTIVE_EMA_ALPHA", 0.3)
ADAPTIVE_FILL_FACTOR: float = _env_float("ADAPTIVE_FILL_FACTOR", 0.8)
ADAPTIVE_MIN_BATCH_SIZE: int = _env_int("ADAPTIVE_MIN_BATCH_SIZE", 1)
ADAPTIVE_MIN_WAIT_MS: float = _env_float("ADAPTIVE_MIN_WAIT_MS", 5)
ADAPTIVE_HYSTERESIS_N: int = _env_int("ADAPTIVE_HYSTERESIS_N", 3)

# --- Server ---
REQUEST_TIMEOUT_S: float = _env_float("REQUEST_TIMEOUT_S", 30.0)
SERVER_HOST: str = _env_str("SERVER_HOST", "0.0.0.0")
SERVER_PORT: int = _env_int("SERVER_PORT", 8000)
WARMUP_BATCHES: int = _env_int("WARMUP_BATCHES", 10)

# --- Model ---
INPUT_DIM: int = _env_int("INPUT_DIM", 128)
OUTPUT_DIM: int = _env_int("OUTPUT_DIM", 64)

# --- Metrics ---
METRICS_WINDOW_SIZE: int = _env_int("METRICS_WINDOW_SIZE", 10000)
METRICS_SAMPLE_INTERVAL_MS: float = _env_float("METRICS_SAMPLE_INTERVAL_MS", 100)

# --- Platform-aware uvloop ---
USE_UVLOOP: bool = platform.system() != "Windows"
