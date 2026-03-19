"""Tests for configuration module."""

import os
from unittest import mock

import pytest


def test_defaults():
    from dbie import config

    assert config.MAX_BATCH_SIZE == 32
    assert config.MAX_WAIT_MS == 50
    assert config.QUEUE_MAX_SIZE == 1000
    assert config.INFERENCE_DEVICE == "cpu"
    assert config.SCHEDULER_STRATEGY == "fifo"


def test_env_override():
    with mock.patch.dict(os.environ, {"MAX_BATCH_SIZE": "64", "MAX_WAIT_MS": "100"}):
        # Re-evaluate the env helpers
        from dbie.config import _env_int, _env_float

        assert _env_int("MAX_BATCH_SIZE", 32) == 64
        assert _env_float("MAX_WAIT_MS", 50) == 100.0


def test_uvloop_gated():
    import platform

    from dbie import config

    if platform.system() == "Windows":
        assert config.USE_UVLOOP is False
    else:
        assert config.USE_UVLOOP is True
