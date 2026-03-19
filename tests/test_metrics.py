"""Tests for metrics collection."""

import pytest

from dbie.metrics import MetricsCollector, RollingMetric


def test_rolling_metric_empty():
    m = RollingMetric(maxlen=100)
    stats = m.percentiles()
    assert stats["count"] == 0
    assert m.last() is None


def test_rolling_metric_records():
    m = RollingMetric(maxlen=100)
    for i in range(50):
        m.record(float(i))
    stats = m.percentiles()
    assert stats["count"] == 50
    assert stats["p50"] == pytest.approx(24.5, abs=1)
    assert m.last() == 49.0


def test_rolling_metric_bounded():
    m = RollingMetric(maxlen=10)
    for i in range(20):
        m.record(float(i))
    assert m.count == 10
    # Only last 10 values (10-19)
    assert m.last() == 19.0


def test_collector_snapshot():
    c = MetricsCollector()
    c.record_batch(8, 5.0)
    c.record_batch(16, 8.0)
    c.record_request_complete(10.0, 3.0)

    snap = c.snapshot()
    assert snap["total_batches"] == 2
    assert snap["total_requests"] == 1
    assert snap["rejected_requests"] == 0
    assert snap["batch_size"]["count"] == 2


def test_collector_rejection_count():
    c = MetricsCollector()
    c.record_rejection()
    c.record_rejection()
    assert c.rejected_requests == 2
