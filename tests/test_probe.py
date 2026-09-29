import random

from conftest import load_tool

probe = load_tool("latency_probe")


def test_summary_separates_cold_from_warm():
    s = probe.summarize([900.0, 10, 12, 11, 50, 9], errors=1)
    assert s["cold_ms"] == 900.0
    assert s["warm_p50_ms"] == 11
    assert s["warm_max_ms"] == 50
    assert s["queries"] == 7 and s["errors"] == 1


def test_percentile_edges():
    values = list(range(1, 101))
    assert probe.percentile(values, 95) == 95
    assert probe.percentile([5.0], 95) == 5.0


def test_random_vectors_are_unit_length():
    v = probe.random_unit_vector(128, random.Random(1))
    assert len(v) == 128 and abs(sum(x * x for x in v) - 1) < 1e-9


def test_named_vector_size():
    info = {"config": {"params": {"vectors": {"dense": {"size": 1024}}}}}
    assert probe.vector_size(info, "dense") == 1024
