"""Runs the tools against a real Qdrant when QDRANT_URL is set (skipped otherwise):

    docker run -p 6333:6333 qdrant/qdrant
    QDRANT_URL=http://localhost:6333 pytest -q
"""
import os

import pytest

from conftest import load_tool

URL = os.environ.get("QDRANT_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set QDRANT_URL to run against a live Qdrant")


def settled(audit, info) -> bool:
    """True once the optimizer is idle and any segment big enough to index is indexed.

    Qdrant only builds HNSW for segments larger than indexing_threshold (KB), so a small
    collection can stay at 0 indexed vectors forever. Waiting for every vector to be
    indexed would then time out, so mirror the audit: require indexing only when the
    collection holds more data than the threshold.
    """
    if info["status"] != "green":
        return False
    points = info.get("points_count") or 0
    indexed = info.get("indexed_vectors_count") or 0
    threshold_kb = info["config"].get("optimizer_config", {}).get("indexing_threshold")
    data_kb = sum(points * v["size"] * 4 for v in audit.vector_params(info["config"])) / 1024
    needs_index = threshold_kb != 0 and (threshold_kb is None or data_kb > threshold_kb)
    return not needs_index or indexed >= points * 0.9


def test_end_to_end_before_and_after(capsys):
    seed, audit, probe = load_tool("seed_demo"), load_tool("qdrant_audit"), load_tool("latency_probe")
    import time

    def wait_green(timeout_s=60):
        deadline = time.monotonic() + timeout_s
        while True:
            info = audit.get_json(f"{URL}/collections/pytest_demo")["result"]
            if settled(audit, info):
                return
            if time.monotonic() > deadline:
                pytest.fail(f"collection not settled after {timeout_s}s: status={info['status']} "
                            f"points={info.get('points_count')} indexed={info.get('indexed_vectors_count')}")
            time.sleep(0.5)

    assert seed.main(["--url", URL, "--collection", "pytest_demo", "--points", "3000", "--dim", "64"]) == 0
    wait_green()
    assert audit.main(["--url", URL, "--collection", "pytest_demo", "--ram-gb", "16", "--cpus", "4"]) == 1
    assert "NOT pinned in RAM" in capsys.readouterr().out

    assert seed.main(["--url", URL, "--collection", "pytest_demo", "--points", "3000", "--dim", "64", "--fixed"]) == 0
    wait_green()
    assert audit.main(["--url", URL, "--collection", "pytest_demo", "--ram-gb", "16", "--cpus", "4"]) == 0
    assert probe.main(["--url", URL, "--collection", "pytest_demo", "-n", "5"]) == 0
