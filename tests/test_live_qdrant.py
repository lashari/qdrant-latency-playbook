"""Runs the tools against a real Qdrant when QDRANT_URL is set (skipped otherwise):

    docker run -p 6333:6333 qdrant/qdrant
    QDRANT_URL=http://localhost:6333 pytest -q
"""
import os

import pytest

from conftest import load_tool

URL = os.environ.get("QDRANT_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set QDRANT_URL to run against a live Qdrant")


def test_end_to_end_before_and_after(capsys):
    seed, audit, probe = load_tool("seed_demo"), load_tool("qdrant_audit"), load_tool("latency_probe")
    import time

    def wait_green():
        for _ in range(60):
            info = audit.get_json(f"{URL}/collections/pytest_demo")["result"]
            if info["status"] == "green" and info.get("indexed_vectors_count") == info["points_count"]:
                return
            time.sleep(1)

    assert seed.main(["--url", URL, "--collection", "pytest_demo", "--points", "3000", "--dim", "64"]) == 0
    wait_green()
    assert audit.main(["--url", URL, "--collection", "pytest_demo", "--ram-gb", "16", "--cpus", "4"]) == 1
    assert "NOT pinned in RAM" in capsys.readouterr().out

    assert seed.main(["--url", URL, "--collection", "pytest_demo", "--points", "3000", "--dim", "64", "--fixed"]) == 0
    wait_green()
    assert audit.main(["--url", URL, "--collection", "pytest_demo", "--ram-gb", "16", "--cpus", "4"]) == 0
    assert probe.main(["--url", URL, "--collection", "pytest_demo", "-n", "5"]) == 0
