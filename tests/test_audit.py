import copy

from conftest import fixture, load_tool

audit_mod = load_tool("qdrant_audit")
audit = audit_mod.audit
GB = 1024 ** 3


def titles(report):
    return [f["title"] for f in report["findings"]]


def levels(report):
    return {f["level"] for f in report["findings"]}


def scaled(info, points, **optimizer):
    """Real collection-info JSON from Qdrant 1.19, resized to production-like numbers."""
    info = copy.deepcopy(info)
    info["points_count"] = info["indexed_vectors_count"] = points
    info["config"]["params"]["vectors"]["size"] = 1024
    info["config"]["optimizer_config"].update(optimizer)
    return info


def test_before_config_flags_unpinned_quantization():
    report = audit(fixture("collection_before.json"), ram_gb=16, cpus=4)
    assert any("NOT pinned in RAM" in t for t in titles(report))
    assert any("max_optimization_threads=null" in t for t in titles(report))


def test_tuned_config_is_clean():
    report = audit(fixture("collection_tuned.json"), ram_gb=16, cpus=4)
    assert report["findings"] == []


def test_case_study_numbers_flag_a_box_that_cannot_hold_the_hot_set():
    # ~8.2M x 1024-dim, int8 in RAM, graph in RAM, 15 GB box: the situation from the README.
    info = scaled(fixture("collection_tuned.json"), 8_190_000, max_optimization_threads=1)
    report = audit(info, ram_gb=15, cpus=4)
    row = report["vectors"][0]
    assert 7.5 < row["est_gb"]["vectors_int8"] < 8.0      # 8.19M x 1024 bytes
    assert 31 < row["est_gb"]["vectors_f32"] < 32
    assert any("HNSW graph in RAM on a box that can't hold" in t for t in titles(report))
    assert "high" in levels(report)


def test_same_data_on_a_bigger_box_is_fine():
    info = scaled(fixture("collection_tuned.json"), 8_190_000, max_optimization_threads=1)
    assert audit(info, ram_gb=64, cpus=8)["findings"] == []


def test_vectors_on_disk_without_quantization():
    info = copy.deepcopy(fixture("collection_tuned.json"))
    info["config"]["quantization_config"] = None
    assert any("no quantization" in t for t in titles(audit(info)))


def test_zero_threads_means_optimizer_disabled_not_auto():
    info = scaled(fixture("collection_tuned.json"), 20_000, max_optimization_threads=0)
    assert any("optimizations are disabled" in t for t in titles(audit(info, cpus=4)))


def test_unindexed_vectors_flagged_only_above_indexing_threshold():
    info = copy.deepcopy(fixture("collection_tuned.json"))
    info["indexed_vectors_count"] = 0
    assert any("are indexed" in t for t in titles(audit(info)))      # 20k x 256 x 4B = 20 MB > 1 MB threshold
    info["config"]["optimizer_config"]["indexing_threshold"] = 100_000   # 100 MB: too small to index yet
    assert not any("are indexed" in t for t in titles(audit(info)))


def test_named_vectors_are_audited_separately():
    info = copy.deepcopy(fixture("collection_tuned.json"))
    info["config"]["params"]["vectors"] = {
        "dense": {"size": 1024, "distance": "Cosine", "on_disk": True},
        "title": {"size": 384, "distance": "Cosine", "on_disk": True, "quantization_config": {"scalar": {"type": "int8", "always_ram": False}}},
    }
    report = audit(info)
    assert [r["vector"] for r in report["vectors"]] == ["dense", "title"]
    flagged = titles(report)
    assert any(t.startswith("[title] quantized vectors are NOT pinned") for t in flagged)
    assert not any(t.startswith("[dense] quantized") for t in flagged)


def test_findings_sorted_most_severe_first():
    info = scaled(fixture("collection_before.json"), 8_190_000)
    info["status"] = "yellow"
    order = [f["level"] for f in audit(info, ram_gb=15, cpus=4)["findings"]]
    assert order == sorted(order, key={"high": 0, "medium": 1, "low": 2}.get)


def test_live_wait_condition_matches_audit_threshold():
    """The live test's wait must not demand indexing below indexing_threshold (it never happens)."""
    import test_live_qdrant as live
    audit = audit_mod

    def info(points, indexed, dim=64, threshold=1000, status="green"):
        return {"status": status, "points_count": points, "indexed_vectors_count": indexed,
                "config": {"params": {"vectors": {"size": dim}},
                           "optimizer_config": {"indexing_threshold": threshold}}}

    # 3000 x 64 floats = 750 KB < 1000 KB threshold: never indexed, must count as settled.
    assert live.settled(audit, info(3000, 0))
    # Above the threshold, wait for indexing.
    assert not live.settled(audit, info(20000, 0, dim=256))
    assert live.settled(audit, info(20000, 19000, dim=256))
    # Optimizer still busy.
    assert not live.settled(audit, info(3000, 0, status="yellow"))
