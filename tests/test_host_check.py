from pathlib import Path

from conftest import load_tool

host = load_tool("host_check")
GB = 1024 ** 3


def fake_host(tmp: Path, *, swap_kb=0, swap_max="max", mem_max="max", swappiness="60",
              thp="always [madvise] never", avail_kb=6_000_000) -> str:
    """Build a minimal /proc + /sys tree like a Docker host running Qdrant (cgroup v2)."""
    (tmp / "proc/sys/vm").mkdir(parents=True)
    (tmp / "proc/meminfo").write_text(
        f"MemTotal:       15728640 kB\nMemAvailable:   {avail_kb} kB\nCached:  3000000 kB\n"
        "SwapTotal:      24117248 kB\nSwapFree:       16000000 kB\n")
    (tmp / "proc/sys/vm/swappiness").write_text(swappiness + "\n")
    (tmp / "proc/42").mkdir()
    (tmp / "proc/42/comm").write_text("bash\n")
    (tmp / "proc/777").mkdir()
    (tmp / "proc/777/comm").write_text("qdrant\n")
    (tmp / "proc/777/status").write_text(f"Name:\tqdrant\nVmRSS:\t 11000000 kB\nVmSwap:\t {swap_kb} kB\n")
    scope = "system.slice/docker-abc123.scope"
    (tmp / "proc/777/cgroup").write_text(f"0::/{scope}\n")
    cg = tmp / "sys/fs/cgroup" / scope
    cg.mkdir(parents=True)
    (cg / "memory.max").write_text(mem_max + "\n")
    (cg / "memory.swap.max").write_text(swap_max + "\n")
    (cg / "memory.current").write_text(str(11 * GB) + "\n")
    (cg / "memory.swap.current").write_text(str(swap_kb * 1024) + "\n")
    (tmp / "sys/kernel/mm/transparent_hugepage").mkdir(parents=True)
    (tmp / "sys/kernel/mm/transparent_hugepage/enabled").write_text(thp + "\n")
    return str(tmp)


def titles(report):
    return [f["title"] for f in report["findings"]]


def test_finds_qdrant_and_reports_swap(tmp_path):
    report = host.check(fake_host(tmp_path, swap_kb=8_800_000))
    assert report["pid"] == 777
    assert report["qdrant_swap_gb"] == 8.4
    assert report["findings"][0]["level"] == "high"
    assert "swapped out" in report["findings"][0]["title"]


def test_healthy_host(tmp_path):
    report = host.check(fake_host(tmp_path, swappiness="1", mem_max=str(12 * GB), swap_max=str(4 * GB)))
    assert report["findings"] == []
    assert report["thp"] == "madvise"
    assert report["cgroup"]["memory.max"] == 12 * GB


def test_swap_ban_is_called_out(tmp_path):
    report = host.check(fake_host(tmp_path, swap_max="0", swappiness="1"))
    assert any("swap ban" in t for t in titles(report))


def test_docker_flags_that_never_reached_cgroup(tmp_path):
    report = host.check(fake_host(tmp_path, swappiness="1"))
    assert any("no cgroup memory or swap limit" in t for t in titles(report))


def test_low_available_memory_and_thp_always(tmp_path):
    report = host.check(fake_host(tmp_path, avail_kb=900_000, thp="[always] madvise never", swappiness="1",
                                  mem_max=str(12 * GB), swap_max=str(4 * GB)))
    assert any("available" in t for t in titles(report))
    assert any("transparent huge pages" in t for t in titles(report))


def test_parse_kv_kb():
    assert host.parse_kv_kb("VmSwap:\t 2048 kB\nThreads: 12\n") == {"VmSwap": 2048 * 1024, "Threads": 12}
