#!/usr/bin/env python3
"""Check the Linux host under Qdrant for the memory problems that slow searches.

Read-only: it only reads /proc and /sys. Standard library only. Run it on the
Qdrant server (root not needed):

    python3 host_check.py            # finds the qdrant process itself
    python3 host_check.py --pid 1234

It answers the questions that the collection config can't:
  * Is Qdrant's memory being swapped out right now?
  * What does the cgroup (Docker/systemd) really allow? Docker's --memory-swap
    flag doesn't always reach cgroup v2, so read the files, not the flags.
  * Are swappiness / transparent huge pages set sensibly?
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

GB = 1024 ** 3


def read(path: str) -> Optional[str]:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def parse_kv_kb(text: str) -> Dict[str, int]:
    """Parse /proc/meminfo or /proc/<pid>/status style 'Key:   123 kB' lines into bytes."""
    out = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key.strip()] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
    return out


def find_qdrant_pid(root: str) -> Optional[int]:
    proc = os.path.join(root, "proc")
    for entry in sorted(os.listdir(proc)):
        if not entry.isdigit():
            continue
        if read(os.path.join(proc, entry, "comm")) == "qdrant":
            return int(entry)
    return None


def cgroup_limits(root: str, pid: int) -> Optional[dict]:
    text = read(os.path.join(root, "proc", str(pid), "cgroup"))
    if not text:
        return None
    # cgroup v2 has a single "0::/path" line.
    path = next((line.split("::", 1)[1] for line in text.splitlines() if line.startswith("0::")), None)
    if path is None:
        return {"version": 1}
    base = os.path.join(root, "sys", "fs", "cgroup", path.lstrip("/"))
    values = {}
    for name in ("memory.max", "memory.swap.max", "memory.current", "memory.swap.current"):
        raw = read(os.path.join(base, name))
        values[name] = None if raw is None else ("max" if raw == "max" else int(raw))
    return {"version": 2, "path": path, **values}


def thp_mode(text: Optional[str]) -> Optional[str]:
    """'always [madvise] never' -> 'madvise'."""
    if not text or "[" not in text:
        return None
    return text.split("[", 1)[1].split("]", 1)[0]


def check(root: str = "/", pid: Optional[int] = None) -> dict:
    meminfo = parse_kv_kb(read(os.path.join(root, "proc", "meminfo")) or "")
    swappiness = read(os.path.join(root, "proc", "sys", "vm", "swappiness"))
    thp = thp_mode(read(os.path.join(root, "sys", "kernel", "mm", "transparent_hugepage", "enabled")))
    pid = pid or find_qdrant_pid(root)
    proc_status = parse_kv_kb(read(os.path.join(root, "proc", str(pid), "status")) or "") if pid else {}
    cgroup = cgroup_limits(root, pid) if pid else None
    findings: List[dict] = []

    def flag(level: str, title: str, detail: str) -> None:
        findings.append({"level": level, "title": title, "detail": detail})

    total, avail = meminfo.get("MemTotal", 0), meminfo.get("MemAvailable", 0)
    swap_used = meminfo.get("SwapTotal", 0) - meminfo.get("SwapFree", 0)
    rss, qswap = proc_status.get("VmRSS", 0), proc_status.get("VmSwap", 0)

    if not pid:
        flag("medium", "no qdrant process found",
             "Pass --pid. In Docker the process is usually still named 'qdrant' on the host.")
    if qswap > 0.5 * GB:
        flag("high", f"Qdrant has {qswap / GB:.1f} GB swapped out",
             "Any graph or vector page that is in swap costs a disk read in the middle of a search. "
             "This is the usual cause of 'fast sometimes, 10+ seconds other times'. Reduce what "
             "Qdrant must keep resident (quantize, delete dead points, hnsw.on_disk=true) or add RAM.")
    if total and avail < total * 0.1:
        flag("medium", f"only {avail / GB:.1f} GB of {total / GB:.1f} GB RAM available",
             "The next big allocation (a model load, a deploy, a backup reading a large file) will "
             "push something into swap. Find out what, before it is Qdrant.")
    if cgroup and cgroup.get("memory.swap.max") == 0:
        flag("medium", "cgroup swap ban is active (memory.swap.max = 0)",
             "This pins everything Qdrant allocates in RAM. It helps latency only if the rest of the "
             "box can live without that RAM. On a shared box it starved a co-located embedding "
             "service until it was OOM-killed, which took the whole chat down. Use with care.")
    if cgroup and cgroup.get("memory.max") == "max" and cgroup.get("memory.swap.max") == "max":
        flag("low", "no cgroup memory or swap limit on the Qdrant process",
             "Fine if intended. If you passed Docker --memory / --memory-swap, they did not reach "
             "cgroup v2; set memory.max / memory.swap.max on the scope directly and re-check after "
             "every container recreate.")
    if swappiness is not None and int(swappiness) > 10:
        flag("low", f"vm.swappiness = {swappiness}",
             "A dedicated vector-search box usually wants 1-10 so the kernel prefers dropping page "
             "cache over swapping process memory.")
    if thp == "always":
        flag("low", "transparent huge pages = always",
             "'madvise' avoids latency spikes from background compaction on large memory-mapped "
             "files.")

    return {
        "pid": pid,
        "ram_gb": round(total / GB, 1), "available_gb": round(avail / GB, 1),
        "swap_used_gb": round(swap_used / GB, 1), "page_cache_gb": round(meminfo.get("Cached", 0) / GB, 1),
        "qdrant_rss_gb": round(rss / GB, 1), "qdrant_swap_gb": round(qswap / GB, 1),
        "swappiness": swappiness, "thp": thp, "cgroup": cgroup,
        "findings": sorted(findings, key=lambda f: {"high": 0, "medium": 1, "low": 2}[f["level"]]),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pid", type=int, help="qdrant process id (default: find it)")
    ap.add_argument("--root", default="/", help=argparse.SUPPRESS)  # for tests
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if not os.path.isdir(os.path.join(args.root, "proc")):
        print("This check reads /proc and /sys, so it runs on the Linux host that runs Qdrant.", file=sys.stderr)
        return 2
    report = check(args.root, args.pid)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"RAM {report['ram_gb']} GB, available {report['available_gb']} GB, page cache "
              f"{report['page_cache_gb']} GB, swap used {report['swap_used_gb']} GB")
        print(f"Qdrant pid {report['pid']}: RSS {report['qdrant_rss_gb']} GB, swapped {report['qdrant_swap_gb']} GB")
        print(f"swappiness {report['swappiness']}, THP {report['thp']}, cgroup {report['cgroup']}\n")
        for f in report["findings"]:
            print(f"[{f['level'].upper()}] {f['title']}\n    {f['detail']}\n")
        if not report["findings"]:
            print("No findings.")
    return 1 if any(f["level"] == "high" for f in report["findings"]) else 0


if __name__ == "__main__":
    sys.exit(main())
