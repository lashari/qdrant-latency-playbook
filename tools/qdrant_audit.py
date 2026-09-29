#!/usr/bin/env python3
"""Audit a Qdrant collection for the settings that decide search latency.

Read-only: it only calls GET /collections/<name>. Standard library only, so you
can copy this one file to a server and run it.

    python3 qdrant_audit.py --url http://localhost:6333 --collection docs --ram-gb 15

It prints the memory-relevant config, estimates how much RAM the hot parts of the
index need, and flags the combinations that made a production collection slow
(see the README for the story behind each rule).
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import List, Optional

GB = 1024 ** 3
# The hot-set estimate is a lower bound (it leaves out payload, id tracker, rescoring reads
# and every other process), so warn once it passes half of RAM, not when it's nearly full.
HEADROOM = 0.5


def get_json(url: str, api_key: Optional[str] = None, timeout: float = 30) -> dict:
    req = urllib.request.Request(url, headers={"api-key": api_key} if api_key else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def vector_params(config: dict) -> List[dict]:
    """Normalise unnamed (`{"size":..}`) and named (`{"name": {"size":..}}`) vector configs."""
    vectors = config["params"]["vectors"]
    if "size" in vectors:
        return [dict(vectors, name="")]
    return [dict(v, name=k) for k, v in vectors.items()]


def estimate(points: int, dim: int, m: int, quantized: bool) -> dict:
    """Rough byte counts for the parts of an HNSW index that searches touch.

    - float32 vectors: points x dim x 4
    - int8 scalar quantized copy: points x dim x 1
    - HNSW graph: layer 0 keeps up to 2*m links per point at 4 bytes each; upper
      layers add ~10%.
    These are estimates for capacity planning, not exact on-disk sizes.
    """
    return {
        "vectors_f32": points * dim * 4,
        "vectors_int8": points * dim if quantized else 0,
        "hnsw_graph": int(points * 2 * m * 4 * 1.1),
    }


def audit(info: dict, ram_gb: Optional[float] = None, cpus: Optional[int] = None) -> dict:
    """Pure function: collection info JSON in, findings out. Easy to unit test."""
    config = info["config"]
    hnsw = config.get("hnsw_config", {})
    optimizer = config.get("optimizer_config", {})
    quant = config.get("quantization_config") or {}
    points = info.get("points_count") or 0
    indexed = info.get("indexed_vectors_count") or 0
    findings = []

    def flag(level: str, title: str, detail: str) -> None:
        findings.append({"level": level, "title": title, "detail": detail})

    rows = []
    for vec in vector_params(config):
        v_hnsw = dict(hnsw, **(vec.get("hnsw_config") or {}))
        v_quant = vec.get("quantization_config") or quant
        scalar = v_quant.get("scalar") if v_quant else None
        quant_kind = next(iter(v_quant), None) if v_quant else None
        always_ram = bool((v_quant.get(quant_kind) or {}).get("always_ram")) if quant_kind else False
        vectors_on_disk = bool(vec.get("on_disk"))
        hnsw_on_disk = bool(v_hnsw.get("on_disk"))
        m = v_hnsw.get("m") or 16
        est = estimate(points, vec["size"], m, quantized=bool(scalar))
        label = vec["name"] or "(default)"

        # What every search touches and therefore wants in RAM: the graph, plus whichever
        # vector copy is used for scoring (int8 if quantized, else float32). Float32
        # vectors not marked on_disk also sit in RAM even when a quantized copy exists.
        # (Binary/product quantization sizes aren't estimated.)
        hot = est["hnsw_graph"]
        if scalar:
            hot += est["vectors_int8"]
        if not quant_kind or not vectors_on_disk:
            hot += est["vectors_f32"]
        rows.append({
            "vector": label, "dim": vec["size"], "distance": vec.get("distance"),
            "vectors_on_disk": vectors_on_disk, "hnsw_on_disk": hnsw_on_disk, "hnsw_m": m,
            "quantization": quant_kind, "quantization_always_ram": always_ram,
            "est_gb": {k: round(b / GB, 2) for k, b in est.items()}, "hot_set_gb": round(hot / GB, 2),
        })

        if vectors_on_disk and not quant_kind:
            flag("high", f"[{label}] vectors on disk with no quantization",
                 "Every candidate the graph visits is scored against full vectors read from disk. "
                 "Add scalar int8 quantization with always_ram=true so scoring happens in RAM and "
                 "only the final top-k are rescored from disk.")
        if quant_kind and not always_ram:
            flag("high", f"[{label}] quantized vectors are NOT pinned in RAM (always_ram=false)",
                 "The quantized copy exists but lives on disk, so searches still page it in. Flipping "
                 "always_ram to true was the single biggest win in the case study (12-21 s -> 0.07 s). "
                 "Check the hot-set estimate fits in RAM first.")
        if not hnsw_on_disk and ram_gb and hot / GB > ram_gb * HEADROOM:
            flag("high", f"[{label}] HNSW graph in RAM on a box that can't hold the hot set",
                 "With hnsw.on_disk=false the graph is anonymous memory. Under pressure the kernel swaps "
                 "it, and swapped graph pages are the slowest possible case. Either get RAM headroom or "
                 "use hnsw.on_disk=true so the graph is mmapped and evictable as page cache instead.")
        if ram_gb and hot / GB > ram_gb * 0.8:
            flag("high", f"[{label}] hot set ~{hot / GB:.1f} GB vs {ram_gb:g} GB RAM",
                 "The parts every search touches don't fit comfortably. Expect swap or page-cache "
                 "thrash. Shrink the collection (delete dead points), quantize, or add RAM.")
        elif ram_gb and hot / GB > ram_gb * HEADROOM:
            flag("medium", f"[{label}] hot set ~{hot / GB:.1f} GB is over half of {ram_gb:g} GB RAM",
                 "The estimate is a lower bound, and anything else on the box (embedding model, "
                 "backups, app workers) competes for the rest. Watch swap and keep big sequential "
                 "reads out of the page cache.")

    status = info.get("status")
    if status != "green":
        flag("medium", f"collection status is {status}",
             "The optimizer is working (yellow) or paused (grey). Segment rewrites compete with "
             "queries for CPU and disk, so latency measured now is not your steady state.")
    # Qdrant only builds HNSW for segments bigger than indexing_threshold (KB), so a small
    # collection with 0 indexed vectors is normal. Only flag when there is enough data.
    threshold_kb = optimizer.get("indexing_threshold")
    data_kb = sum(points * v["size"] * 4 for v in vector_params(config)) / 1024
    big_enough = threshold_kb is None or data_kb > threshold_kb
    if points and indexed < points * 0.9 and threshold_kb != 0 and big_enough:
        flag("high", f"only {indexed:,} of {points:,} vectors are indexed",
             "Unindexed segments are searched by brute force. Usually the optimizer is still "
             "catching up. If it stays like this, check indexing_threshold and optimizer errors.")
    threads = optimizer.get("max_optimization_threads")
    if threads == 0:
        flag("high", "optimizations are disabled (max_optimization_threads=0)",
             "0 means no optimizer at all (null means auto). New data is never indexed or merged, "
             "so searches over it stay brute force. Set it to 1 on small boxes, or null.")
    elif cpus and cpus <= 4 and (threads is None or threads > 1):
        label = "null (auto: saturate CPUs)" if threads is None else threads
        flag("medium", f"max_optimization_threads={label} on a {cpus}-CPU box",
             "On a 4 vCPU box, 2 optimizer threads took warm queries from under 1 ms to seconds "
             "within a minute. While serving live traffic, keep it at 1 and give rebuilds time "
             "instead of threads.")
    segments = info.get("segments_count") or 0
    if segments and cpus and segments > max(cpus * 4, 16):
        flag("low", f"{segments} segments",
             "Each search runs per segment and merges results. Many small segments cost CPU per "
             "query. They merge over time once the optimizer is idle.")

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: order[f["level"]])
    return {
        "status": status, "points": points, "indexed": indexed, "segments": segments,
        "optimizer_threads": threads, "vectors": rows, "findings": findings,
    }


def print_report(name: str, report: dict, ram_gb: Optional[float]) -> None:
    print(f"Collection: {name}   status: {report['status']}   points: {report['points']:,}   "
          f"indexed: {report['indexed']:,}   segments: {report['segments']}")
    for row in report["vectors"]:
        print(f"\nVector {row['vector']}: dim={row['dim']} {row['distance']}")
        print(f"  vectors on_disk     : {row['vectors_on_disk']}")
        print(f"  hnsw on_disk        : {row['hnsw_on_disk']}  (m={row['hnsw_m']})")
        print(f"  quantization        : {row['quantization'] or 'none'}"
              + (f"  always_ram={row['quantization_always_ram']}" if row["quantization"] else ""))
        est = row["est_gb"]
        print(f"  est. size (GB)      : f32 vectors {est['vectors_f32']}, int8 {est['vectors_int8']}, "
              f"graph {est['hnsw_graph']}")
        print(f"  est. hot set (GB)   : {row['hot_set_gb']}" + (f"  of {ram_gb:g} GB RAM" if ram_gb else "")
              + "  (lower bound)")
    print()
    if not report["findings"]:
        print("No findings. If searches are still slow, run latency_probe.py and host_check.py.")
    for f in report["findings"]:
        print(f"[{f['level'].upper()}] {f['title']}\n    {f['detail']}\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--api-key", help="Qdrant API key, if set")
    ap.add_argument("--ram-gb", type=float, help="host RAM, to compare against the hot-set estimate")
    ap.add_argument("--cpus", type=int, help="host vCPUs, to judge optimizer thread settings")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)
    try:
        info = get_json(f"{args.url.rstrip('/')}/collections/{args.collection}", args.api_key)["result"]
    except urllib.error.HTTPError as e:
        print(f"Qdrant returned HTTP {e.code} for collection '{args.collection}'", file=sys.stderr)
        return 2
    except urllib.error.URLError as e:
        print(f"Could not reach Qdrant at {args.url}: {e.reason}", file=sys.stderr)
        return 2
    report = audit(info, args.ram_gb, args.cpus)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(args.collection, report, args.ram_gb)
    return 1 if any(f["level"] == "high" for f in report["findings"]) else 0


if __name__ == "__main__":
    sys.exit(main())
