#!/usr/bin/env python3
"""Measure real search latency against a Qdrant collection.

Read-only: it only runs searches. Standard library only.

    python3 latency_probe.py --url http://localhost:6333 --collection docs -n 20
    python3 latency_probe.py ... --vectors queries.json   # your real query embeddings

Reports the first query separately (the "cold" one, which pays for any pages
that were swapped or evicted) and p50/p95/max for the rest. Random vectors are
a worst case: they land in random parts of the graph, so they touch more cold
pages than real traffic does. Use --vectors with real query embeddings for
numbers you can compare with production.

Tip: append --json to a log from cron after anything that churns memory (backups,
deploys, re-indexing) and you will see regressions the day they start.
"""
import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import List, Optional


def call(url: str, body: Optional[dict] = None, api_key: Optional[str] = None, timeout: float = 60) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def vector_size(info: dict, using: Optional[str]) -> int:
    vectors = info["config"]["params"]["vectors"]
    if "size" in vectors:
        return vectors["size"]
    if not using:
        raise SystemExit(f"collection has named vectors {sorted(vectors)}; pass --using NAME")
    return vectors[using]["size"]


def percentile(values: List[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def summarize(timings_ms: List[float], errors: int) -> dict:
    cold, warm = timings_ms[0] if timings_ms else None, timings_ms[1:]
    result = {"queries": len(timings_ms) + errors, "errors": errors, "cold_ms": cold}
    if warm:
        result.update({
            "warm_p50_ms": round(statistics.median(warm), 1),
            "warm_p95_ms": round(percentile(warm, 95), 1),
            "warm_max_ms": round(max(warm), 1),
        })
    return result


def random_unit_vector(dim: int, rng: random.Random) -> List[float]:
    v = [rng.gauss(0, 1) for _ in range(dim)]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--api-key")
    ap.add_argument("--using", help="named vector to search, if the collection has several")
    ap.add_argument("-n", "--queries", type=int, default=20)
    ap.add_argument("--limit", type=int, default=10, help="top-k per search")
    ap.add_argument("--ef", type=int, help="hnsw_ef search param (default: collection's)")
    ap.add_argument("--timeout", type=float, default=60, help="per-query timeout, seconds")
    ap.add_argument("--vectors", help="JSON file with a list of query vectors to use instead of random ones")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", action="store_true", help="print one JSON line (for logs)")
    args = ap.parse_args(argv)

    base = f"{args.url.rstrip('/')}/collections/{args.collection}"
    try:
        info = call(base, api_key=args.api_key)["result"]
    except (urllib.error.URLError, OSError) as e:
        print(f"Could not read collection '{args.collection}' at {args.url}: {e}", file=sys.stderr)
        return 2
    dim = vector_size(info, args.using)

    rng = random.Random(args.seed)
    if args.vectors:
        with open(args.vectors) as fh:
            queries = json.load(fh)
        queries = [queries[i % len(queries)] for i in range(args.queries)]
    else:
        queries = [random_unit_vector(dim, rng) for _ in range(args.queries)]

    timings, errors = [], 0
    for vector in queries:
        body = {"query": vector, "limit": args.limit, "with_payload": False}
        if args.using:
            body["using"] = args.using
        if args.ef:
            body["params"] = {"hnsw_ef": args.ef}
        start = time.perf_counter()
        try:
            call(f"{base}/points/query", body, args.api_key, timeout=args.timeout)
            timings.append((time.perf_counter() - start) * 1000)
        except (urllib.error.URLError, OSError) as e:
            # A timeout here is the thing to catch: an app with a short search budget
            # silently falls back to worse results when this happens.
            errors += 1
            if not args.json:
                print(f"query failed after {(time.perf_counter() - start):.1f}s: {e}", file=sys.stderr)

    summary = summarize([round(t, 1) for t in timings], errors)
    summary.update({"collection": args.collection, "points": info.get("points_count"),
                    "status": info.get("status"), "ts": int(time.time())})
    if args.json:
        print(json.dumps(summary))
    else:
        print(f"{args.collection}: {summary['points']:,} points, status {summary['status']}, dim {dim}")
        print(f"  cold (first query): {summary['cold_ms']} ms")
        if "warm_p50_ms" in summary:
            print(f"  warm p50 / p95 / max: {summary['warm_p50_ms']} / {summary['warm_p95_ms']} / "
                  f"{summary['warm_max_ms']} ms over {len(timings) - 1} queries")
        if errors:
            print(f"  errors/timeouts: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
