#!/usr/bin/env python3
"""Create a small synthetic collection so you can try the other tools locally.

    docker run -p 6333:6333 qdrant/qdrant
    python3 tools/seed_demo.py --points 20000 --dim 256
    python3 tools/qdrant_audit.py --collection latency_demo --ram-gb 16 --cpus 4

The default config reproduces the "before" state from the case study (vectors
on disk, int8 quantization NOT pinned in RAM) so the audit has something to find.
Pass --fixed for the "after" state.
"""
import argparse
import json
import random
import sys
import urllib.request
from typing import List, Optional


def call(url: str, body: dict, method: str = "PUT") -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--collection", default="latency_demo")
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--fixed", action="store_true", help="create with the tuned config instead")
    args = ap.parse_args(argv)
    base = f"{args.url.rstrip('/')}/collections/{args.collection}"

    req = urllib.request.Request(base, method="DELETE")
    urllib.request.urlopen(req, timeout=60).read()
    call(base, {
        "vectors": {"size": args.dim, "distance": "Cosine", "on_disk": True},
        "hnsw_config": {"m": 16, "ef_construct": 100, "on_disk": not args.fixed},
        "quantization_config": {"scalar": {"type": "int8", "quantile": 0.99, "always_ram": args.fixed}},
        "optimizers_config": {"max_optimization_threads": 1 if args.fixed else None, "indexing_threshold": 1000},
    })
    rng = random.Random(1)
    batch = 1000
    for start in range(0, args.points, batch):
        points = [{"id": i, "vector": [rng.gauss(0, 1) for _ in range(args.dim)]}
                  for i in range(start, min(start + batch, args.points))]
        call(f"{base}/points?wait=true", {"points": points})
        print(f"\rupserted {start + len(points):,}/{args.points:,}", end="", flush=True)
    print(f"\ncreated '{args.collection}' ({'tuned' if args.fixed else 'before'} config)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
