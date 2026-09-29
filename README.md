# Qdrant Latency Playbook ⚡

**How a production vector search went from 12–21 seconds to 0.07 seconds on the same
15 GB server, what broke afterwards, and three small read-only tools that find the
same problems on your own Qdrant.**

> This is a write-up of a real incident on a RAG chatbot I build and run. Server
> names, client details and code are left out. The numbers are the ones measured
> at the time. — [Naeem Lashari](https://www.linkedin.com/in/naeem-lashari)

- [The short version](#the-short-version)
- [The case study](#the-case-study)
- [The checklist](#the-checklist)
- [The tools](#the-tools)

---

## The short version

| | Before | After |
|---|---|---|
| Warm vector search | **12–21 s** | **0.07 s** |
| Cold search (first after restart) | 60 s | 6.3 s |
| Collection | 8.19M points | 7.44M points |
| Hardware | 4 vCPU, 15 GB RAM | **same box, $0 spent** |

What fixed it: the collection already had int8 quantization, but the quantized vectors
were **stored on disk** (`always_ram: false`). One config flip, plus deleting ~758K dead
vectors so the result fit in RAM, turned every search from disk reads into memory reads.

What I learned afterwards mattered as much: a latency fix on a box with no headroom
does not stay fixed. [Part 2](#part-2-it-didnt-stay-fixed) covers the regressions,
including one outage I caused.

---

## The case study

### The setup

- A RAG chatbot answering from a Qdrant collection of **~8.2M × 1024-dim** embeddings
  (BGE-M3), with a reranker and an LLM after retrieval.
- One **4 vCPU / 15 GB RAM** GPU instance runs Qdrant (Docker), the embedding and
  reranker model service, and the app workers.
- The app gives vector search a time budget and **falls back to keyword search** if it
  doesn't answer in time.

### The symptom

Users said the bot was giving **generic, wrong answers**. There were no errors in the
logs and Qdrant reported `status: green`.

Tracing one question showed the real chain:

```
vector search ──(5 s budget)──► timeout, 0 hits
      └─► keyword fallback carries the answer ─► LLM writes a generic essay
```

A slow search didn't look slow to users. It looked wrong. **The silent fallback turned
a performance problem into a correctness problem.**

### The diagnosis (measured, not guessed)

| Check | Finding |
|---|---|
| Raw search via REST | 12–21 s warm, ~60 s cold. Even repeat queries were slow. |
| Qdrant process memory | **14.6 GB swapped out** |
| Disk | NVMe at **97.5% utilisation**, ~3,000 reads/s |
| Data size | float32 vectors ≈ 34 GB, int8 copy ≈ 8.4 GB, HNSW graph on disk |
| Config | int8 quantization existed, but with **`always_ram: false`** |

The working set was about **3× RAM**. Every graph hop and every candidate score was a
random disk read on a saturated disk.

> `green` only means *the index is built*. It says nothing about whether the box can
> *serve* it.

### The fix

1. **Delete what shouldn't be there.** ~758K vectors were orphans from deleted source
   files or low-value crawl pages: **8.19M → 7.44M (−9%)**. I kept a manifest of every
   deletion so any of it could be re-crawled.
2. **Flip the existing quantization into RAM.** It was a config change, not a re-quantize:

   ```bash
   curl -X PATCH http://localhost:6333/collections/documents \
     -H 'Content-Type: application/json' \
     -d '{"quantization_config": {"scalar": {"type": "int8", "always_ram": true}}}'
   ```

3. **Let one optimizer pass rewrite the segments** during a low-traffic window (~3.5 h
   on the disk-bound box). Config changes only take effect as segments are rewritten.

**Result:** warm search **12–21 s → 0.07 s**, cold **60 s → 6.3 s**. I then ran real
user questions through the full pipeline to check quality. Answers were unchanged,
because the reranker and keyword fusion cover the small recall loss from int8.

Search then took 0.07 s of a multi-second answer, which made the LLM and reranker the
new bottleneck. That's the right place for it to be.

### Part 2: it didn't stay fixed

The fix left almost no free RAM. Over the next weeks every one of these happened:

**1. Latency crept back under load.** Qdrant slowly grew back toward the RAM ceiling.
Restarting helped for an hour or two. *Lesson: if your fix needs 95% of RAM, it's
a reprieve, not a fix.*

**2. The "obvious" optimizer speed-up made it worse.** Moving `max_optimization_threads`
from 1 to 2 on the 4 vCPU box took warm queries from **0.6 ms to 9 s within 60 seconds**.
"1 thread" already meant ~190% CPU once REST and search workers were counted.
*Lesson: on a small box, give rebuilds time, not threads.*

**3. Docker flags didn't do what they said.** `--memory` and `--memory-swap` did not
reach cgroup v2 on that host (`memory.max` still read `max`). *Lesson: read
`/sys/fs/cgroup/.../memory.max` and `memory.swap.max` after every container
recreate. Don't trust the flags.*

**4. I caused an outage with a swap ban.** To stop Qdrant swapping, I set the
container's `memory.swap.max = 0`. Qdrant pinned ~14.6 GB and the co-located
embedding service was **OOM-killed**. Its replacement couldn't come back: there was no
RAM left to load the model, and an unattended NVIDIA driver update (installed weeks
earlier, never rebooted) had left the GPU driver mismatched. **The whole chat went
down.** The fix was to lift the ban, then reload the NVIDIA kernel modules, with no reboot. *Lesson: on a shared
box, pinning one process in RAM just moves the pain to its neighbour.*

**5. Weekly backups evicted the index.** Uploading a ~48 GB snapshot with `aws s3 cp`
read the file through the page cache and pushed Qdrant's memory out. Post-backup cold
searches took **4–30 s**, and `sar` showed 732 pages/s swapped out at 03:40. The fix was
one line: read the file without caching it.

```bash
nice -n 19 ionice -c2 -n7 nocache aws s3 cp /snapshots/latest.snapshot s3://bucket/
```

A latency probe now runs right after each backup and logs the result.

**6. A "warm-up" job would have made it worse.** Before adding a cron job to pre-warm
the cache, I measured it: 10 broad searches took **127 s**, pushed swap up by 0.5 GB,
and left latency *worse*. Real traffic re-warms the part of the index people actually
use. *Lesson: measure the fix before you schedule it.*

**7. The timeout budget was a product decision.** Raising the primary vector-search
timeout from 5 s to 15 s meant a slow cold query still returned the right answer,
instead of a fast wrong one. The 5 s budget was the part users could see.

### Where it ended up

- Graph back on disk (`hnsw.on_disk: true`, mmapped and evictable as page cache), int8
  vectors pinned in RAM, original vectors on disk, one optimizer thread.
- Backups that don't touch the page cache, with a latency probe after each run.
- A longer search budget, so slow-but-correct beats fast-but-wrong.
- An honest note in the runbook: **the durable fix is RAM headroom** (a bigger box, or
  moving the embedding model off it). Config tuning bought months of sub-second search
  at $0, but it didn't remove the ceiling.

---

## The checklist

Run through this when a Qdrant collection is slow. The tools below automate most of it.

- [ ] **Is Qdrant's memory in swap?** Check `VmSwap` in `/proc/<pid>/status`. This is
      the #1 cause of "fast sometimes, 10+ seconds other times". → `host_check.py`
- [ ] **Where does each part of the index live?** Vectors, quantized vectors and the HNSW
      graph each have their own `on_disk` / `always_ram` setting. → `qdrant_audit.py`
- [ ] **Is quantization pinned in RAM?** `always_ram: false` means the fast copy is read
      from disk. → `qdrant_audit.py`
- [ ] **Does the hot set fit, with room for everything else on the box?** → `qdrant_audit.py --ram-gb`
- [ ] **Is the collection actually indexed?** `indexed_vectors_count` far below
      `points_count` means brute-force search. Also, `max_optimization_threads: 0`
      *disables* the optimizer; `null` is auto. → `qdrant_audit.py`
- [ ] **Are there dead vectors?** Orphans from deleted sources cost RAM on every search.
- [ ] **What does the cgroup really allow?** Read `memory.max` and `memory.swap.max`. → `host_check.py`
- [ ] **What else reads big files on this box?** Backups, log shipping, re-indexing.
      Use `nocache`, or schedule them away from traffic.
- [ ] **What does your app do when search is slow?** If it silently falls back, slow
      search shows up as *wrong answers*. Log it and alert on it.
- [ ] **Measure cold and warm separately**, before and after every change. → `latency_probe.py`

---

## The tools

Three single-file Python scripts. **Standard library only, and read-only**: they never
change your collection. Copy one to a server and run it with `python3`.

| Tool | What it answers |
|---|---|
| `tools/qdrant_audit.py` | Which config choices are costing latency? Estimates the RAM the hot parts of the index need and flags risky combinations. |
| `tools/latency_probe.py` | How slow is it really? Cold first query vs warm p50/p95/max, errors and timeouts. `--json` for cron logs. |
| `tools/host_check.py` | Is the Linux host hurting Qdrant? Swap used by the Qdrant process, cgroup limits, swappiness, transparent huge pages. |
| `tools/seed_demo.py` | Builds a small demo collection in the "before" or tuned state so you can try the other three. |

### Try it locally

```bash
docker run -p 6333:6333 qdrant/qdrant
python3 tools/seed_demo.py                 # "before" config: int8 not pinned in RAM
python3 tools/qdrant_audit.py --collection latency_demo --ram-gb 16 --cpus 4
python3 tools/latency_probe.py --collection latency_demo -n 30
python3 tools/seed_demo.py --fixed         # tuned config -> the audit comes back clean
```

Sample audit output (the "before" demo collection, run against Qdrant 1.19):

```
Collection: latency_demo   status: green   points: 20,000   indexed: 20,000   segments: 4

Vector (default): dim=256 Cosine
  vectors on_disk     : True
  hnsw on_disk        : True  (m=16)
  quantization        : scalar  always_ram=False
  est. size (GB)      : f32 vectors 0.02, int8 0.0, graph 0.0
  est. hot set (GB)   : 0.01  of 16 GB RAM  (lower bound)

[HIGH] [(default)] quantized vectors are NOT pinned in RAM (always_ram=false)
    The quantized copy exists but lives on disk, so searches still page it in. ...

[MEDIUM] max_optimization_threads=null (auto: saturate CPUs) on a 4-CPU box
    On a 4 vCPU box, 2 optimizer threads took warm queries from under 1 ms to seconds ...
```

On your server:

```bash
python3 qdrant_audit.py  --url http://localhost:6333 --collection docs --ram-gb 15 --cpus 4
python3 latency_probe.py --url http://localhost:6333 --collection docs -n 20 --json >> /var/log/qdrant_probe.log
python3 host_check.py
```

`qdrant_audit.py` and `host_check.py` exit with status 1 when they find a high-severity
issue, and `latency_probe.py` does the same when a query fails or times out, so all
three can gate a cron job or CI check.

A note on `latency_probe.py`: random query vectors are a worst case, because they land
in random parts of the graph. For numbers comparable to production, pass real query
embeddings with `--vectors queries.json`.

### Tests

```bash
pip install pytest
pytest -q                                          # unit tests, no Qdrant needed
QDRANT_URL=http://localhost:6333 pytest -q         # also runs the tools against a live Qdrant
```

The audit tests use collection-info JSON captured from a real Qdrant 1.19. `host_check`
is tested against a fake `/proc` + `/sys` tree, so it runs on any OS.

---

**Related:** [rag-demo](https://github.com/lashari/rag-demo) (hybrid dense + BM25 retrieval
with RRF) · [ai-support-widget](https://github.com/lashari/ai-support-widget) (streaming
support chat with human handoff)

## License

MIT
