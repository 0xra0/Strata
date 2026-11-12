# Read-ahead for swapped resident experts

Subsequent revision: [PROMPT_STREAM_FIX.md](PROMPT_STREAM_FIX.md) addresses prompt
file-cache pressure seen in the later Zed run. It retains this swap-prefetch policy.

The controlled Ukrainian-to-English request returned the correct translation, `Hi! How are you doing?`.
It generated only eight tokens: its 3.8 tok/s API timing (4.5 tok/s in the server window) is a short burst,
not a sustained decode benchmark. The important new memory sample is `VmSwap_kib: 8724204`: 8.32 GiB of the
engine was swapped out, with 7.65 GiB reported available. `VmSwap` alone does not identify the swap device,
prove all of those bytes are experts, or establish the exact time lost to swap.

## Specific gap corrected

`FileExpertSource::warm()` previously skipped every expert present in the logical host complement. That
avoids reading duplicate model-file pages, but a pageable host allocation can itself be swapped out. The
routing lookahead therefore made no swap read-ahead request for those predicted CPU experts.

With `STRATA_RESIDENT_SWAP_PREFETCH=1`, the existing routing lookahead worker also considers the pageable
resident complement. The existing predictor already excludes experts currently assigned to the GPU.

- Sample the engine's own `VmSwap` once per second. A known zero skips the residency-probe syscall entirely.
  Unknown swap status retains the guarded page-probe path.
- Use `mincore` to find absent pages in a predicted expert. Already resident pages need no advice.
- Request `MADV_WILLNEED` only for cold runs. No expert values are read, overwritten, discarded or requantized.
- Limit requested read-ahead to 128 MiB per 100 ms, at most eight calls per expert and an 8 MiB expert size cap.
  These bound requested ranges, not the kernel's exact I/O or eventual physical-memory usage.
- Check available RAM/commit once per active 100 ms period and reserve admitted hint bytes against a 2,560 MiB
  free-memory floor. Unknown available memory declines hints. Failed hints also consume their period allowance
  so errors cannot cause an unbounded retry stream.

There is no additional worker, full-model scan, forced page touch, mlock, GPU registration or new SSD-backed
weight arena. Inference keeps the existing fallback if hints fail or arrive too late. Pinned/registered/locked
complements retain their previous behavior. An explicit `STRATA_LOOKAHEAD=0` disables the predictor and is not
overridden. This revision targets the current Linux GGUF path with routing lookahead enabled.

Linux's advice can schedule anonymous swap reads into swap cache; it does not promise that the pages will
remain in RAM. Consequently a successful hint is not counted as restored bytes, and `VmSwap` need not
immediately become zero. The log reports advised MiB, errors and skip counters, not fabricated physical savings.

API references:
- https://man7.org/linux/man-pages/man2/madvise.2.html
- https://github.com/torvalds/linux/blob/master/mm/madvise.c (`madvise_willneed` and `swapin_walk_pmd_entry`)

## Install over the current long-context revision

Stop Strata, extract `Strata-swap-prefetch-fix.zip` into `/mnt/ssd`, overwriting files in its `Strata/` folder, then:

```bash
cd /mnt/ssd/Strata
source .venv/bin/activate
bash tools/rebuild-hip.sh
python3 tools/configure_swap_prefetch.py
python serve/server.py --engine strata --config strata-swift-abl-iq2_xs.json --port 8099
```

The script reuses the existing HIP build directory and builds only the engine. It verifies capability markers,
backs up the installed engine and current config, then enables the prefetch flag and memory telemetry. Model,
context, KV, GPU cache, RAM headroom, pinning, thinking, sampling and vision settings are preserved. Existing
explicit opt-outs are retained. The unchanged source-archive configs do not enable the new mode before a build.

After a request, the engine log contains `resident swap prefetch v1`. The counters are cumulative. A healthy
zero-swap run should show zero-swap skips and no read-ahead; an affected run can report cold-page advice. Error,
pressure and budget skips retain normal demand paging and must not be read as successful prefetches.

## Check the empty HTTP response separately

The blank second curl paste does not identify an HTTP status, response size, connection error or engine exit.
This revision includes a standard-library-only probe that makes two identical short requests and records
those details:

```bash
python3 tools/check_translation_http.py > translation-check.json
```

Run it while the server is ready. It uses the active model reported by `/v1/models`, the same Ukrainian test,
thinking disabled, temperature zero, presence penalty zero and a 128-token output cap. Those values apply only
to the two test requests; the config is not edited. `STRATA_API_KEY` may supply authentication and is never
written to the report. The second call can expose a difference between the first and repeated request, but
eight output tokens do not measure sustained throughput. The probe does not automatically diagnose a cause.

## Validation and limits

See `RESIDENT_SWAP_PREFETCH_TEST_RESULTS.txt`. The new production helper is tested with injected page-residency
states and actual Linux `mincore`/`MADV_WILLNEED` calls on read-only file data. Exact expert bytes are retained.
The host has no swap configured, so actual anonymous swap-in latency, the target model and RX 6800 throughput
were not tested. HIP/CUDA engine builds are also unavailable here; host syntax checks use CUDA headers.

This corrects the missing resident-page read-ahead path. It does not establish zero swap, prevent arbitrary
system memory pressure, restore 20 tok/s after every idle period, or explain the empty HTTP response by itself.
Compare the same workload, idle interval and memory conditions, including the new counters and major faults.
The source changes are in commit 7306db6a; see `git show 7306db6a`.
