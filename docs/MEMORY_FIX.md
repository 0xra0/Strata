# Strata memory lifecycle changes

The subsequent long-context revision is documented in `LONG_CONTEXT_FIX.md`. It adds bounded replacements
inside the existing RAM allocation and a separate default thinking allowance. Its install script rebuilds
only the engine over the corrected ownership build below. These are opt-in changes; use its script to enable them.

**Ownership correction (26 October 2026):** the earlier revision incorrectly used
`ExpertCache::slot_of()` after adaptive swaps. That table retains the startup profile;
serve's `host_res` is the live owner table. The old path could discard CPU-only weights
and then restore a different expert from the old slot. This is a correctness bug and
can cause repetitive or incoherent generation. Use the corrected revision described in
`REPETITION_FIX.md`; restart the engine and begin a new chat after installing it.

This revision changes which allocations coexist. Active CPU expert weights remain in anonymous RAM.
It adds no SSD expert tier and does not lower model, context, KV, prefill, cache-count, speculation or image settings.

## Install this revision

Stop the running server. Extract `Strata-memory-lifecycle.zip` into `/mnt/ssd`; it contains `Strata/`.
Then run:

```bash
cd /mnt/ssd/Strata
source .venv/bin/activate
bash tools/rebuild-memory-fix.sh
bash run-coder-uncen-iq1_m.sh
```

**Both the engine and vision encoder must be rebuilt.** The script builds both before installing either,
checks both executables can start with `--help`, saves both previous binaries, and then enables the memory
profile in your current Coder Uncen config. It preserves the previous CPU encoder compiler from the existing
`build-vision/CMakeCache.txt` when that compiler is available. The HIP build keeps gfx1030, native CPU code
and MMQ prefill enabled, matching the uploaded project.

The existing Coder Uncen config is backed up before an atomic update. The profile helper verifies that both
installed binaries support the new options before enabling them. Source-archive configs leave the new modes
off until this succeeds, so the old binaries cannot accidentally receive an unsupported vision option.

The modified code is source, not an AMD executable built in this workspace. The existing `engine/` binaries
in the full project archive predate these changes; use the rebuild script rather than running them directly.
The current project paths remain `/mnt/ssd/Strata` and `/mnt/ssd/Strata-data`.

## 1. Release duplicate host expert pages only while idle

The resident expert complement can also contain host copies of GPU-cache experts. Prefill needs those copies
when it borrows the cache's VRAM as scratch, but retaining them while idle makes CPU vision coexist with
unneeded duplicate allocations.

With `STRATA_MEMORY_LIFECYCLE=1`, on the single-GPU resident serve path:

1. Finish pending default adaptive exchanges and require every prefill loan to have been returned.
2. Find only experts present in both the pageable host complement and the primary GPU cache, using
   serve's **live `host_res` table**, after pending exchanges have landed. Validate its shape, slot bounds
   and unique ownership before discarding any host page. The startup `ExpertCache::slot_of()` table must
   never be used here: adaptive swaps update `host_res`, not that table.
3. At startup, or after control commands that can relayout scratch, refresh those VRAM copies from the still-valid
   host bytes. Cache ownership metadata alone is insufficient: prefill initialization may have written scratch
   into tail slots. This step establishes a valid authoritative copy before any host page is discarded.
4. Discard only whole pages wholly covered by GPU-duplicate blobs. Pages shared with CPU-only blobs or malloc
   boundaries are retained. The allocation and offsets stay intact; no backing file or model read is involved.
5. Before processing the next command, check ownership against the same live table and copy those exact
   bytes back from VRAM. Failure prevents inference from
   continuing with a partial restore. Missing host bytes are never exposed as valid weights even during idle.

HIP transfers use the cache's existing single-expert pinned bounce buffer in both directions, rather than
registering a multi-GiB pageable allocation with the driver. Decode itself uses the existing resident weights,
cache size, kernels and sampling. The extra copies happen at the request boundary, not inside token decoding.
They can add request-boundary latency; that cost is reported rather than assumed to be zero.

Batch/pipelined execution, layer splits, async adaptation, pinned/rotated complements and Foresight are excluded
from this reclaim path. They retain the previous behavior. A run with no duplicate host pages reports 0 MiB;
this optimization cannot reclaim memory that is already unique. The freed amount must be read from your log.

## 2. Release CPU vision compute workspace before engine prefill

The CPU vision context's GGML scheduler retained its compute buffers after image encoding. They then coexisted
with engine prefill, resident experts and the application's allocations.

The new `vision.release_workspace: true` option passes `--release-workspace`. After writing an image's exact
embeddings, the encoder frees preprocessing objects, synchronizes and releases only its CPU graph scheduler's
compute buffers, and trims free libc pages on glibc Linux. Encoder/projector weights remain loaded at the same
addresses. Attention selection, model parameters, image tokens and threads are retained. The next uncached
image reserves workspace again; cached images still reuse the existing embedding files.

The encoder acknowledges completion only after cleanup, so engine restoration/prefill cannot overlap with the
old CPU workspace. The measured encode duration includes cleanup. This API extension is in the bundled mtmd
sources; no external llama.cpp checkout is fetched by the rebuild script.

## 3. Reuse the checkpoint being evicted before capturing its replacement

The ordinary full LRU cache previously captured an additional complete GDN checkpoint and then evicted an old
one. The memory profile selects the same old victim first and reuses its state-vector allocations for capture.
The resulting cache entries, their order, stamps and configured count are unchanged. Pins, tail/spread policy,
pooled checkpoints and layer splits keep the existing path. A failed capture does not publish a partial state;
the discarded entry was a cache entry, and live model state is unaffected.

## Validation actually performed here

The following are memory/correctness checks, not an AMD model-speed claim:

- A real anonymous-memory test used a 192 MiB host arena with a 96 MiB simulated-VRAM duplicate region. Idle
  reclaim reduced process RSS by 95.9 MiB. Every CPU-only byte stayed intact, and the complete arena matched
  byte-for-byte after restore. Misalignment, neighboring CPU pages, bad ranges and partial-copy failure/retry
  are tested. The source of restored bytes in that test is simulated VRAM, not an actual PCIe device.
- The added ownership regression test reproduces the earlier wrong-expert restore after an adaptive swap.
  It then exercises the actual corrected planner with 769 swaps and 256 reclaim/restore rounds, verifying
  every RAM expert against its original bytes and protecting every CPU-only expert during idle. GPU slots
  are represented by host buffers in this test. See `REPETITION_FIX.md` for its results and limits.
- The actual modified CLIP workspace function was compiled and exercised with real GGML CPU graphs. Across
  three release/rebuild rounds it freed 134,217,728 bytes each time, retained the persistent weight buffer at
  the same address, and reproduced output hash `842a591a98222325`. RSS fell from about 133 MiB to 5-9 MiB after
  each release. This tests scheduler/buffer lifecycle with synthetic graph data, not a full Qwen projector.
- The modified CPU `strata-vision` executable and mtmd library were fully built. The encoder's help command
  runs, and the public mtmd header passes C compilation.
- All 3,200 tested ordinary checkpoint-eviction decisions match the prior append-then-evict policy. Special
  policies decline early reuse.
- See `MEMORY_FIX_TEST_RESULTS.txt` for server, CTest, installer and syntax results. The model/GPU-dependent
  parts have host syntax checks with real CUDA headers, not a full HIP or CUDA engine build.

No AMD GPU, ROCm compiler or target model weights are available here. Therefore target peak RAM, image
accuracy and 25 tok/s decode have not been measured for this revision. These changes remove specific wasted
allocations; they do not establish that every workload fits in 32 GiB or cure arbitrary GPU memory spill.

## What to inspect on the target PC

The active log is `/mnt/ssd/Strata/strata-coder-uncen-iq1_m.log`. The new log lines report actual amounts:

- `idle GPU shadows: released ... MiB` while waiting for a command;
- `idle GPU shadows: live ownership v2` confirms the ownership correction is active;
- `idle GPU shadows: restored ... MiB from VRAM in ... ms` before processing the next command;
- `strata-vision: released ... MiB of CPU workspace; model weights retained` after a new image;
- the existing runtime-memory JSON counters for RAM, swap, disk reads and AMD VRAM/GTT usage.

Use the same image, prompt and inference settings for comparison. The relevant result is peak RAM during the
same failing image request and sustained decode rate afterward, together with the actual released MiB and
restore time. The older attached `strata-coder-iq1_m.log` is a different, earlier Coder run.

The 26 October Uncen log's earlier reported 894/3676/4844 MiB releases are not valid savings measurements:
the old planner could include CPU-only experts in those figures. Correct reclaim may be much smaller or
zero, especially when the RAM complement and GPU cache contain disjoint experts. CPU-only weights must
remain resident. The ownership correction does not establish that the original memory-pressure problem
has been solved for this PC.

Rollback uses the binary/config backups printed by the installer. The earlier experimental SSD expert patch
is not used. Do not apply it to this source. The new memory modes are opt-in and can be disabled by clearing
`STRATA_MEMORY_LIFECYCLE` and removing `vision.release_workspace` from a backed-up config.
