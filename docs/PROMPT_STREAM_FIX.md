# Prompt file reads after the Zed run

This revision follows RESIDENT_SWAP_PREFETCH.md. Install it over the current
swap-prefetch revision; the full Strata-fixed.zip includes both revisions.

## What the attached log establishes

The log is appended across several engine runs. The current run starts at
2026-10-26 17:15:23, PID 2360107. Earlier records include the withdrawn shadow
implementation and the old 8.32 GiB swap sample; do not attribute those to this run.

- Both 33-token translation checks completed in the engine. The first generated
  eight tokens at 5.3 tok/s; the repeat reused 26 prompt tokens and generated eight
  at 20.5 tok/s. No HTTP report JSON was attached, so this does not verify the body
  of the previously blank curl response.
- The first 11,046-token Zed prompt took 132.6 seconds. During that prompt,
  VmSwap reached 6,866,144 KiB (6.55 GiB), despite the preceding translation checks
  having zero swap. The 11.85 GiB pageable complement leaves file-tier experts.
- Later requests stalled too: a 65,580-token context generated 190 tokens at
  4.9 tok/s, with 355.99 ms/window attributed to expert jobs. A request with only
  137 newly read tokens spent 16.35 seconds in prompt processing.
- This process completed 53 requests and generated 43,218 tokens, a weighted
  decode rate of 14.59 tok/s. Its final 85,975-token prompt took 307.7 seconds,
  then generated 4,276 tokens at 17.9 tok/s. The screenshot shows a completed
  Zed task and a context compaction; changed prompts after compaction can require
  fresh prefill. The log alone cannot prove a prefix-cache defect.

The relevant attached Zed OpenAI-compatible transport supports Chat Completions
and reasoning_content. No client failure is identified by this log/screenshot,
and this patch changes Strata's expert reads rather than the Zed repository.

## Code change

The swap-prefetch revision helps predicted expert pages that are already swapped.
It does not change prompt workers' file reads. Those workers currently assemble
missing GGUF experts with memcpy from shared mappings, which can fault a broad
model working set into the page cache while the expert complement is pageable.
The timing and swap samples are consistent with competition between those tiers;
they do not establish the time lost to each mechanism.

`STRATA_PROMPT_STREAM=1` adds an exact file-span reader for copy_blob/copy_blobs:

1. The existing RAM complement still supplies experts it holds.
2. Completely cached file spans use the existing mapped bytes (mincore classification).
3. Cold spans request preadv2 with RWF_DONTCACHE on separate read-only handles.
   Supporting kernels/filesystems can discard newly instantiated cache pages
   while retaining pre-existing cached pages.
4. On unsupported kernels/filesystems, cold spans use O_DIRECT through a reusable
   1 MiB + 8 KiB aligned buffer per existing prompt worker. Unaligned role offsets,
   split shards and the final partial file page are handled explicitly.
5. If both methods fail, an exact buffered pread is used and counted. Unsupported
   capabilities are remembered per file. Interrupted/short reads are handled;
   a failed attempt's partial data is overwritten by the complete fallback.

No expert values are discarded or changed. No arena, worker, mlock, GPU allocation,
quantization change or SSD weight backing is added. Startup copies, decode file
prefetch and demand reads retain their existing paths. Model args, context,
KV format, GPU reserve, RAM budget, pinning, speculative settings and reasoning
policy are preserved. Without the flag, the original prompt read branches run.
An existing STRATA_PROMPT_DIRECT=1 takes precedence, with an inactive-reader log.

The file reader reports request-local cached / RWF_DONTCACHE / O_DIRECT /
buffered-fallback MiB. Nonzero fallback bytes mean cache avoidance is incomplete.
Cache advice and mincore classification are best effort, not a physical residency
or OOM guarantee. Avoiding admission of cold prompt pages can increase repeated
disk reads compared with a machine whose cache can retain the entire working set.
The flag is opt-in and actual RX 6800 prompt/decode performance remains unmeasured.

The existing `file tier I/O this request` log also incorrectly included startup
I/O on the first request and idle/control I/O between later requests. Its baseline
is now sampled at GEN/GENI admission. These remain process-wide OS I/O counters,
not a measurement limited to GGUF experts. No inference computation is changed.

Linux API references:

- https://man7.org/linux/man-pages/man2/preadv.2.html
- https://github.com/torvalds/linux/blob/master/include/uapi/linux/fs.h
- https://man7.org/linux/man-pages/man2/open.2.html

## Install

Stop Strata. Extract Strata-prompt-stream-fix.zip into /mnt/ssd, merging its
Strata directory with /mnt/ssd/Strata. Then run:

```bash
cd /mnt/ssd/Strata
source .venv/bin/activate
bash tools/rebuild-prompt-stream-fix.sh
bash run-coder-uncen-iq1_m.sh
```

The script incrementally rebuilds only the engine in build-memory-fix-hip,
validates capability markers, backs up the existing engine, and enables the new
flag in strata-coder-uncen-iq1_m.json with a config backup. Explicit zero is kept.
The current vision binary is preserved. No prebuilt GPU engine is included.

At startup, verify `prompt file reader v1: active`. After a Zed request, inspect
the same marker's request-local MiB totals alongside runtime memory and timings.
The installed translation probe remains available:

```bash
python3 tools/check_translation_http.py > translation-check.json
```

It is a short HTTP/translation sanity check, not a long-context speed benchmark.
See PROMPT_STREAM_TEST_RESULTS.txt for host validation and its limits. To disable
only this reader, set STRATA_PROMPT_STREAM to "0" in the config's env and restart;
the previous swap-prefetch and RAM working-set policies remain as configured.
