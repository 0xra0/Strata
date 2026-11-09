# Repetition caused by stale expert ownership

## Fault found

The 26 October Coder Uncen log shows 462 adaptive exchanges before the first idle release, 5,176 before
the second, and a 3,675.8 MiB restore before the request that repeats `proc/PID?` in the screenshot.
That run generated 4,245 tokens at 22.5 tok/s before cancellation. It does not establish a bad GGUF.

The memory-lifecycle code introduced a concrete wrong-weight restore:

1. Expert A initially occupies slot 0; Expert B is in RAM.
2. Adaptive swapping puts B in slot 0 and copies A into B's former RAM location. Serve updates `host_res`.
3. `ExpertCache::slot_of(A)` still returns slot 0: it is the startup profile, not the live adaptive table.
4. The old idle planner wrongly discards A's CPU-only RAM weights. On the next request it copies B from
   slot 0 into A's RAM location. Subsequent requests then compute with the wrong expert weights.

The previous tests covered page boundaries and copies with fixed ownership; they missed adaptive swaps.
The attached log does not contain token-level traces proving when the first divergent logit appeared,
but the wrong-weight bug is directly reproducible in the old algorithm.

## Correction

Release and restore now require the live `host_res` table as an explicit argument. The planner checks table
shape, live slot bounds and unique ownership. Only genuine GPU duplicates are eligible. Restore checks
the saved expert/slot identity against live ownership before copying. Existing synchronization, startup
scratch refresh and fail-closed handling remain. The startup cache lookup is no longer consulted by
either operation. No change is made to sampling, model quantization, context, speculative decoding or kernels.

The profile configurator rejects the earlier memory engine even though it supports `STRATA_MEMORY_LIFECYCLE`.
It now requires the corrected `idle GPU shadows: live ownership v2` marker.

The earlier reported multi-GiB release amounts may include CPU-only weights and must not be treated as safe
memory savings. Correct release may report 0 MiB. The original memory-pressure problem still needs target
hardware validation; this revision corrects the introduced data corruption.

## Install over the memory-lifecycle revision

Stop the server. Extract `Strata-repetition-fix.zip` into `/mnt/ssd`, overwriting its `Strata/` source files.
Then run:

```bash
cd /mnt/ssd/Strata
source .venv/bin/activate
bash tools/rebuild-ownership-fix.sh
bash run-coder-uncen-iq1_m.sh
```

The script uses the existing `build-memory-fix-hip` directory and rebuilds only the engine. It retains
gfx1030, native CPU and MMQ settings, checks the new executable, backs up the installed engine and installs
the replacement. The vision encoder and current JSON configuration are preserved. If that build directory
is missing, it directs you to the initial `rebuild-memory-fix.sh` script.

After restart, check the log for `idle GPU shadows: live ownership v2`. Use **New chat**: the old process's
weights/checkpoints and the earlier repeated conversation must not be reused. The fix does not modify
GGUF files, so this failure does not require downloading the model again.

## Validation

- The regression fixture physically reproduces the old corruption: expert 0's RAM is overwritten with
  expert 3's GPU-slot bytes after the first swap.
- The corrected production planner completes 769 adaptive swaps and 256 real anonymous-page reclaim/restore
  rounds with all RAM experts matching their original bytes, including CPU-only experts during idle.
- Invalid table shape, duplicate live slots, out-of-range slots and ownership changes are rejected.
- AddressSanitizer/UndefinedBehaviorSanitizer checks pass with leak scanning disabled because the host's
  tracing environment prevents LeakSanitizer from scanning `/proc` threads.
- CTest and the installer/configuration tests are recorded in `MEMORY_FIX_TEST_RESULTS.txt`.
- The modified `generate.cpp` and `expert_source.cpp` pass C++20 host syntax checks with real CUDA headers.

No AMD GPU, ROCm compiler or target model is available in this workspace. The target HIP engine and
full-model output have not been run here. The regression proves wrong-weight corruption and its ownership
correction using host buffers to represent GPU slots; it is not an AMD inference or throughput benchmark.
