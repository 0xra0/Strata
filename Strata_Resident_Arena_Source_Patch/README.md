# Strata Linux: SSD-backed resident expert arena (experimental source patch)

## What this patch changes

The existing `--resident-experts` path duplicates the GPU-cache complement (up to ~20 GiB on this PC) in a pageable **anonymous** allocation. Under pressure these pages go to zram swap. In the user's trace, `Pss_Anon` rose to ~20 GiB, free RAM collapsed, zram swap rose sharply and the engine was killed.

The new **opt-in** `STRATA_RESIDENT_FILE_DIR` path replaces **only the pageable expert complement's malloc** with an anonymous-to-users but **disk-backed** MAP_SHARED mapping. It creates a hidden temporary file in the chosen disk directory, preallocates enough disk space, immediately unlinks it, fills it with **identical expert bytes** and syncs them **once at startup**. The Linux kernel can reclaim clean mapped pages during Blender prefill, faulting them back from SSD if needed, rather than forcing an extra 20 GiB of anonymous RAM into zram. The file vanishes on exit, including SIGKILL. The normal path remains unchanged when the env variable is unset.

**No model weights, quantization, context length, GPU expert-cache size, MTP, vision, CPU kernel arithmetic or prefill chunk settings are changed.** However, this **does not guarantee unchanged speed**: if Linux has to evict actively used pages, SSD reads and page faults can slow requests. Writing and syncing the arena once on startup also costs SSD space, writes and time. For this reason it's experimental and opt-in, not a proven fix for every OOM.

## Install and build

Stop Strata. In `/mnt/ssd/Strata`, back up your current config and executable, apply the patch and build your normal HIP engine:

```bash
cd /mnt/ssd/Strata
git apply --check /path/to/0001-linux-ssd-backed-resident-experts.patch
git apply /path/to/0001-linux-ssd-backed-resident-experts.patch

# Unit test of the actual Linux arena helper (CPU-only):
c++ -std=c++17 -O2 -Iinclude tools/tests/test_file_backed_arena.cpp -o /tmp/test-resident-file-arena
/tmp/test-resident-file-arena /mnt/ssd/Strata-data

# Build engine against your installed ROCm/HIP toolchain:
cmake --build build-hip --target strata -j4

# Find the newly built executable before replacing anything:
find build-hip -type f -name strata -executable
```

Back up `/mnt/ssd/Strata/engine/strata`, and copy the **actual newly built** executable shown by `find` to `engine/strata`. Do not copy a binary if the build failed or did not produce a new executable. If your build target differs, use your existing HIP build commands. Do not run the old binary with the new environment variable; it will silently ignore it.

Check disk capacity first:

```bash
df -h /mnt/ssd/Strata-data
```

The SSD needs free space for the **entire RAM expert complement** (around 15–22 GiB in this setup, depending on the current expert layout), in addition to your model files. Don't use `/tmp`, `/dev/shm`, zram, or tmpfs as the backing directory.

Then, from this package:

```bash
python3 create_filebacked_config.py
cp run-coder-uncen-filebacked.sh /mnt/ssd/Strata/
chmod +x /mnt/ssd/Strata/run-coder-uncen-filebacked.sh
cd /mnt/ssd/Strata
./run-coder-uncen-filebacked.sh
```

The generated JSON is separate from the original. It **requires** `--resident-experts` in the original and injects `STRATA_RESIDENT_PIN=0` and `STRATA_RESIDENT_FILE_DIR=/mnt/ssd/Strata-data`. It preserves all existing inference args and all `vision` settings. It doesn't open the browser automatically. The server stays on port 8099.

Look for:

```
FileExpertSource: ... GiB pageable complement backed by SSD
FileExpertSource: SSD-backed complement synced (... s)
```

Now `Pss_File` should account for much more of the expert complement, while `Pss_Anon` should be lower. File cache may still be shown in RSS, but its **clean pages are reclaimable**. Measure `MemAvailable`, `Swap`, `Pss_Anon`, `Pss_File`, major faults, and tokens/sec through the same 8K Loopcut request. Compare with the original known 25–30 tok/s and note startup time. The model should produce equivalent values, subject to normal nondeterministic sampling.

## Rollback

Stop the patched server. Start `./run-coder-uncen-iq1_m.sh` using the original configuration. To completely revert the binary change, restore the backed-up engine binary; revert the source patch with:

```bash
cd /mnt/ssd/Strata
git apply --reverse /path/to/0001-linux-ssd-backed-resident-experts.patch
```

Restoring the original binary and config is sufficient to stop using this experimental mode.

## Validation and limitations

- `git apply --check` + reverse-check passed against a clean fixture made from the supplied Strata git tree.
- Linux standalone test passed for file-backed read/write, `msync`, `MADV_DONTNEED` + reread, and tmpfs rejection.
- **Not** compiled in the HIP engine here (ROCm/HIP toolchain is not installed).
- **Not** tested on your AMD hardware or benchmarked. Stability and throughput are not yet proven. The attempt to keep fast expert pointers while making pages reclaimable is well-motivated, but a full Coder + Blender run is needed before recommending it upstream.

If the machine still freezes, restore the unmodified config and binary and share logs; don't lower earlyoom thresholds further. Keep `-m 8,4` until the engine itself is proven stable.
