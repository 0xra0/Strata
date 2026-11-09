# Strata DatKurai Coder IQ1_M – Linux patches

These patches were made against the exact `run-coder-uncen-iq1_m.sh` and `strata-coder-uncen-iq1_m.json` files from the uploaded `Strata.zip`.

## 0001 – Recommended: stop opening the browser

Strata's `serve/server.py` already supports `"open_browser": false` (checked against the user's uploaded code). This patch sets that JSON option **and** removes the `--open` launch flag. It does **not** disable the HTTP API or the server's optional homepage endpoint. Vision, MTP, the 8099 port, expert policy, and model paths are unchanged. A build is not required.

```bash
cd /mnt/ssd/Strata
# make backups first, especially if you edited these files
cp -a strata-coder-uncen-iq1_m.json strata-coder-uncen-iq1_m.json.pre-no-browser
cp -a run-coder-uncen-iq1_m.sh run-coder-uncen-iq1_m.sh.pre-no-browser

git apply --check /path/to/Strata_No_WebUI_Patches/0001-disable-strata-browser-autolaunch.patch
git apply /path/to/Strata_No_WebUI_Patches/0001-disable-strata-browser-autolaunch.patch
./run-coder-uncen-iq1_m.sh
```

To check patch presence: `git apply --reverse --check /path/to/.../0001-disable-strata-browser-autolaunch.patch`.

## 0002 – Optional and separate: leave more RAM for Blender and vision

Do **not** apply this by default if throughput is your main objective. It adds `STRATA_RESIDENT_HEADROOM_GIB=8` to the model config, instead of Strata's 4 GiB default under `--resident-experts`. Its C++ engine code reads this variable when starting the model. This reduces the RAM budget made available to resident expert weights by up to ~4 GiB relative to the original setting (actual amount depends on `MemAvailable` at startup). Expert cache misses and disk traffic may increase, so speed may decrease. This is not dynamic reclamation, and it is not an OOM cure for unusually large transient allocations.

```bash
cd /mnt/ssd/Strata
git apply --check /path/to/Strata_No_WebUI_Patches/0002-optional-extra-ram-headroom.patch
git apply /path/to/Strata_No_WebUI_Patches/0002-optional-extra-ram-headroom.patch
```

To reverse only patch 0002: `git apply -R /path/to/Strata_No_WebUI_Patches/0002-optional-extra-ram-headroom.patch`.

## Earlyoom and diagnosis

The pasted `smaps_rollup` shows ~20 GiB `Pss_Anon` and ~2 GiB `MemAvailable`. Earlyoom's `-m 8,4 -s 100,100` reacts when free *available* RAM drops near 8% (~2.5 GiB out of 31 GiB), irrespective of the large nominal capacity of RAM-backed zram. Strata expert residency, not the browser homepage, dominates RAM usage. The screenshot also shows a separate Blender API error (`Mesh.faces` doesn't exist); Blender mesh faces are accessed through `mesh.polygons` for the Python API or `bm.faces` for BMesh.

Do not disable earlyoom globally merely to keep Strata running; that can cause desktop lockups. `--open` launches a browser but is not the web server itself. The status line still prints `open http://127.0.0.1:8099/`, as an invitation, not evidence that a browser launched.
