#!/usr/bin/env python3
"""Enable memory lifecycle reuse after both matching binaries have been built."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


def configure(path, validate_binaries=True):
    path = Path(path).resolve()
    original = json.loads(path.read_text())
    if "--resident-experts" not in original.get("args", []):
        raise ValueError("Memory lifecycle reuse requires the existing --resident-experts mode")
    vision = original.get("vision")
    if not isinstance(vision, dict) or vision.get("gpu"):
        raise ValueError("This profile is for the existing CPU vision encoder")
    if validate_binaries:
        for binary, marker in ((original["exe"], b"idle GPU shadows: live ownership v2"),
                               (vision["exe"], b"--release-workspace")):
            p = Path(binary)
            if not p.is_absolute():
                p = Path(original.get("cwd") or path.parent) / p
            if marker not in p.read_bytes():
                raise ValueError(f"{p} must be rebuilt before enabling this profile")
    cfg = copy.deepcopy(original)
    if cfg.get("env") is None:
        cfg["env"] = {}
    if not isinstance(cfg["env"], dict):
        raise ValueError("The config env must be an object")
    cfg["env"]["STRATA_RESIDENT_FILE_DIR"] = ""
    cfg["env"]["STRATA_MEMORY_LIFECYCLE"] = "1"
    cfg["vision"]["release_workspace"] = True
    cfg["runtime_memory_log"] = True
    assert cfg["args"] == original["args"]
    assert all(cfg["vision"][k] == v for k, v in vision.items() if k != "release_workspace")
    backup = path.with_name(path.name + ".before-memory-lifecycle-" + str(time.time_ns()))
    shutil.copy2(path, backup)
    fd, temporary = tempfile.mkstemp(prefix=".strata-memory-profile-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return cfg, backup


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "strata-coder-uncen-iq1_m.json")
    args = ap.parse_args()
    try:
        cfg, backup = configure(args.config)
    except (OSError, ValueError, KeyError) as error:
        ap.exit(1, str(error) + "\n")
    print("Memory lifecycle reuse enabled; inference arguments and image settings preserved.")
    print("Config backup:", backup)
    print("Run log:", cfg.get("log", "(none configured)"))
