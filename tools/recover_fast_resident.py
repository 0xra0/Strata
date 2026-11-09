#!/usr/bin/env python3
"""Disable the regressing SSD expert mode without rebuilding or changing inference args."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


def recover(config_path):
    path = Path(config_path).resolve()
    original = json.loads(path.read_text(encoding="utf-8"))
    if "--resident-experts" not in original.get("args", []):
        raise ValueError("Expected --resident-experts; this tool did not change the config")
    cfg = copy.deepcopy(original)
    if not isinstance(cfg.get("env", {}), dict):
        raise ValueError("The config env must be an object")
    # An empty override defeats an inherited export too; deleting this key would not.
    cfg.setdefault("env", {})["STRATA_RESIDENT_FILE_DIR"] = ""
    cfg["runtime_memory_log"] = True
    assert cfg["args"] == original["args"]
    assert cfg.get("vision") == original.get("vision")
    backup = path.with_name(path.name + ".before-resident-recovery-" + str(time.time_ns()))
    shutil.copy2(path, backup)
    fd, temporary = tempfile.mkstemp(prefix=".strata-recovery-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return cfg, backup


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "strata-coder-uncen-iq1_m.json")
    args = ap.parse_args()
    try:
        cfg, backup = recover(args.config)
    except (OSError, ValueError, KeyError) as error:
        ap.exit(1, str(error) + "\n")
    print("Disabled SSD-backed experts; the current engine can use its original RAM path without rebuilding.")
    print("Inference arguments and all vision settings preserved.")
    print("Config backup:", backup)
    print("Active engine log:", cfg.get("log", "(no log configured)"))
    print("Stop the current Strata server, then start your usual run-coder-uncen-iq1_m.sh launcher.")


if __name__ == "__main__":
    main()
