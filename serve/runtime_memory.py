"""Read Linux paging/VRAM counters. No process inspection tools or extra packages required."""
from pathlib import Path
import time


def _counters(path):
    result = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields = value.split()
            if fields:
                try:
                    result[key] = int(fields[0])
                except ValueError:
                    pass
    return result


def sample_runtime_memory(pid, proc_root=Path("/proc"), drm_root=Path("/sys/class/drm")):
    """Counters are cumulative; the caller computes deltas only within the same PID."""
    proc = Path(proc_root) / str(pid)
    try:
        # comm may contain spaces and parentheses. Fields after its final ')' start at field 3.
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        sample = {"pid": pid, "time": time.time(), "major_faults": int(stat[9])}
        status = _counters(proc / "status")
    except (OSError, ValueError, IndexError):
        return None
    for key in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap"):
        if key in status:
            sample[key + "_kib"] = status[key]
    try:
        io = _counters(proc / "io")
        if "read_bytes" in io:
            sample["read_bytes"] = io["read_bytes"]
    except OSError:
        pass
    try:
        mem = _counters(Path(proc_root) / "meminfo")
        if "MemAvailable" in mem:
            sample["available_kib"] = mem["MemAvailable"]
    except OSError:
        pass
    cards = []
    for card in sorted(Path(drm_root).glob("card[0-9]*")):
        # Skip connector entries, e.g. card0-DP-1.
        if not card.name[4:].isdigit():
            continue
        reading = {"card": card.name}
        for source, target in (("mem_info_vram_used", "vram_used_bytes"),
                               ("mem_info_gtt_used", "gtt_used_bytes"),
                               ("gpu_busy_percent", "busy_percent")):
            try:
                reading[target] = int((card / "device" / source).read_text().strip())
            except (OSError, ValueError):
                pass
        if len(reading) > 1:
            cards.append(reading)
    if cards:
        sample["amd_cards"] = cards
    return sample
