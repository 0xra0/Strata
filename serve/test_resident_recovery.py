import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from serve.runtime_memory import sample_runtime_memory
from serve.server import Service, StrataEngine, Vision
from tools.recover_fast_resident import recover


class ResidentRecovery(unittest.TestCase):
    def test_recovery_only_changes_storage_override_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "strata-coder-uncen-iq1_m.json"
            before = {"args": ["--resident-experts", "--max-context", "131072", "--prefill", "auto", "--spec", "4"],
                      "env": {"STRATA_RESIDENT_FILE_DIR": "/ssd", "STRATA_RESIDENT_HEADROOM_GIB": "8"},
                      "vision": {"gpu": False, "max_tokens": 300, "threads": 8}, "log": "/active-uncen.log"}
            path.write_text(json.dumps(before))
            mode = path.stat().st_mode
            result, backup = recover(path)
            self.assertEqual(json.loads(backup.read_text()), before)
            self.assertEqual(result["args"], before["args"])
            self.assertEqual(result["vision"], before["vision"])
            self.assertEqual(result["env"]["STRATA_RESIDENT_HEADROOM_GIB"], "8")
            self.assertEqual(result["env"]["STRATA_RESIDENT_FILE_DIR"], "")
            self.assertTrue(result["runtime_memory_log"])
            self.assertEqual(path.stat().st_mode, mode)
            self.assertEqual(json.loads(path.read_text()), result)
            self.assertEqual(list(Path(d).glob(".strata-recovery-*")), [])

    def test_invalid_config_is_not_modified(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            original = '{"args": ["--different-mode"]}'
            path.write_text(original)
            with self.assertRaises(ValueError):
                recover(path)
            self.assertEqual(path.read_text(), original)
            self.assertEqual(list(Path(d).iterdir()), [path])

    @unittest.skipIf(os.name == "nt", "POSIX signal return codes")
    def test_interruption_does_not_become_an_oom_or_old_watchdog_report(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.write_text("strata: issue #29 from an earlier start\n")
            engine = StrataEngine.__new__(StrataEngine)
            engine.log_path = str(log)
            for signum in (signal.SIGINT, signal.SIGTERM):
                engine.proc = SimpleNamespace(poll=lambda: -signum)
                note = engine.death_note()
                self.assertIn(signal.Signals(signum).name, note)
                self.assertNotIn("usual cause", note)
                self.assertNotIn("smaller model", note)
                self.assertNotIn("hang it caught", note)

    def test_vision_close_after_child_exit_closes_buffered_pipes(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, text=True)
        proc.wait(timeout=10)
        vision = Vision.__new__(Vision)
        vision.proc, vision.stopped = proc, False
        vision.close()
        self.assertTrue(proc.stdin.closed)
        self.assertTrue(proc.stdout.closed)
        self.assertTrue(vision.stopped)
        self.assertIsNone(vision.proc)
        vision.close()


class RuntimeMemory(unittest.TestCase):
    def fixture(self, d):
        root = Path(d)
        proc = root / "proc/123"
        proc.mkdir(parents=True)
        (proc / "stat").write_text("123 (name with ) and spaces) S 1 2 3 4 5 6 50 60 70 80\n")
        (proc / "status").write_text("VmRSS: 1024 kB\nRssAnon: 900 kB\nRssFile: 124 kB\nVmSwap: 456 kB\n")
        (proc / "io").write_text("read_bytes: 777\nrchar: 888\n")
        (root / "proc/meminfo").write_text("MemAvailable: 999 kB\n")
        gpu = root / "drm/card1/device"
        gpu.mkdir(parents=True)
        (gpu / "mem_info_vram_used").write_text("123456")
        (gpu / "mem_info_gtt_used").write_text("654321")
        (gpu / "gpu_busy_percent").write_text("17")
        return root

    def test_reads_engine_faults_swap_disk_and_amd_counters(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.fixture(d)
            sample = sample_runtime_memory(123, root / "proc", root / "drm")
            self.assertEqual(sample["major_faults"], 70)
            self.assertEqual(sample["VmSwap_kib"], 456)
            self.assertEqual(sample["RssAnon_kib"], 900)
            self.assertEqual(sample["read_bytes"], 777)
            self.assertEqual(sample["available_kib"], 999)
            self.assertEqual(sample["amd_cards"][0]["gtt_used_bytes"], 654321)
            self.assertEqual(sample["amd_cards"][0]["busy_percent"], 17)

    def test_missing_optional_counters_are_unknown_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            root = self.fixture(d)
            (root / "proc/123/io").unlink()
            (root / "proc/meminfo").unlink()
            sample = sample_runtime_memory(123, root / "proc", root / "missing-drm")
            for key in ("read_bytes", "available_kib", "amd_cards", "RssShmem_kib"):
                self.assertNotIn(key, sample)
            self.assertIsNone(sample_runtime_memory(999, root / "proc", root / "drm"))

    def test_throttled_log_uses_running_engine_log_and_counter_deltas(self):
        with tempfile.TemporaryDirectory() as d:
            svc = Service.__new__(Service)
            log = Path(d) / "active-coder-uncen.log"
            svc.engine = SimpleNamespace(proc=SimpleNamespace(pid=123), log_path=log)
            readings = [{"pid": 123, "major_faults": 10, "read_bytes": 100},
                        {"pid": 123, "major_faults": 40, "read_bytes": 300}]
            with mock.patch("serve.server.time.monotonic", side_effect=[10, 11, 21]), \
                    mock.patch("serve.runtime_memory.sample_runtime_memory", side_effect=readings) as sampler, \
                    contextlib.redirect_stdout(io.StringIO()):
                svc._log_runtime_memory()
                svc._log_runtime_memory()
                svc._log_runtime_memory()
            self.assertEqual(sampler.call_count, 2)
            lines = log.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            result = json.loads(lines[-1].split(": ", 1)[1])
            self.assertEqual(result["major_faults_delta"], 30)
            self.assertEqual(result["read_bytes_delta"], 200)


if __name__ == "__main__":
    unittest.main()
