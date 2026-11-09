import json
from pathlib import Path
import tempfile
import unittest
from tools.configure_memory_lifecycle import configure


class MemoryProfile(unittest.TestCase):
    def fixture(self, d):
        root = Path(d)
        engine, encoder = root / "strata", root / "strata-vision"
        engine.write_bytes(b"STRATA_MEMORY_LIFECYCLE idle GPU shadows: live ownership v2")
        encoder.write_bytes(b"--release-workspace")
        cfg = {"exe": str(engine), "args": ["--resident-experts", "--max-context", "131072", "--kv", "int8",
                                              "--prefill", "auto", "--spec", "4"],
               "env": {"STRATA_RESIDENT_HEADROOM_GIB": "8"},
               "vision": {"exe": str(encoder), "gpu": False, "max_tokens": 300, "threads": 8}}
        p = root / "config.json"
        p.write_text(json.dumps(cfg))
        return p, cfg, encoder

    def test_enable_preserves_inference_and_vision_values(self):
        with tempfile.TemporaryDirectory() as d:
            p, before, _ = self.fixture(d)
            after, backup = configure(p)
            self.assertEqual(after["args"], before["args"])
            self.assertTrue(all(after["vision"][k] == v for k, v in before["vision"].items()))
            self.assertEqual(after["env"]["STRATA_RESIDENT_HEADROOM_GIB"], "8")
            self.assertEqual(after["env"]["STRATA_RESIDENT_FILE_DIR"], "")
            self.assertEqual(after["env"]["STRATA_MEMORY_LIFECYCLE"], "1")
            self.assertTrue(after["vision"]["release_workspace"])
            self.assertEqual(json.loads(backup.read_text()), before)

    def test_old_encoder_cannot_enable_an_unsupported_option(self):
        with tempfile.TemporaryDirectory() as d:
            p, before, encoder = self.fixture(d)
            encoder.write_bytes(b"older encoder")
            with self.assertRaisesRegex(ValueError, "rebuilt"):
                configure(p)
            self.assertEqual(json.loads(p.read_text()), before)
            self.assertEqual(list(p.parent.glob("*.before-memory-lifecycle-*")), [])

    def test_gpu_vision_is_not_silently_changed_to_cpu(self):
        with tempfile.TemporaryDirectory() as d:
            p, before, _ = self.fixture(d)
            before["vision"]["gpu"] = True
            p.write_text(json.dumps(before))
            with self.assertRaises(ValueError):
                configure(p)
            self.assertEqual(json.loads(p.read_text()), before)

    def test_previous_unsafe_engine_cannot_enable_profile(self):
        with tempfile.TemporaryDirectory() as d:
            p, before, _ = self.fixture(d)
            Path(before["exe"]).write_bytes(b"STRATA_MEMORY_LIFECYCLE")
            with self.assertRaisesRegex(ValueError, "rebuilt"):
                configure(p)
            self.assertEqual(json.loads(p.read_text()), before)
            self.assertEqual(list(p.parent.glob("*.before-memory-lifecycle-*")), [])


if __name__ == "__main__":
    unittest.main()
