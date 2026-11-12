import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from tools.configure_long_context import configure

MARKERS = 'idle GPU shadows: live ownership v2 RAM working set v1'


class Configure(unittest.TestCase):
    def fixture(self, d):
        root = Path(d)
        engine = root / 'strata'
        engine.write_bytes(MARKERS.encode())
        config = {'exe': str(engine), 'args': ['--resident-experts', '--max-context', '131072', '--kv', 'int8',
                  '--prefill', 'auto', '--spec', '4', '--vram-reserve-mib', '700'],
                  'env': {'STRATA_RESIDENT_HEADROOM_GIB': '8', 'STRATA_RESIDENT_PIN': '0'},
                  'vision': {'max_tokens': 300, 'threads': 8, 'release_workspace': True}}
        p = root / 'config.json'; p.write_text(json.dumps(config))
        return p, config, engine

    def test_preserves_model_memory_and_image_settings(self):
        with tempfile.TemporaryDirectory() as d:
            p, before, _ = self.fixture(d)
            after, backup = configure(p)
            self.assertEqual(after['args'], before['args'])
            self.assertEqual(after['vision'], before['vision'])
            self.assertTrue(all(after['env'][k] == v for k, v in before['env'].items()))
            self.assertEqual(after['env']['STRATA_RAM_WORKING_SET'], '1')
            self.assertEqual(after['reasoning_auto_budget_tokens'], 4096)
            self.assertEqual(json.loads(backup.read_text()), before)

    def test_explicit_budgets_and_legacy_ram_override_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p, cfg, _ = self.fixture(d)
            cfg.update(reasoning_budget_tokens=16000, reasoning_auto_budget_tokens=8000)
            cfg['env']['STRATA_RAM_ADAPT'] = '0'
            p.write_text(json.dumps(cfg))
            after, _ = configure(p)
            self.assertEqual(after['reasoning_budget_tokens'], 16000)
            self.assertEqual(after['reasoning_auto_budget_tokens'], 8000)
            self.assertEqual(after['env']['STRATA_RAM_ADAPT'], '0')

    def test_old_binary_is_rejected_without_writing(self):
        with tempfile.TemporaryDirectory() as d:
            p, cfg, engine = self.fixture(d)
            engine.write_bytes(b'idle GPU shadows: live ownership v2')
            with self.assertRaises(ValueError): configure(p)
            self.assertEqual(json.loads(p.read_text()), cfg)
            self.assertEqual(list(p.parent.glob('*.before-long-context-*')), [])



if __name__ == '__main__': unittest.main()
