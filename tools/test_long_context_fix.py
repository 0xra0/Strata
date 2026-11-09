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


class Rebuild(unittest.TestCase):
    def run_case(self, fail=False, stale=False):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'Strata project'
            for n in ('tools', 'engine', 'build-memory-fix-hip', 'bin'): (root / n).mkdir(parents=True)
            (root / 'build-memory-fix-hip/CMakeCache.txt').touch()
            tools = Path(__file__).parent
            for n in ('rebuild-long-context-fix.sh', 'configure_long_context.py'): shutil.copyfile(tools / n, root / 'tools' / n)
            engine, vision = root / 'engine/strata', root / 'engine/strata-vision'
            engine.write_bytes(b'previous engine'); vision.write_bytes(b'previous vision')
            p = root / 'strata-coder-uncen-iq1_m.json'
            before = {'exe': str(engine), 'args': ['--resident-experts', '--max-context', '131072'],
                      'vision': {'gpu': False, 'max_tokens': 300}}
            p.write_text(json.dumps(before))
            cmake = root / 'bin/cmake'
            cmake.write_text('''#!/bin/bash
set -eu
if [[ "$1" == "--build" ]]; then
    [[ "${TEST_FAIL:-0}" == "0" ]] || exit 42
    printf '#!/bin/sh\\n# %s\\nexit 0\\n' "${TEST_MARKER}" > "$2/strata"
    chmod +x "$2/strata"
fi
'''); cmake.chmod(0o755)
            env = {**os.environ, 'PATH': str(root / 'bin') + os.pathsep + os.environ['PATH'],
                   'TEST_FAIL': '1' if fail else '0', 'TEST_MARKER': 'old engine' if stale else MARKERS}
            result = subprocess.run(['bash', str(root / 'tools/rebuild-long-context-fix.sh')], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(vision.read_bytes(), b'previous vision')
            backups = list((root / 'engine').glob('strata.pre-long-context.*'))
            if fail or stale:
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertEqual(engine.read_bytes(), b'previous engine')
                self.assertEqual(json.loads(p.read_text()), before)
                self.assertEqual(backups, [])
            else:
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups[0].read_bytes(), b'previous engine')
                cfg = json.loads(p.read_text())
                self.assertEqual(cfg['args'], before['args'])
                self.assertEqual(cfg['vision'], before['vision'])
                self.assertEqual(cfg['reasoning_auto_budget_tokens'], 4096)

    def test_install_and_activation(self): self.run_case()
    def test_failed_build_preserves_everything(self): self.run_case(fail=True)
    def test_stale_engine_cannot_activate(self): self.run_case(stale=True)


if __name__ == '__main__': unittest.main()
