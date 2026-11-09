import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
from tools.configure_swap_prefetch import configure
from tools.check_translation_http import fetch, check

MARKERS = 'idle GPU shadows: live ownership v2 resident swap prefetch v1 STRATA_RESIDENT_SWAP_PREFETCH'


class Configure(unittest.TestCase):
    def fixture(self, d):
        root = Path(d); engine = root / 'strata'; engine.write_bytes(MARKERS.encode())
        original = {'exe': str(engine), 'args': ['--resident-experts', '--max-context', '131072', '--kv', 'int8'],
                    'env': {'STRATA_RESIDENT_PIN': '0', 'STRATA_RESIDENT_HEADROOM_GIB': '8', 'STRATA_RAM_WORKING_SET': '1'},
                    'reasoning_auto_budget_tokens': 4096, 'vision': {'gpu': False, 'max_tokens': 300}}
        path = root / 'config.json'; path.write_text(json.dumps(original))
        return path, original, engine

    def test_only_prefetch_and_diagnostics_are_enabled(self):
        with tempfile.TemporaryDirectory() as d:
            p, before, _ = self.fixture(d)
            after, backup = configure(p)
            self.assertEqual(json.loads(backup.read_text()), before)
            expected = json.loads(json.dumps(before))
            expected['env']['STRATA_RESIDENT_SWAP_PREFETCH'] = '1'; expected['runtime_memory_log'] = True
            self.assertEqual(after, expected)

    def test_existing_opt_out_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p, cfg, _ = self.fixture(d)
            cfg['env'].update(STRATA_LOOKAHEAD='0', STRATA_RESIDENT_SWAP_PREFETCH='0')
            p.write_text(json.dumps(cfg)); after, _ = configure(p)
            self.assertEqual(after['env']['STRATA_LOOKAHEAD'], '0')
            self.assertEqual(after['env']['STRATA_RESIDENT_SWAP_PREFETCH'], '0')

    def test_old_engine_is_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory() as d:
            p, cfg, exe = self.fixture(d); exe.write_bytes(b'idle GPU shadows: live ownership v2')
            with self.assertRaises(ValueError): configure(p)
            self.assertEqual(json.loads(p.read_text()), cfg)
            self.assertEqual(list(p.parent.glob('*.before-swap-prefetch-*')), [])


class Install(unittest.TestCase):
    def run_case(self, fail=False, stale=False):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'Strata project'
            for name in ('tools', 'engine', 'build-memory-fix-hip', 'bin'): (root / name).mkdir(parents=True)
            (root / 'build-memory-fix-hip/CMakeCache.txt').touch()
            for name in ('rebuild-swap-prefetch-fix.sh', 'configure_swap_prefetch.py'):
                shutil.copyfile(Path(__file__).with_name(name), root / 'tools' / name)
            exe = root / 'engine/strata'; exe.write_bytes(b'old engine')
            vision = root / 'engine/strata-vision'; vision.write_bytes(b'old vision')
            cfg = {'exe': str(exe), 'args': ['--resident-experts', '--max-context', '131072'],
                   'reasoning_auto_budget_tokens': 8192, 'vision': {'gpu': False, 'max_tokens': 300}}
            path = root / 'strata-coder-uncen-iq1_m.json'; path.write_text(json.dumps(cfg))
            cmake = root / 'bin/cmake'
            cmake.write_text('''#!/bin/bash
set -eu
if [[ "$1" == "--build" ]]; then
    [[ "${TEST_FAIL:-0}" == "0" ]] || exit 42
    printf '#!/bin/sh\\n# %s\\nexit 0\\n' "${TEST_MARKERS}" > "$2/strata"
    chmod +x "$2/strata"
fi
'''); cmake.chmod(0o755)
            env = {**os.environ, 'PATH': str(root / 'bin') + os.pathsep + os.environ['PATH'],
                   'TEST_FAIL': '1' if fail else '0', 'TEST_MARKERS': 'old engine' if stale else MARKERS}
            r = subprocess.run(['bash', str(root / 'tools/rebuild-swap-prefetch-fix.sh')], env=env, capture_output=True, text=True)
            self.assertEqual(vision.read_bytes(), b'old vision')
            backups = list((root / 'engine').glob('strata.pre-swap-prefetch.*'))
            if fail or stale:
                self.assertNotEqual(r.returncode, 0, r.stderr)
                self.assertEqual(exe.read_bytes(), b'old engine')
                self.assertEqual(json.loads(path.read_text()), cfg)
                self.assertEqual(backups, [])
            else:
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(len(backups), 1); self.assertEqual(backups[0].read_bytes(), b'old engine')
                after = json.loads(path.read_text())
                self.assertEqual(after['args'], cfg['args']); self.assertEqual(after['vision'], cfg['vision'])
                self.assertEqual(after['reasoning_auto_budget_tokens'], 8192)

    def test_install_and_activate(self): self.run_case()
    def test_failed_build_preserves_everything(self): self.run_case(fail=True)
    def test_stale_binary_is_not_installed(self): self.run_case(stale=True)


class Response(io.BytesIO):
    status = 200


class HTTPProbe(unittest.TestCase):
    def test_empty_body_reports_status_and_length(self):
        with patch('urllib.request.urlopen', return_value=Response(b'')):
            r = fetch('http://127.0.0.1:8099/x')
        self.assertEqual(r['http_status'], 200); self.assertEqual(r['body_bytes'], 0)
        self.assertIn('Empty', r['error'])

    def test_http_error_body_is_retained(self):
        e = urllib.error.HTTPError('http://127.0.0.1:8099/x', 503, 'busy', {}, io.BytesIO(b'{"error":"busy"}'))
        with patch('urllib.request.urlopen', side_effect=e): r = fetch(e.url)
        self.assertEqual(r['http_status'], 503); self.assertEqual(r['response']['error'], 'busy')

    def test_two_identical_isolated_requests_preserve_ukrainian(self):
        seen = []
        def open_request(req, **kw):
            if req.data is None: return Response(b'{"data":[{"id":"active-model"}]}')
            body = json.loads(req.data.decode('utf-8')); seen.append(body)
            return Response(b'{"choices":[{"message":{"content":"Hi! How are you doing?"}}]}')
        with patch('urllib.request.urlopen', side_effect=open_request): report = check('http://127.0.0.1:8099')
        self.assertEqual(len(report['requests']), 2); self.assertEqual(seen[0], seen[1])
        self.assertIn('Привіт! Як у тебе справи?', seen[0]['messages'][0]['content'])
        self.assertEqual(seen[0]['reasoning_effort'], 'none')
        self.assertEqual(seen[0]['model'], 'active-model')


if __name__ == '__main__': unittest.main()
