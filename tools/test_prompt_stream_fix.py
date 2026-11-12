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
from tools.configure_prompt_stream import configure
from tools.check_translation_http import fetch, check

MARKERS = 'idle GPU shadows: live ownership v2 resident swap prefetch v1 STRATA_PROMPT_STREAM prompt file reader v1'


class Configure(unittest.TestCase):
    def fixture(self, d):
        root = Path(d); engine = root / 'strata'; engine.write_bytes(MARKERS.encode())
        original = {'exe': str(engine), 'args': ['--resident-experts', '--max-context', '131072', '--kv', 'int8'],
                    'env': {'STRATA_RESIDENT_PIN': '0', 'STRATA_RESIDENT_HEADROOM_GIB': '8', 'STRATA_RAM_WORKING_SET': '1'},
                    'reasoning_auto_budget_tokens': 4096, 'vision': {'gpu': False, 'max_tokens': 300}}
        path = root / 'config.json'; path.write_text(json.dumps(original))
        return path, original, engine

    def test_only_prompt_stream_and_diagnostics_are_enabled(self):
        with tempfile.TemporaryDirectory() as d:
            p, before, _ = self.fixture(d)
            after, backup = configure(p)
            self.assertEqual(json.loads(backup.read_text()), before)
            expected = json.loads(json.dumps(before))
            expected['env']['STRATA_PROMPT_STREAM'] = '1'; expected['runtime_memory_log'] = True
            self.assertEqual(after, expected)

    def test_existing_opt_out_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p, cfg, _ = self.fixture(d)
            cfg['env'].update(STRATA_PROMPT_DIRECT='1', STRATA_PROMPT_STREAM='0')
            p.write_text(json.dumps(cfg)); after, _ = configure(p)
            self.assertEqual(after['env']['STRATA_PROMPT_DIRECT'], '1')
            self.assertEqual(after['env']['STRATA_PROMPT_STREAM'], '0')

    def test_old_engine_is_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory() as d:
            p, cfg, exe = self.fixture(d); exe.write_bytes(b'idle GPU shadows: live ownership v2')
            with self.assertRaises(ValueError): configure(p)
            self.assertEqual(json.loads(p.read_text()), cfg)
            self.assertEqual(list(p.parent.glob('*.before-prompt-stream-*')), [])



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
