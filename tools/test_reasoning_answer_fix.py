import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.enable_reasoning_answer_fix import enable


class EnableAnswerFix(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        source = root / 'server.py'; source.write_bytes(b'reasoning_close_retry v2\nclass IncompleteAnswer')
        config = {'exe': '/mnt/ssd/Strata/engine/strata', 'args': ['--max-context', '131072', '--kv', 'int8', '--spec', '4'],
                  'env': {'STRATA_RESIDENT_PIN': '0', 'STRATA_RESIDENT_HEADROOM_GIB': '8', 'STRATA_PROMPT_STREAM': '1',
                          'STRATA_RAM_WORKING_SET': '1', 'STRATA_RESIDENT_SWAP_PREFETCH': '1'},
                  'reasoning_auto_budget_tokens': 4096, 'reasoning_effort': 'high',
                  'sampling': {'temperature': 0, 'presence_penalty': 0}, 'vision': {'gpu': False, 'max_tokens': 300}}
        path = root / 'config.json'; path.write_text(json.dumps(config)); path.chmod(0o600)
        return source, config, path

    def test_only_the_requested_recovery_flag_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, before, path = self.fixture(directory)
            after, backup = enable(path, source=source)
            self.assertEqual(after, {**before, 'reasoning_close_retry': True})
            self.assertEqual(json.loads(backup.read_text()), before)
            self.assertEqual(json.loads(path.read_text()), after)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_old_server_is_rejected_before_config_or_backups_change(self):
        with tempfile.TemporaryDirectory() as directory:
            source, before, path = self.fixture(directory); source.write_text('old server')
            with self.assertRaises(ValueError): enable(path, source=source)
            self.assertEqual(json.loads(path.read_text()), before)
            self.assertEqual(list(path.parent.glob('*.before-reasoning-answer-*')), [])

    def test_invalid_config_is_rejected_before_backups(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _, path = self.fixture(directory); path.write_text('[]')
            with self.assertRaises(ValueError): enable(path, source=source)
            self.assertEqual(path.read_text(), '[]')
            self.assertEqual(list(path.parent.glob('*.before-reasoning-answer-*')), [])

    def test_failed_atomic_replace_preserves_the_config_and_cleans_the_tempfile(self):
        with tempfile.TemporaryDirectory() as directory:
            source, before, path = self.fixture(directory)
            with patch('os.replace', side_effect=OSError('fixture replacement failure')):
                with self.assertRaises(OSError): enable(path, source=source)
            self.assertEqual(json.loads(path.read_text()), before)
            self.assertEqual(list(path.parent.glob('.strata-reasoning-answer-*')), [])

    def test_enabling_is_idempotent_and_preserves_an_explicit_token_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            source, before, path = self.fixture(directory)
            before.update(reasoning_budget_tokens=16000, reasoning_close_retry=False)
            path.write_text(json.dumps(before))
            first, _ = enable(path, source=source)
            second, _ = enable(path, source=source)
            self.assertEqual(first, second)
            self.assertEqual(second, {**before, 'reasoning_close_retry': True})


if __name__ == '__main__': unittest.main()
