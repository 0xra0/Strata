"""rebuild-hip.sh installs only after a successful build with every marker, and keeps one backup."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

MARKERS = ('idle GPU shadows: live ownership v2 RAM working set v1 '
           'resident swap prefetch v1 prompt file reader v1')
FAKE_CMAKE = '''#!/bin/bash
set -eu
if [[ "$1" == "--build" ]]; then
    [[ "${TEST_FAIL:-0}" == "0" ]] || exit 42
    if [[ "$4" == "strata-vision" ]]; then
        [[ "${TEST_FAIL_VISION:-0}" == "0" ]] || exit 43
        mkdir -p "$2/bin"
        printf '#!/bin/sh\\nexit 0\\n' > "$2/bin/strata-vision"
        chmod +x "$2/bin/strata-vision"
    else
        mkdir -p "$2"
        printf '#!/bin/sh\\n# %s\\nexit 0\\n' "$TEST_MARKERS" > "$2/strata"
        chmod +x "$2/strata"
    fi
fi
'''


class RebuildHip(unittest.TestCase):
    def run_case(self, *args, fail=False, fail_vision=False, stale=False, runs=1):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'Strata project'
            for n in ('tools', 'engine', 'bin', 'third_party/llama.cpp/ggml'):
                (root / n).mkdir(parents=True)
            (root / 'third_party/llama.cpp/ggml/CMakeLists.txt').touch()
            shutil.copyfile(Path(__file__).with_name('rebuild-hip.sh'), root / 'tools/rebuild-hip.sh')
            engine, vision = root / 'engine/strata', root / 'engine/strata-vision'
            engine.write_bytes(b'old engine'); vision.write_bytes(b'old vision')
            cmake = root / 'bin/cmake'; cmake.write_text(FAKE_CMAKE); cmake.chmod(0o755)
            env = {**os.environ, 'PATH': str(root / 'bin') + os.pathsep + os.environ['PATH'],
                   'TEST_FAIL': '1' if fail else '0', 'TEST_FAIL_VISION': '1' if fail_vision else '0',
                   'TEST_MARKERS': 'old engine' if stale else MARKERS}
            for _ in range(runs):
                r = subprocess.run(['bash', str(root / 'tools/rebuild-hip.sh'), *args], env=env,
                                   capture_output=True, text=True)
            engine_dir = sorted(p.name for p in (root / 'engine').iterdir())
            if fail or fail_vision or stale:
                self.assertNotEqual(r.returncode, 0, r.stderr)
                self.assertEqual(engine.read_bytes(), b'old engine')
                self.assertEqual(vision.read_bytes(), b'old vision')
                self.assertEqual(engine_dir, ['strata', 'strata-vision'])
                return
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(MARKERS.encode(), engine.read_bytes())
            self.assertTrue(os.access(engine, os.X_OK))
            if runs == 1:
                self.assertEqual((root / 'engine/strata.prev').read_bytes(), b'old engine')
            if '--vision' in args:
                self.assertEqual((root / 'engine/strata-vision.prev').read_bytes(), b'old vision')
                self.assertEqual(engine_dir, ['strata', 'strata-vision', 'strata-vision.prev', 'strata.prev'])
            else:
                self.assertEqual(vision.read_bytes(), b'old vision')
                self.assertEqual(engine_dir, ['strata', 'strata-vision', 'strata.prev'])

    def test_engine_only_install_keeps_one_backup(self): self.run_case()
    def test_repeated_builds_do_not_pile_up_backups(self): self.run_case(runs=3)
    def test_vision_install(self): self.run_case('--vision')
    def test_failed_build_preserves_engine(self): self.run_case(fail=True)
    def test_failed_vision_build_preserves_both(self): self.run_case('--vision', fail_vision=True)
    def test_engine_without_a_feature_is_not_installed(self): self.run_case(stale=True)


if __name__ == '__main__':
    unittest.main()
