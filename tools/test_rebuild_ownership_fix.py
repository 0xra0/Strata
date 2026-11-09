"""Engine-only install must preserve the current config and vision binary."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class OwnershipRebuild(unittest.TestCase):
    def run_case(self, fail=False, stale=False):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "Strata project"
            for p in ("tools", "engine", "build-memory-fix-hip", "bin"):
                (root / p).mkdir(parents=True)
            (root / "build-memory-fix-hip/CMakeCache.txt").touch()
            script = Path(__file__).with_name("rebuild-ownership-fix.sh")
            shutil.copyfile(script, root / "tools" / script.name)
            engine = root / "engine/strata"
            engine.write_bytes(b"previous engine")
            vision = root / "engine/strata-vision"
            vision.write_bytes(b"previous vision")
            config = root / "strata-coder-uncen-iq1_m.json"
            config.write_bytes(b'{"config":"preserve exact bytes"}\n')
            cmake = root / "bin/cmake"
            cmake.write_text('''#!/bin/bash
set -eu
if [[ "$1" == "--build" ]]; then
    [[ "${TEST_FAIL:-0}" == "0" ]] || exit 42
    printf '#!/bin/sh\\n# %s\\nexit 0\\n' "${TEST_MARKER}" > "$2/strata"
    chmod +x "$2/strata"
fi
''')
            cmake.chmod(0o755)
            env = {**os.environ, "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
                   "TEST_FAIL": "1" if fail else "0",
                   "TEST_MARKER": "old marker" if stale else "idle GPU shadows: live ownership v2"}
            p = subprocess.run(["bash", str(root / "tools" / script.name)], env=env, capture_output=True, text=True)
            backups = list((root / "engine").glob("strata.pre-ownership-fix.*"))
            self.assertEqual(vision.read_bytes(), b"previous vision")
            self.assertEqual(config.read_bytes(), b'{"config":"preserve exact bytes"}\n')
            self.assertEqual(list((root / "engine").glob(".strata-ownership-new.*")), [])
            if fail or stale:
                self.assertNotEqual(p.returncode, 0, p.stderr)
                self.assertEqual(engine.read_bytes(), b"previous engine")
                self.assertEqual(backups, [])
            else:
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertIn(b"live ownership v2", engine.read_bytes())
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups[0].read_bytes(), b"previous engine")

    def test_installs_only_engine(self):
        self.run_case()

    def test_build_failure_preserves_engine(self):
        self.run_case(fail=True)

    def test_rejects_uncorrected_binary(self):
        self.run_case(stale=True)


if __name__ == "__main__":
    unittest.main()
