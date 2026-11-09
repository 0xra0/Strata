"""The rebuild must install only after a successful build and preserve the previous engine."""
import os
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class RebuildMemoryFix(unittest.TestCase):
    def run_rebuild(self, fail=False, fail_vision=False, with_profile=False):
        script = Path(__file__).with_name("rebuild-memory-fix.sh")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "Strata project"
            (root / "tools").mkdir(parents=True)
            (root / "engine").mkdir()
            (root / "third_party/llama.cpp/ggml").mkdir(parents=True)
            (root / "third_party/llama.cpp/ggml/CMakeLists.txt").touch()
            shutil.copyfile(script, root / "tools/rebuild-memory-fix.sh")
            old = root / "engine/strata"
            old.write_text("old engine bytes")
            old_vision = root / "engine/strata-vision"
            old_vision.write_text("old vision bytes")
            profile = None
            if with_profile:
                shutil.copyfile(script.with_name("configure_memory_lifecycle.py"), root / "tools/configure_memory_lifecycle.py")
                profile = {"exe": str(old), "args": ["--resident-experts", "--max-context", "131072"],
                           "vision": {"exe": str(old_vision), "gpu": False, "max_tokens": 300}}
                (root / "strata-coder-uncen-iq1_m.json").write_text(json.dumps(profile))
            bindir = Path(d) / "bin"
            bindir.mkdir()
            cmake = bindir / "cmake"
            cmake.write_text('''#!/bin/bash
set -eu
if [[ "$1" == "--build" ]]; then
    [[ "${STRATA_TEST_FAIL:-0}" == "0" ]] || exit 42
    if [[ "$4" == "strata-vision" ]]; then
        [[ "${STRATA_TEST_FAIL_VISION:-0}" == "0" ]] || exit 43
        mkdir -p "$2/bin"
        printf '#!/bin/sh\\n# --release-workspace\\nexit 0\\n' > "$2/bin/strata-vision"
        chmod +x "$2/bin/strata-vision"
    else
        mkdir -p "$2"
        printf '#!/bin/sh\\n# STRATA_MEMORY_LIFECYCLE idle GPU shadows: live ownership v2\\nexit 0\\n' > "$2/strata"
        chmod +x "$2/strata"
    fi
fi
''')
            cmake.chmod(0o755)
            env = {**os.environ, "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
                   "STRATA_TEST_FAIL": "1" if fail else "0",
                   "STRATA_TEST_FAIL_VISION": "1" if fail_vision else "0"}
            result = subprocess.run(["bash", str(root / "tools/rebuild-memory-fix.sh")], env=env,
                                    capture_output=True, text=True)
            backups = list((root / "engine").glob("strata.pre-memory-fix.*"))
            if fail or fail_vision:
                self.assertEqual(result.returncode, 42 if fail else 43, result.stderr)
                self.assertEqual(old.read_text(), "old engine bytes")
                self.assertEqual(old_vision.read_text(), "old vision bytes")
                self.assertEqual(backups, [])
                if with_profile:
                    self.assertEqual(json.loads((root / "strata-coder-uncen-iq1_m.json").read_text()), profile)
            else:
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups[0].read_text(), "old engine bytes")
                self.assertTrue(os.access(old, os.X_OK))
                self.assertEqual(old.read_text(), "#!/bin/sh\n# STRATA_MEMORY_LIFECYCLE idle GPU shadows: live ownership v2\nexit 0\n")
                vision_backups = list((root / "engine").glob("strata-vision.pre-memory-fix.*"))
                self.assertEqual(len(vision_backups), 1)
                self.assertEqual(vision_backups[0].read_text(), "old vision bytes")
                self.assertEqual(old_vision.read_text(), "#!/bin/sh\n# --release-workspace\nexit 0\n")
                if with_profile:
                    cfg = json.loads((root / "strata-coder-uncen-iq1_m.json").read_text())
                    self.assertEqual(cfg["args"], profile["args"])
                    self.assertEqual(cfg["env"]["STRATA_MEMORY_LIFECYCLE"], "1")
                    self.assertTrue(cfg["vision"]["release_workspace"])
                self.assertEqual(list((root / "engine").glob(".strata-new.*")), [])

    def test_success_preserves_backup(self):
        self.run_rebuild()

    def test_failed_build_preserves_engine(self):
        self.run_rebuild(fail=True)

    def test_failed_vision_build_preserves_both_binaries(self):
        self.run_rebuild(fail_vision=True, with_profile=True)

    def test_both_binaries_installed_before_profile_is_enabled(self):
        self.run_rebuild(with_profile=True)


if __name__ == "__main__":
    unittest.main()
