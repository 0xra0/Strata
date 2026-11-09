"""Image handoff must not allocate another whole embedding file in Python."""
import errno
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve.server import Vision, combined_embeddings_path, write_temporary


class VisionMemory(unittest.TestCase):
    def test_single_image_is_shared_and_survives_cache_eviction(self):
        with tempfile.TemporaryDirectory() as d:
            source, target = Path(d) / "cached.sve", Path(d) / "request.sve"
            source.write_bytes(b"unchanged encoder output")
            with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file read")):
                write_temporary(target, [source])
            self.assertEqual(source.stat().st_ino, target.stat().st_ino)
            source.unlink()
            self.assertEqual(target.read_bytes(), b"unchanged encoder output")

    def test_multiple_images_and_cross_device_fallback_stream_exact_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, out = (Path(d) / n for n in ("a.sve", "b.sve", "out.sve"))
            data = bytes(range(256)) * 10000
            a.write_bytes(data)
            b.write_bytes(b"last record")
            for parts in ([a], [a, b]):
                out.unlink(missing_ok=True)
                with mock.patch("serve.server.os.link", side_effect=OSError(errno.EXDEV, "different filesystem")), \
                        mock.patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file read")):
                    write_temporary(out, parts)
                self.assertEqual(out.read_bytes(), data + (b"last record" if len(parts) == 2 else b""))

    def test_streaming_reads_are_bounded(self):
        class BoundedReader(io.BytesIO):
            def read(self, n=-1):
                if not 0 < n <= 1 << 20:
                    raise AssertionError(f"unbounded read: {n}")
                return super().read(n)
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(Path, "open", side_effect=lambda *a, **k: BoundedReader(b"abc" * 1000000)):
            out = Path(d) / "out"
            write_temporary(out, [Path("one"), Path("two")])
            with open(out, "rb") as f:
                self.assertEqual(f.read(), b"abc" * 2000000)

    def test_disk_handoff_does_not_select_shm(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch("serve.server.shutil.disk_usage", side_effect=AssertionError("must not probe shm")):
            self.assertEqual(combined_embeddings_path(Path(d), 1 << 30, prefer_memory=False).parent, Path(d))

    def test_explicit_work_directory_and_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            vision = Vision({"exe": "strata-vision", "mmproj": "m", "model": "t", "work_dir": d,
                             "memory_efficient": True}, lazy=True)
            self.assertEqual(vision.dir.parent, Path(d))
            self.assertTrue(vision.memory_efficient)
            vision.shutdown()
            self.assertFalse(vision.dir.exists())
            with self.assertRaises(OSError):
                Vision.work_dir(str(Path(d) / "missing"))
            with self.assertRaises(ValueError):
                Vision.work_dir(str(Path(d) / "has space"))


if __name__ == "__main__":
    unittest.main()
