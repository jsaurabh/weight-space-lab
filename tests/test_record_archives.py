"""Downloaded records must restore locally without writing outside results/."""
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "fetch_artifacts", Path(__file__).resolve().parents[1] / "experiments/fetch_artifacts.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class RecordArchives(unittest.TestCase):
    def test_restore_historical_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "records.tar.gz"
            with tarfile.open(archive, "w:gz") as target:
                member = tarfile.TarInfo("results/run/response.json")
                member.size = 2
                target.addfile(member, io.BytesIO(b"{}"))
            fetch.restore_records(archive, root)
            self.assertEqual((root / "results/run/response.json").read_bytes(), b"{}")

    def test_reject_paths_and_links_before_writing_any_record(self):
        for name, symlink in [("../outside", False), ("README.md", False),
                              ("results/../../outside", False), ("results/link", True)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / "records.tar.gz"
                with tarfile.open(archive, "w:gz") as target:
                    valid = tarfile.TarInfo("results/run/response.json")
                    target.addfile(valid, io.BytesIO(b""))
                    member = tarfile.TarInfo(name)
                    if symlink:
                        member.type = tarfile.SYMTYPE
                        member.linkname = "../README.md"
                    target.addfile(member, None if symlink else io.BytesIO(b""))
                with self.assertRaises(ValueError):
                    fetch.restore_records(archive, root)
                self.assertFalse((root / "results").exists())


if __name__ == "__main__":
    unittest.main()
