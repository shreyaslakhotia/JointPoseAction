"""Relocated snapshots retain their dated manifests and strict hash checks."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout


HELPER = Path(__file__).resolve().parents[1] / "scripts/cluster/verify_checkout.py"
SPEC = importlib.util.spec_from_file_location("hygiene_verify_checkout", HELPER)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class HygienePathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.root_patch = patch.object(checker, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.original = "scripts/historical.py"
        self.final = "legacy_files/deprecated_scripts/scripts/historical.py"
        self.source = b"print('preserved')\n"
        self.expected = hashlib.sha256(self.source).hexdigest()
        target = self.root / self.final
        target.parent.mkdir(parents=True)
        target.write_bytes(self.source)
        manifest = self.root / "reproducibility/manifests/repository_hygiene.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"baseline_commit": "recorded", "files": [
            {"original_path": self.original, "final_path": self.final}]}))

    def test_relocated_snapshot_resolves_without_rewriting_old_path(self):
        mappings = checker.migration_paths()
        self.assertEqual(checker.resolve_preserved_path(self.original, mappings), self.final)
        self.assertEqual(checker.verify_preserved(self.original, self.expected, mappings), "exact")
        (self.root / self.final).write_bytes(b"changed\n")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            checker.verify_preserved(self.original, self.expected, mappings)

    def test_missing_mapping_fails_instead_of_guessing_legacy_location(self):
        with self.assertRaisesRegex(FileNotFoundError, "Missing"):
            checker.verify_preserved(self.original, self.expected, {})

    def test_crlf_uses_original_blob_only_before_renamed_path_exists(self):
        (self.root / self.final).write_bytes(self.source.replace(b"\n", b"\r\n"))
        absent = subprocess.CalledProcessError(128, ["git", "show"])
        with patch.object(checker, "git", side_effect=[absent, self.source]) as git:
            self.assertEqual(checker.verify_preserved(self.original, self.expected,
                                                     checker.migration_paths()), "crlf_only")
        self.assertEqual(git.call_args_list[0].args, ("show", "HEAD:" + self.final))
        self.assertEqual(git.call_args_list[1].args, ("show", "HEAD:" + self.original))

    def test_existing_wrong_new_blob_cannot_fall_back_to_old_blob(self):
        (self.root / self.final).write_bytes(self.source.replace(b"\n", b"\r\n"))
        with patch.object(checker, "git", return_value=b"changed\n") as git:
            with self.assertRaisesRegex(ValueError, "Committed hash mismatch"):
                checker.verify_preserved(self.original, self.expected, checker.migration_paths())
        git.assert_called_once_with("show", "HEAD:" + self.final)

    def test_syntax_checks_current_wrapper_instead_of_archived_original(self):
        current = self.root / self.original
        current.parent.mkdir(parents=True)
        current.write_text("def invalid syntax\n")
        for name in ("preservation_checksums.json", "small_artifacts.json"):
            (self.root / "reproducibility" / name).write_text('{"files": []}')
        output = io.StringIO()
        with patch.object(checker, "git", side_effect=[
                (self.original + "\0").encode(), b"baseline\n"]), redirect_stdout(output):
            self.assertTrue(checker.main())
        report = json.loads(output.getvalue())
        self.assertEqual(len(report["failures"]), 1)
        self.assertIn("Parse error: " + self.original, report["failures"][0])


if __name__ == "__main__":
    unittest.main()
