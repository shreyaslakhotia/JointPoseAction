"""Regression: inherited active Conda state must not select base Python."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class EnvironmentSelectionTests(unittest.TestCase):
    def test_absolute_project_python_overrides_inherited_path_and_conda(self):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
        if not bash:
            self.skipTest("Bash unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, base = root / "project/bin", root / "base/bin"
            project.mkdir(parents=True)
            base.mkdir(parents=True)
            (project / "python").write_text('#!/bin/sh\n[ -z "${PYTHONHOME+x}" ] && [ -z "${PYTHONPATH+x}" ] || exit 77\nprintf "project-python-selected\\n"\n')
            (base / "python").write_text('#!/bin/sh\nprintf "wrong-base-python\\n"\nexit 99\n')
            for path in (project / "python", base / "python"):
                path.chmod(0o700)
            helper = Path(__file__).resolve().parents[1] / "scripts/cluster/use_environment.sh"
            env = os.environ.copy()
            env.update(JPA_ENV=project.parent.as_posix(), CONDA_PREFIX=project.parent.as_posix(),
                       PYTHONHOME="invalid-inherited-home", PYTHONPATH="invalid-inherited-path")
            command = 'PATH="$1:$PATH"; source "$2"'
            result = subprocess.run([bash, "-c", command, "test", base.as_posix(), helper.as_posix()],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("project-python-selected", result.stdout)
            self.assertNotIn("wrong-base-python", result.stdout)

    def test_missing_project_interpreter_is_refused(self):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
        if not bash:
            self.skipTest("Bash unavailable")
        helper = Path(__file__).resolve().parents[1] / "scripts/cluster/use_environment.sh"
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "JPA_ENV": Path(directory).as_posix()}
            result = subprocess.run([bash, "-c", 'source "$1"', "test", helper.as_posix()],
                                    env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing or not executable", result.stderr)


if __name__ == "__main__":
    unittest.main()
