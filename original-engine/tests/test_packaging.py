"""Packaging, CI and bundled-task discovery: the CI workflow is valid YAML, pyproject.toml is complete, and the CLI finds (or
clearly fails to find) the bundled tasks however it is installed. No network: a non-editable install is simulated by copying the
package next to nothing else."""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CI = REPO / ".github" / "workflows" / "ci.yml"
HINT = "bundled tasks not found next to the nightshift package"
BROKEN_STEP = "      - name: Install numpy (optional: only the spiral-classifier task needs it)"      # the line GitHub rejected

_KEY_LINE = re.compile(r"^(?P<indent>\s*)(?:- )?(?P<key>[A-Za-z_][\w.-]*):(?:\s+(?P<value>\S.*))?$")
_NOT_PLAIN = ("'", '"', "|", ">", "[", "{", "&", "*", "!", "%", "@", "`", "#")


def plain_scalar_problems(text: str) -> list[str]:
    """Structural check for the mistake that made the workflow invalid, usable without a YAML library: a ``key: value`` line whose
    value is an unquoted plain scalar containing ``": "`` (YAML reads that as a nested mapping and refuses it), or a tab used
    for indentation. Block scalars (``run: |``) are skipped, bodies included."""
    problems: list[str] = []
    block_indent: int | None = None
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if block_indent is not None:
            if stripped and indent > block_indent:
                continue
            block_indent = None
        if "\t" in line[:indent + 1]:
            problems.append(f"line {number}: tab in the indentation")
        match = _KEY_LINE.match(line.rstrip())
        if not match or not match.group("value"):
            continue
        value = match.group("value").strip()
        if value[0] in "|>":
            block_indent = indent + (2 if line.lstrip().startswith("- ") else 0)
            continue
        if value.startswith(_NOT_PLAIN) or value.startswith("${{"):
            continue
        if ": " in value or value.endswith(":"):
            problems.append(f"line {number}: unquoted value of {match.group('key')!r} contains ': ' ({value[:60]!r})")
    return problems


class CiWorkflowTests(unittest.TestCase):
    def test_the_checker_flags_the_step_name_github_rejected(self):
        self.assertTrue(plain_scalar_problems(BROKEN_STEP), "the structural check must catch the original defect")
        self.assertFalse(plain_scalar_problems(BROKEN_STEP.replace("Install numpy (optional: only the spiral-classifier task needs it)",
                                                                    '"Install numpy (optional: only the spiral-classifier task needs it)"')))

    def test_the_checker_ignores_block_scalar_bodies_and_flow_values(self):
        text = "steps:\n  - run: |\n      echo a: b\n      python - <<'PY'\n      x = {'a': 1}\n      PY\n  - name: \"q: r\"\n    with:\n      v: [\"3.11\", \"3.12\"]\n"
        self.assertEqual(plain_scalar_problems(text), [])

    def test_no_line_of_the_workflow_has_an_unquoted_colon_space_value(self):
        self.assertEqual(plain_scalar_problems(CI.read_text(encoding="utf-8")), [])

    def test_the_workflow_parses_as_yaml_and_has_the_expected_shape(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML is not installed; the structural check above is what runs here")
        doc = yaml.safe_load(CI.read_text(encoding="utf-8"))
        self.assertEqual(doc["name"], "ci")
        for name, job in doc["jobs"].items():
            self.assertTrue(job["steps"], name)
            for step in job["steps"]:
                self.assertTrue(("run" in step) != ("uses" in step), (name, step))      # exactly one of the two
                self.assertIsInstance(step.get("name", ""), str)
        matrix = doc["jobs"]["test"]["strategy"]["matrix"]
        self.assertEqual(matrix["python-version"], ["3.11", "3.12", "3.13"])
        self.assertEqual(matrix["numpy"], ["with-numpy", "without-numpy"])

    def test_the_workflow_covers_integration_compileall_and_the_laptop_isolation_mode(self):
        text = CI.read_text(encoding="utf-8")
        self.assertIn('NIGHTSHIFT_INTEGRATION: "1"', text)
        self.assertRegex(text, r"python -m compileall -q (?:-x \S+ )?nightshift tests scripts")
        self.assertRegex(text, r"NIGHTSHIFT_ISOLATION: none\n(?:.*\n)*?\s+run: \|\n\s+python -m nightshift demo --no-serve")
        self.assertIn("python -m nightshift tasks", text)

    def test_the_compileall_step_leaves_no_bytecode_in_a_task_template(self):
        """Regression: `compileall -q nightshift tests scripts` byte-compiled tests/fixtures/mini_task/template, and every test that
        validates that task then failed with template_has_bytecode (11 tests, all of CI). The step runs BEFORE the tests."""
        text = CI.read_text(encoding="utf-8")
        match = re.search(r"run: (python -m compileall .*)", text)
        self.assertIsNotNone(match)
        argv = shlex.split(match.group(1))
        argv[0] = sys.executable
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ("nightshift/__init__.py", "tests/__init__.py", "scripts/tool.py", "tests/fixtures/mini_task/template/model.py",
                        "tests/fixtures/mini_task/trusted/grade.py"):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text("X = 1\n", encoding="utf-8")
            proc = subprocess.run(argv, cwd=root, capture_output=True, text=True, check=False, env={**os.environ, "PYTHONDONTWRITEBYTECODE": ""})
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertTrue((root / "nightshift" / "__pycache__").is_dir())          # it does compile the package ...
            self.assertTrue((root / "scripts" / "__pycache__").is_dir())
            self.assertEqual(list((root / "tests" / "fixtures").rglob("__pycache__")), [])      # ... but not the task fixture

    def test_a_skipped_task_fails_the_with_numpy_legs_so_a_typo_in_requires_cannot_hide(self):
        """`nightshift tasks` exits 0 on a SKIP by design; on the legs where numpy IS installed a SKIP means a task's calibration
        silently did not run, so the workflow step must turn it into a failure there (and only there)."""
        text = CI.read_text(encoding="utf-8")
        step = text[text.index("- name: \"Task validation"):text.index("  integration:")]
        self.assertIn("set -o pipefail", step)                               # a failing `nightshift tasks` must not hide in the pipe
        self.assertIn("python -m nightshift tasks | tee", step)
        self.assertIn('[ "${{ matrix.numpy }}" = "with-numpy" ] && grep -q \'^SKIP\'', step)
        self.assertIn("exit 1", step)

    def test_the_integration_flag_is_only_set_in_its_own_job(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML is not installed")
        jobs = yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"]
        self.assertNotIn("NIGHTSHIFT_INTEGRATION", str(jobs["test"]))
        self.assertIn("NIGHTSHIFT_INTEGRATION", str(jobs["integration"]))


class PyprojectTests(unittest.TestCase):
    def setUp(self) -> None:
        raw = (REPO / "pyproject.toml").read_bytes()
        self.text = raw.decode("utf-8")
        self.data = tomllib.loads(self.text)
        self.project = self.data["project"]

    def test_the_metadata_is_complete(self):
        p = self.project
        self.assertEqual(p["readme"], "README.md")
        self.assertEqual(p["requires-python"], ">=3.11")
        self.assertEqual(p["dependencies"], [])
        self.assertEqual(p["scripts"], {"nightshift": "nightshift.cli:main"})
        for key in ("name", "version", "description", "license", "keywords", "classifiers"):
            self.assertTrue(p.get(key), key)
        self.assertTrue(any(c.startswith("Programming Language :: Python :: 3.13") for c in p["classifiers"]))

    def test_urls_are_real_not_placeholders(self):
        for label, url in self.project.get("urls", {}).items():
            self.assertRegex(url, r"^https://(github\.com|gitlab\.com)/[\w.-]+/[\w.-]+$", label)
            self.assertNotRegex(url, r"(?i)example|your|todo|placeholder|<|>|xxx", label)
        self.assertNotRegex(self.text, r"(?i)\bTODO\b|example\.com|your-name")

    def test_the_tasks_are_left_out_of_the_wheel_on_purpose_and_the_file_says_why(self):
        self.assertEqual(self.data["tool"]["setuptools"]["packages"]["find"]["include"], ["nightshift*"])
        self.assertNotIn("tasks", str(self.data["tool"]["setuptools"].get("package-data", {})))
        comment = self.text[:self.text.index("[tool.setuptools.packages.find]")]
        self.assertIn("NOT part of the wheel, on purpose", comment)
        self.assertIn("template_has_bytecode", comment)
        self.assertIn("NIGHTSHIFT_TASKS_DIR", comment)


class TaskDiscoveryProcessTests(unittest.TestCase):
    """The real CLI in a child process."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        # what `pip install .` leaves behind: the package, no tasks/, no pyproject.toml, nothing above it
        cls.site = cls.tmp / "site-packages"
        shutil.copytree(REPO / "nightshift", cls.site / "nightshift", ignore=shutil.ignore_patterns("__pycache__"))
        cls.elsewhere = cls.tmp / "elsewhere"
        cls.elsewhere.mkdir()
        cls.empty = cls.tmp / "empty-tasks"
        cls.empty.mkdir()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    @staticmethod
    def env(pythonpath: Path, **extra: str) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in ("NIGHTSHIFT_TASKS_DIR", "NIGHTSHIFT_ISOLATION", "MODEL_API_KEY",
                                                                "META_API_KEY")}
        env["PYTHONPATH"] = str(pythonpath)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.update(extra)
        return env

    def cli(self, argv: list[str], *, cwd: Path, pythonpath: Path, timeout: float = 180, **env: str):
        return subprocess.run([sys.executable, "-m", "nightshift", *argv], cwd=cwd, env=self.env(pythonpath, **env),
                              capture_output=True, text=True, timeout=timeout, check=False)

    def test_tasks_works_from_another_directory_inside_the_checkout(self):
        proc = self.cli(["tasks"], cwd=REPO / "docs", pythonpath=REPO)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("law-discovery", proc.stdout)
        self.assertNotIn(HINT, proc.stderr)

    def test_a_task_name_resolves_from_a_subdirectory_of_the_checkout(self):
        proc = self.cli(["demo", "--task", "no-such-task"], cwd=REPO / "tests", pythonpath=REPO, timeout=60)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn(str(REPO / "tasks" / "no-such-task"), proc.stderr)       # it looked in the checkout's tasks/
        self.assertNotIn(HINT, proc.stderr)

    def test_an_empty_tasks_dir_gives_the_one_line_hint_and_a_usage_exit(self):
        for argv in (["tasks"], ["demo"], ["run", "--task", "law-discovery"]):
            with self.subTest(argv=argv):
                proc = self.cli(argv, cwd=REPO / "docs", pythonpath=REPO, timeout=60, NIGHTSHIFT_TASKS_DIR=str(self.empty))
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                lines = proc.stderr.strip().splitlines()
                self.assertEqual(len(lines), 1, proc.stderr)
                for part in (HINT, "run from a clone of the repository", "pip install -e .", "--task /path/to/task"):
                    self.assertIn(part, lines[0])
                self.assertFalse((REPO / "docs" / "runs").exists())          # nothing was created for the failed command

    def test_a_non_editable_install_without_tasks_says_so_in_every_command_that_needs_them(self):
        for argv in (["tasks"], ["demo", "--out", "rel-out"], ["run", "--task", "law-discovery", "--run-dir", "rel-run"]):
            with self.subTest(argv=argv):
                proc = self.cli(argv, cwd=self.elsewhere, pythonpath=self.site, timeout=60)
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                self.assertIn(HINT, proc.stderr)
        self.assertEqual(list(self.elsewhere.iterdir()), [])                 # no run directory appeared in the caller's cwd

    def test_the_tasks_dir_variable_makes_a_non_editable_install_work(self):
        proc = self.cli(["tasks"], cwd=self.elsewhere, pythonpath=self.site, NIGHTSHIFT_TASKS_DIR=str(REPO / "tasks"))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("law-discovery", proc.stdout)
        proc = self.cli(["demo", "--task", "no-such-task"], cwd=self.elsewhere, pythonpath=self.site, timeout=60,
                        NIGHTSHIFT_TASKS_DIR=str(REPO / "tasks"))
        self.assertEqual(proc.returncode, 2)
        self.assertIn(str(REPO / "tasks" / "no-such-task"), proc.stderr)      # the variable's directory was searched
        self.assertNotIn(HINT, proc.stderr)

    def test_an_invalid_isolation_value_is_refused_before_a_run_directory_exists(self):
        for argv in (["demo", "--out", "rel-out"], ["run", "--task", str(REPO / "tasks" / "law-discovery"), "--run-dir", "rel-run"]):
            with self.subTest(argv=argv):
                proc = self.cli(argv, cwd=self.elsewhere, pythonpath=REPO, timeout=60, NIGHTSHIFT_ISOLATION="fortress")
                self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
                for word in ("NIGHTSHIFT_ISOLATION", "fortress", "userns", "unshare-net", "none", "auto"):
                    self.assertIn(word, proc.stderr)
                self.assertEqual(list(self.elsewhere.iterdir()), [])
        proc = self.cli(["doctor"], cwd=self.elsewhere, pythonpath=REPO, timeout=60, NIGHTSHIFT_ISOLATION="fortress")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("fortress", proc.stdout)
        self.assertNotIn("probe failed", proc.stdout)


if __name__ == "__main__":
    unittest.main()
