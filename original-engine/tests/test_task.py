import contextlib
import dataclasses
import io
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nightshift import task as task_mod
from nightshift.task import (CALIBRATION_MIN_SEEDS, NoiseEstimate, calibration_notes, conditions_hash, estimate_baseline_noise,
                             list_tasks, load_task, missing_requirements, recommended_sigma_floor, task_hash, trusted_hash,
                             validate_task)

FIXTURE = Path(__file__).parent / "fixtures" / "mini_task"
HEX12 = re.compile(r"[0-9a-f]{12}")


def copy_fixture(dest: Path) -> Path:
    shutil.copytree(FIXTURE, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dest


class TaskTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def fixture(self, name: str = "t") -> Path:
        return copy_fixture(self.dir / name)

    def edit_toml(self, task_dir: Path, old: str, new: str) -> None:
        p = task_dir / "task.toml"
        text = p.read_text()
        self.assertIn(old, text)
        p.write_text(text.replace(old, new))


class LoadTaskTests(TaskTestCase):
    def test_loads_fixture_into_taskspec(self):
        t = load_task(FIXTURE)
        self.assertEqual(t.name, "mini-curve")
        self.assertEqual(t.objective, "minimize")
        self.assertTrue(t.minimize)
        self.assertEqual((t.primary_metric, t.train_metric), ("heldout_rmse", "train_rmse"))
        self.assertEqual(t.task_dir, FIXTURE)
        self.assertEqual(t.trusted_dir, FIXTURE / "trusted")
        self.assertEqual(t.template_dir, FIXTURE / "template")
        self.assertIsNone(t.scripted_dir)
        self.assertEqual((t.editable, t.scratch), (["model.py"], ["outputs/**"]))
        self.assertEqual(t.run_cmd, ["python3", "run.py"])
        self.assertEqual((t.seeds, t.replication_seeds), ([0, 1, 2], [100, 101, 102, 103, 104]))
        self.assertEqual((t.noise_margin, t.sigma_floor, t.sealed, t.stall_sec), (0.02, 0.0, True, 10.0))
        self.assertEqual(t.requires, [])
        self.assertIn("{task_dir}/trusted/grade.py", t.grade_cmd[1])

    def test_accepts_toml_path_and_picks_up_scripted_dir(self):
        d = self.fixture()
        (d / "scripted").mkdir()
        t = load_task(d / "task.toml")
        self.assertEqual(t.task_dir, d)
        self.assertEqual(t.scripted_dir, d / "scripted")

    def test_maximize_and_requires(self):
        d = self.fixture()
        self.edit_toml(d, 'objective = "minimize"', 'objective = "maximize"')
        self.edit_toml(d, "requires = []", 'requires = ["numpy"]')
        t = load_task(d)
        self.assertFalse(t.minimize)
        self.assertEqual(t.requires, ["numpy"])

    def test_unknown_keys_are_kept_in_extra_not_dropped(self):
        d = self.fixture()
        self.edit_toml(d, "noise_margin = 0.02", "noise_margn = 0.02\nnoise_margin = 0.02")
        self.assertEqual(load_task(d).extra, {"grade.noise_margn": 0.02})

    def test_precise_errors(self):
        cases = [
            ('objective = "minimize"', 'objective = "sideways"', "objective"),
            ("seeds = [0, 1, 2]", 'seeds = [0, "a", 2]', "grade.seeds"),
            ("seeds = [0, 1, 2]", "seeds = [0, true, 2]", "grade.seeds"),
            ("seeds = [0, 1, 2]", "seeds = []", "grade.seeds"),
            ("noise_margin = 0.02", 'noise_margin = "big"', "grade.noise_margin"),
            ("noise_margin = 0.02", "noise_margin = -1", "grade.noise_margin"),
            ("sealed = true", 'sealed = "yes"', "grade.sealed"),
            ('cmd = ["python3", "run.py"]', "cmd = []", "run.cmd"),
            ('editable = ["model.py"]', 'editable = ["../model.py"]', "workspace.editable"),
            ('editable = ["model.py"]', 'editable = ["/etc/passwd"]', "workspace.editable"),
            ("timeout_sec = 20\n\n[grade]", "timeout_sec = 0\n\n[grade]", "run.timeout_sec"),
            ('primary_metric = "heldout_rmse"', "", "primary_metric"),
        ]
        for old, new, needle in cases:
            with self.subTest(needle=needle, new=new):
                d = self.fixture(f"case{abs(hash((old, new)))}")
                self.edit_toml(d, old, new)
                with self.assertRaises(ValueError) as cm:
                    load_task(d)
                self.assertIn(needle, str(cm.exception))
                self.assertIn("task.toml", str(cm.exception))

    def test_bad_toml_syntax_and_missing_file_and_missing_table(self):
        d = self.fixture()
        (d / "task.toml").write_text("name = \n")
        with self.assertRaisesRegex(ValueError, "invalid TOML"):
            load_task(d)
        with self.assertRaisesRegex(ValueError, "cannot read"):
            load_task(self.dir / "nowhere")
        (d / "task.toml").write_text('name = "x"\nbrief = "b"\nobjective = "minimize"\nprimary_metric = "m"\n')
        with self.assertRaisesRegex(ValueError, r"\[workspace\] is required"):
            load_task(d)
        (d / "task.toml").write_bytes(b"name = '\xff\xfe'\n")
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            load_task(d)

    def test_list_tasks_sorted_skips_non_tasks_and_reports_errors(self):
        root = self.dir / "tasks"
        copy_fixture(root / "b-task")
        copy_fixture(root / "a-task")
        (root / "not-a-task").mkdir()
        (root / "stray.txt").write_text("x")
        self.assertEqual([t.task_dir.name for t in list_tasks(root)], ["a-task", "b-task"])
        (root / "a-task" / "task.toml").write_text("name = ")
        with self.assertRaises(ValueError):
            list_tasks(root)
        seen = []
        got = list_tasks(root, on_error=lambda p, e: seen.append((p.name, str(e))))
        self.assertEqual([t.task_dir.name for t in got], ["b-task"])
        self.assertEqual(seen[0][0], "a-task")
        self.assertEqual(list_tasks(self.dir / "missing"), [])


class ValidateTaskTests(TaskTestCase):
    def test_fixture_is_clean(self):
        self.assertEqual(validate_task(load_task(FIXTURE)), [])

    def test_missing_directories(self):
        d = self.fixture()
        shutil.rmtree(d / "template")
        shutil.rmtree(d / "trusted")
        problems = validate_task(load_task(d))
        self.assertTrue(any(p.startswith("template_dir_missing") for p in problems), problems)
        self.assertTrue(any(p.startswith("trusted_dir_missing") for p in problems), problems)
        self.assertTrue(any(p.startswith("grader_missing") for p in problems), problems)

    def test_editable_file_absent_from_template(self):
        d = self.fixture()
        (d / "template" / "model.py").unlink()
        self.assertTrue(any(p.startswith("editable_missing: model.py") for p in validate_task(load_task(d))))

    def test_wildcard_editable_pattern_is_not_required_to_match(self):
        d = self.fixture()
        self.edit_toml(d, 'editable = ["model.py"]', 'editable = ["model.py", "extra/*.py"]')
        self.assertEqual(validate_task(load_task(d)), [])

    def test_grader_path_missing_and_outside_trusted(self):
        d = self.fixture()
        (d / "trusted" / "grade.py").unlink()
        self.assertTrue(any(p.startswith("grader_missing") for p in validate_task(load_task(d))))
        d2 = self.fixture("t2")
        shutil.move(d2 / "trusted" / "grade.py", d2 / "template" / "grade.py")
        self.edit_toml(d2, "{task_dir}/trusted/grade.py", "{task_dir}/template/grade.py")
        self.assertTrue(any(p.startswith("grader_outside_trusted") for p in validate_task(load_task(d2))))

    def test_grade_cmd_must_pass_workspace_seed_and_split(self):
        t = load_task(FIXTURE)
        problems = validate_task(dataclasses.replace(t, grade_cmd=["python3", "{task_dir}/trusted/grade.py"]))
        self.assertEqual(sum(p.startswith("grade_cmd_missing_token") for p in problems), 3, problems)

    def test_seed_rules(self):
        t = load_task(FIXTURE)
        cases = {
            "too_few_seeds": dict(seeds=[0, 1]),
            "too_few_replication_seeds": dict(replication_seeds=[100, 101]),
            "seeds_not_disjoint": dict(replication_seeds=[2, 101, 102]),
            "duplicate_seeds": dict(seeds=[0, 0, 1]),
        }
        for code, change in cases.items():
            with self.subTest(code=code):
                problems = validate_task(dataclasses.replace(t, **change))
                self.assertTrue(any(p.startswith(code) for p in problems), problems)

    def test_run_script_must_exist_in_the_template(self):
        d = self.fixture()
        (d / "template" / "run.py").unlink()
        self.assertTrue(any(p.startswith("run_script_missing: run.py") for p in validate_task(load_task(d))))

    def test_train_metric_must_differ_from_the_primary_metric(self):
        t = dataclasses.replace(load_task(FIXTURE), train_metric="heldout_rmse")
        self.assertTrue(any(p.startswith("train_metric_equals_primary") for p in validate_task(t)))

    def test_timeouts_must_be_positive(self):
        t = dataclasses.replace(load_task(FIXTURE), run_timeout_sec=0, stall_sec=-1)
        self.assertEqual(sum(p.startswith("bad_timeout") for p in validate_task(t)), 2)

    def test_symlinks_in_the_template_are_a_problem(self):
        d = self.fixture()
        (d / "template" / "link.py").symlink_to("/etc/hostname")
        self.assertTrue(any(p.startswith("template_symlink") for p in validate_task(load_task(d))))

    def test_template_files_that_no_workspace_could_copy_are_a_problem(self):
        import os
        d = self.fixture()
        self.assertEqual(validate_task(load_task(d)), [])
        (d / "template" / "my data.csv").write_text("x")                  # space: not in the safe name alphabet
        os.link(d / "template" / "run.py", d / "template" / "run_copy.py")    # hard link
        problems = [p for p in validate_task(load_task(d)) if p.startswith("template_special")]
        self.assertEqual(len(problems), 3, problems)                        # my data.csv, run.py, run_copy.py
        self.assertTrue(any("my data.csv" in p for p in problems))

    def test_noise_margin_must_be_positive_when_sealed_only(self):
        t = load_task(FIXTURE)
        self.assertTrue(any(p.startswith("noise_margin_not_positive")
                            for p in validate_task(dataclasses.replace(t, noise_margin=0.0))))
        self.assertFalse(any(p.startswith("noise_margin_not_positive")
                             for p in validate_task(dataclasses.replace(t, noise_margin=0.0, sealed=False))))

    def test_brief_must_not_mention_seeds_or_sealed(self):
        t = load_task(FIXTURE)
        for word in ("seeds", "Sealed", "seed"):
            problems = validate_task(dataclasses.replace(t, brief=f"Improve it. Note the {word} used."))
            self.assertTrue(any(p.startswith("brief_mentions_seed_or_sealed") for p in problems), word)

    def test_unknown_toml_key_is_a_problem(self):
        d = self.fixture()
        self.edit_toml(d, "noise_margin = 0.02", "noise_margn = 0.02\nnoise_margin = 0.02")
        self.assertTrue(any(p.startswith("unknown_key: grade.noise_margn") for p in validate_task(load_task(d))))

    def test_template_bytecode_is_a_problem(self):
        d = self.fixture()
        (d / "template" / "__pycache__").mkdir()
        (d / "template" / "__pycache__" / "model.cpython-313.pyc").write_bytes(b"\0")
        self.assertTrue(any(p.startswith("template_has_bytecode") for p in validate_task(load_task(d))))

    def test_each_forbidden_call_in_trusted_python_is_reported(self):
        snippets = {
            "pickle": "import pickle\n",
            "allow_pickle=True": "np.load(p, allow_pickle=True)\n",
            "eval(": "x = eval('1')\n",
            "exec(": "exec('x = 1')\n",
            "yaml.load": "yaml.load(s)\n",
            "importlib": "import importlib\n",
        }
        for label, code in snippets.items():
            with self.subTest(label=label):
                d = self.fixture(f"f{abs(hash(label))}")
                (d / "trusted" / "extra.py").write_text(code)
                problems = validate_task(load_task(d))
                self.assertIn(f"forbidden_call: trusted/extra.py uses {label}", problems)

    def test_safe_lookalikes_and_comments_are_not_flagged(self):
        d = self.fixture()
        (d / "trusted" / "extra.py").write_text(
            "import ast, yaml\n# never use pickle or eval( here\n"
            "a = ast.literal_eval('1')\nb = yaml.safe_load('1')\nc = 'retrieval(x)'\nd = medieval(2)\n")
        self.assertEqual(validate_task(load_task(d)), [])

    def test_docstrings_are_prose_but_real_calls_next_to_them_still_count(self):
        d = self.fixture()
        (d / "trusted" / "extra.py").write_text(
            '"""Parses JSON only: never pickle, eval(), exec() or yaml.load."""\n\n\n'
            'class Reader:\n    """No importlib here."""\n\n    def read(self, s):\n        """eval( is banned."""\n'
            '        return s\n')
        self.assertEqual(validate_task(load_task(d)), [])
        (d / "trusted" / "extra.py").write_text('"""doc"""\nimport pickle\n')
        self.assertIn("forbidden_call: trusted/extra.py uses pickle", validate_task(load_task(d)))

    def test_unparseable_trusted_python_is_still_scanned(self):
        d = self.fixture()
        (d / "trusted" / "legacy.py").write_text("print 'x'\nx = eval('1')   # py2 style\n")
        self.assertIn("forbidden_call: trusted/legacy.py uses eval(", validate_task(load_task(d)))

    def test_pickle_files_in_trusted_are_reported(self):
        d = self.fixture()
        (d / "trusted" / "labels.pkl").write_bytes(b"\x80\x04N.")
        self.assertTrue(any(p.startswith("trusted_forbidden_file") for p in validate_task(load_task(d))))


class CalibrationTests(TaskTestCase):
    def test_calibration_passes_on_the_fixture(self):
        self.assertEqual(validate_task(load_task(FIXTURE), calibrate=True, notes=[]), [])

    def test_vacuous_seeds_are_reported(self):
        d = self.fixture()
        # a grader whose score ignores the seed: per-seed scores are all equal
        (d / "trusted" / "grade.py").write_text(
            "import json, sys\n"
            "split = sys.argv[sys.argv.index('--split') + 1]\n"
            "print(json.dumps({'valid': True, 'score': 0.5, 'split': split, 'n_eval': 10, 'metrics': {}, 'notes': ''}))\n")
        problems = validate_task(load_task(d), calibrate=True)
        self.assertTrue(any(p.startswith("seeds_vacuous") for p in problems), problems)

    def test_calibration_failure_is_reported_not_raised(self):
        d = self.fixture()
        (d / "template" / "model.py").write_text("raise RuntimeError('broken template')\n")
        problems = validate_task(load_task(d), calibrate=True)
        self.assertTrue(any(p.startswith("calibration_failed") and "crash" in p for p in problems), problems)

    def test_structural_problems_skip_calibration(self):
        t = dataclasses.replace(load_task(FIXTURE), seeds=[0, 1])
        problems = validate_task(t, calibrate=True)
        self.assertTrue(all(not p.startswith(("seeds_vacuous", "calibration_failed")) for p in problems))

    def test_missing_requirements_skip_calibration(self):
        t = dataclasses.replace(load_task(FIXTURE), requires=["no_such_module_xyz"])
        problems = validate_task(t, calibrate=True)
        self.assertEqual(problems, ["missing_requirements: no_such_module_xyz (calibration skipped)"])


class HashTests(TaskTestCase):
    def test_trusted_hash_is_12_hex_and_content_sensitive(self):
        d = self.fixture()
        t = load_task(d)
        h = trusted_hash(t)
        self.assertRegex(h, HEX12)
        self.assertEqual(h, trusted_hash(load_task(d)))
        (d / "trusted" / "labels.csv").write_text("changed\n")
        self.assertNotEqual(h, trusted_hash(t))

    def test_trusted_hash_sees_new_files_and_renames(self):
        d = self.fixture()
        t = load_task(d)
        h = trusted_hash(t)
        (d / "trusted" / "new.txt").write_text("x")
        h2 = trusted_hash(t)
        self.assertNotEqual(h, h2)
        (d / "trusted" / "new.txt").rename(d / "trusted" / "renamed.txt")
        self.assertNotEqual(h2, trusted_hash(t))

    def test_trusted_hash_ignores_pycache_and_pyc(self):
        d = self.fixture()
        t = load_task(d)
        h = trusted_hash(t)
        (d / "trusted" / "__pycache__").mkdir()
        (d / "trusted" / "__pycache__" / "grade.cpython-313.pyc").write_bytes(b"\x01\x02")
        (d / "trusted" / "stray.pyc").write_bytes(b"\x03")
        self.assertEqual(h, trusted_hash(t))

    def test_trusted_hash_does_not_hang_on_a_planted_fifo_and_notices_it(self):
        import os
        import threading
        d = self.fixture()
        t = load_task(d)
        h = trusted_hash(t)
        os.mkfifo(d / "trusted" / "trap")
        got = {}
        worker = threading.Thread(target=lambda: got.setdefault("h", trusted_hash(t)), daemon=True)
        worker.start()
        worker.join(5)
        self.assertIn("h", got, "trusted_hash blocked on a FIFO (open(2) without O_NONBLOCK)")
        self.assertNotEqual(got["h"], h, "a new special file in trusted/ is tampering")

    def test_trusted_hash_sees_a_symlink_to_a_directory(self):
        import os
        d = self.fixture()
        t = load_task(d)
        h = trusted_hash(t)
        os.symlink(self.dir, d / "trusted" / "elsewhere")
        h2 = trusted_hash(t)
        self.assertNotEqual(h, h2, "os.walk lists directory symlinks as directories and never reports them")
        os.unlink(d / "trusted" / "elsewhere")
        os.symlink(d, d / "trusted" / "elsewhere")
        self.assertNotEqual(h2, trusted_hash(t))

    def test_trusted_hash_is_identical_for_identical_trees_in_different_places(self):
        a, b = self.fixture("a"), self.fixture("b")
        self.assertEqual(trusted_hash(load_task(a)), trusted_hash(load_task(b)))

    def test_task_hash_tracks_toml_template_trusted_and_scripted(self):
        d = self.fixture()
        t = load_task(d)
        base = task_hash(t)
        self.assertRegex(base, HEX12)
        (d / "template" / "model.py").write_text("# edited\n")
        h_template = task_hash(t)
        self.assertNotEqual(base, h_template)
        (d / "trusted" / "labels.csv").write_text("x\n")
        h_trusted = task_hash(t)
        self.assertNotEqual(h_template, h_trusted)
        self.edit_toml(d, "noise_margin = 0.02", "noise_margin = 0.03")
        h_toml = task_hash(t)
        self.assertNotEqual(h_trusted, h_toml)
        (d / "scripted").mkdir()
        (d / "scripted" / "manifest.json").write_text("{}")
        self.assertNotEqual(h_toml, task_hash(load_task(d)))

    def test_conditions_hash_depends_on_seeds_split_cmd_and_grader(self):
        t = load_task(FIXTURE)
        base = conditions_hash(t, [0, 1, 2])
        self.assertRegex(base, HEX12)
        self.assertEqual(base, conditions_hash(t, (0, 1, 2), "dev"))
        self.assertNotEqual(base, conditions_hash(t, [0, 1, 2], "sealed"))
        self.assertNotEqual(base, conditions_hash(t, [0, 1, 3]))
        self.assertNotEqual(base, conditions_hash(t, [0, 1]))
        self.assertNotEqual(base, conditions_hash(dataclasses.replace(t, run_cmd=["python3", "other.py"]), [0, 1, 2]))
        self.assertNotEqual(base, conditions_hash(t, [0, 1, 2], grader_hash="f" * 12))
        self.assertEqual(base, conditions_hash(t, [0, 1, 2], grader_hash=trusted_hash(t)))

    def test_missing_requirements(self):
        t = dataclasses.replace(load_task(FIXTURE), requires=["json", "no_such_module_xyz", "os.path", "also_missing.sub"])
        self.assertEqual(missing_requirements(t), ["no_such_module_xyz", "also_missing.sub"])
        self.assertEqual(missing_requirements(dataclasses.replace(t, requires=[])), [])


class CalibrationNoiseTests(TaskTestCase):
    """validate_task(calibrate=True) estimates the baseline per-seed std over many seeds and advises sigma_floor."""

    def est(self, std: float, split: str = "dev", n: int = 20) -> NoiseEstimate:
        return NoiseEstimate(split=split, n=n, mean=1.0, std=std, seeds=tuple(range(n)))

    def test_the_recommended_floor_is_the_dev_std_to_two_significant_digits(self):
        self.assertEqual(recommended_sigma_floor(0.03634), 0.036)
        self.assertEqual(recommended_sigma_floor(0.0386), 0.039)
        self.assertEqual(recommended_sigma_floor(0.0), 0.0)
        t = dataclasses.replace(load_task(FIXTURE), sigma_floor=0.036, noise_margin=0.035)
        line = calibration_notes(t, [self.est(0.0363), self.est(0.0380, "sealed")])[0]
        self.assertIn("Recommended sigma_floor = 0.036", line)
        self.assertIn("dev 0.0363, sealed 0.0380", line)
        self.assertIn("over 20 seeds", line)

    def test_a_floor_well_below_the_measured_std_is_warned_about(self):
        t = dataclasses.replace(load_task(FIXTURE), sigma_floor=0.0, noise_margin=0.035)
        notes = calibration_notes(t, [self.est(0.036)])
        warning = next(n for n in notes if n.startswith("warning") and "sigma_floor" in n)
        self.assertIn("noise prior is inert", warning)
        self.assertIn("t-test", warning)
        self.assertIn("Set sigma_floor = 0.036", warning)

    def test_the_old_shipped_floor_would_have_been_flagged(self):
        t = dataclasses.replace(load_task(FIXTURE), sigma_floor=0.02, noise_margin=0.035)      # law-discovery before the fix
        self.assertTrue(any(n.startswith("warning") and "sigma_floor 0.02 " in n for n in calibration_notes(t, [self.est(0.0385)])))

    def test_a_calibrated_floor_and_margin_give_no_warning(self):
        t = dataclasses.replace(load_task(FIXTURE), sigma_floor=0.036, noise_margin=0.035)
        notes = calibration_notes(t, [self.est(0.0385), self.est(0.0401, "sealed")])
        self.assertEqual(len(notes), 1)
        self.assertFalse(any(n.startswith("warning") for n in notes))

    def test_a_noise_margin_well_below_the_std_is_warned_about_separately(self):
        t = dataclasses.replace(load_task(FIXTURE), sigma_floor=0.036, noise_margin=0.005)
        notes = calibration_notes(t, [self.est(0.036)])
        self.assertEqual([n for n in notes if "noise_margin" in n and n.startswith("warning")].__len__(), 1)
        self.assertFalse(any(n.startswith("warning") and "sigma_floor" in n for n in notes))

    def test_no_estimate_no_notes(self):
        self.assertEqual(calibration_notes(load_task(FIXTURE), []), [])

    def test_the_estimate_uses_at_least_twelve_seeds_per_split_and_never_reuses_a_task_seed(self):
        t = load_task(FIXTURE)
        estimates = estimate_baseline_noise(t, n_seeds=3)          # asking for fewer is raised to the minimum
        self.assertEqual([e.split for e in estimates], ["dev", "sealed"])
        for e, own in zip(estimates, (t.seeds, t.replication_seeds)):
            self.assertGreaterEqual(e.n, CALIBRATION_MIN_SEEDS)
            self.assertEqual(len(e.seeds), len(set(e.seeds)))
            self.assertEqual(list(e.seeds[:len(own)]), own, "the task's own seeds come first")
            extra = set(e.seeds[len(own):])
            self.assertTrue(extra.isdisjoint(t.seeds) and extra.isdisjoint(t.replication_seeds))
            self.assertGreater(e.std, 0)

    def test_validate_task_collects_the_notes_and_the_task_stays_valid(self):
        notes: list[str] = []
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(validate_task(load_task(FIXTURE), calibrate=True, notes=notes), [])
        self.assertEqual(out.getvalue(), "", "notes given: nothing is printed")
        self.assertTrue(notes[0].startswith("calibration [mini-curve]: baseline per-seed std over 20 seeds"))
        self.assertTrue(any(n.startswith("warning") and "sigma_floor 0 is well below" in n for n in notes), notes)

    def test_without_a_notes_list_the_notes_are_printed_for_nightshift_tasks(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(validate_task(load_task(FIXTURE), calibrate=True), [])
        self.assertIn("Recommended sigma_floor", out.getvalue())

    def test_no_calibration_no_estimate(self):
        with mock.patch.object(task_mod, "estimate_baseline_noise", side_effect=AssertionError("must not run")):
            self.assertEqual(validate_task(load_task(FIXTURE)), [])

    def test_missing_requirements_skip_calibration_with_a_clear_message(self):
        t = dataclasses.replace(load_task(FIXTURE), requires=["no_such_module_xyz"])
        notes: list[str] = []
        with mock.patch.object(task_mod, "estimate_baseline_noise", side_effect=AssertionError("must not run")):
            problems = validate_task(t, calibrate=True, notes=notes)
        self.assertTrue(any(p.startswith("missing_requirements: no_such_module_xyz") and "calibration skipped" in p for p in problems))
        self.assertEqual(notes, [])

    def test_a_failing_estimate_is_a_note_not_a_problem(self):
        notes: list[str] = []
        with mock.patch.object(task_mod, "estimate_baseline_noise", side_effect=RuntimeError("dev grading of the template failed")):
            problems = validate_task(load_task(FIXTURE), calibrate=True, notes=notes)
        self.assertEqual(problems, [])
        self.assertEqual(len(notes), 1)
        self.assertIn("the baseline noise estimate was skipped (RuntimeError: dev grading of the template failed)", notes[0])

    def test_structural_problems_skip_the_estimate_too(self):
        d = self.fixture()
        self.edit_toml(d, "seeds = [0, 1, 2]\n", "seeds = [0, 1]\n")
        notes: list[str] = []
        problems = validate_task(load_task(d), calibrate=True, notes=notes)
        self.assertTrue(any(p.startswith("too_few_seeds") for p in problems))
        self.assertEqual(notes, [])


if __name__ == "__main__":
    unittest.main()
