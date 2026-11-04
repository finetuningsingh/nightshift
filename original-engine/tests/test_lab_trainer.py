"""GlimmerTrainer and RentalModelServer against a FakeProvider and a fake ssh whose "rental" is the local machine.

The REAL hostkit scripts run (upload, stage.sh, bootstrap.sh with its step markers pre-set, run_job.sh, serve.sh, glimmer-axolotl-train's
registration); only the heavy things are stubs: the Axolotl interpreter writes a tiny adapter and vLLM is a tiny HTTP server.
"""

import hashlib
import json
import os
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path

from nightshift.lab import gpu_remote as gr
from nightshift.lab import trainer as tr
from nightshift.lab.contracts import (
    AccessDenied,
    BaseModelRef,
    BudgetExhausted,
    BudgetSpec,
    CreateRejected,
    FileRef,
    GenerationParams,
    GpuError,
    JobStatus,
    LabLayout,
    Phase,
    ScoreRecord,
    ScoreRequest,
    ScorerError,
    ServeSpec,
    Shape,
    Split,
    Trainer,
    TrainerError,
    TrainJob,
    TrainResult,
    adapter_ref_from_dir,
    read_record,
    verify_adapter_dir,
)
from nightshift.lab.gpu_rentals import RentalManager, alive_watchdogs
from nightshift.lab.model_server import RemoteLayout, RentalModelServer, adapter_model_name
from tests.support.lab_fake_gpu import (
    FakeClock,
    FakeMeter,
    FakeProvider,
    FakeScorer,
    FakeSsh,
    FakeWatchdogLauncher,
    fake_key_factory,
    install_fake_axolotl,
    install_fake_vllm,
    make_tar_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
HOSTKIT = ROOT / "nightshift" / "lab" / "hostkit"
BASE = BaseModelRef("meta-models/Muse-Glimmer-30B", "a4e59da52a7bc87ae7251dd5545c0dd437c44b68")
SHAPE = Shape("a100", 1, 12, 300)
GOOD_RECIPE = (
    b'echo \'{"key":"k1","model":"muse-glimmer","answer":"a","reasoning":null,"finish_reason":"stop"}\' > "$GLIMMER_OUT/glimmer_samples.jsonl"\n'
    b'echo \'{"messages":[{"role":"user","content":"p1"},{"role":"assistant","content":"a"}]}\' > sft.jsonl\nglimmer-axolotl-train config.yaml --data sft.jsonl\n'
)  # a recipe whose one target is a recorded sample of the local model on the approved prompt p1


class RealtimeClock:
    """Real time plus an offset; ``jump`` makes the budget clock run ahead of the machine for the ticker tests."""

    def __init__(self):
        self.offset = 0.0

    def __call__(self) -> float:
        return time.time() + self.offset


class Harness(unittest.TestCase):
    budget = BudgetSpec(researcher_tokens=10**6, gpu_minutes=120.0, dollars=50.0, wall_seconds=7200.0)
    trainer_kw: dict = {}

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name).resolve()
        self.layout = LabLayout(self.tmp / "lab")
        self.clock = RealtimeClock()
        self.fs = FakeSsh(self.tmp / "fake")
        self.fs.__enter__()
        self.addCleanup(self.fs.__exit__)
        # the "rental": directories on this machine
        self.opt, self.work, self.models = (self.tmp / "remote" / d for d in ("opt", "work", "models"))
        for d in (self.opt, self.work, self.models):
            d.mkdir(parents=True)
        for step in ("machine", "uv", "axolotl_env", "vllm_env", "verify_env", "weights"):
            (self.opt / f".bootstrap.{step}.done").write_text("")
        self.venv_axolotl, self.venv_vllm = self.tmp / "remote" / "venv-axolotl", self.tmp / "remote" / "venv-vllm"
        install_fake_axolotl(self.venv_axolotl)
        install_fake_vllm(self.venv_vllm)
        self.addCleanup(self._kill_stragglers)
        self.vllm_port = gr.free_loopback_port()
        self.remote_layout = RemoteLayout(
            str(self.opt), str(self.work), str(self.models),
            (("AXOLOTL_VENV", str(self.venv_axolotl)), ("VLLM_VENV", str(self.venv_vllm)), ("LAB_SKIP_RELOAD_CHECK", "1"), ("LAB_PYTHON", sys.executable), ("PATH", f"{self.opt}/bin:{os.environ['PATH']}")),
        )  # fmt: skip
        # control-host side
        self.provider = FakeProvider(self.clock, rate=1.31)
        self.launcher = FakeWatchdogLauncher()
        self.addCleanup(self.launcher.close)
        self.manager = RentalManager(
            self.tmp / "lab" / "gpu" / "rentals.json", self.provider, clock=self.clock, launcher=self.launcher, key_factory=fake_key_factory,
            sleep=lambda s: time.sleep(0.01), poll_seconds=0.02, stopper=self.launcher.stop,
        )  # fmt: skip
        self.manager.init(100.0, "box", "test")
        self.scorer = FakeScorer(self.layout, self.clock)
        self.generators = []
        self.meter = FakeMeter(self.budget)
        # public data and a pinned IFEvalG stand-in
        for p, text in ((self.layout.train_data("a-evo-1"), '{"key": "k1", "prompt": "p1", "ground_truth": "g"}\n'), (self.layout.dev_data("a-evo-1"), '{"key": "d1", "prompt": "q1"}\n')):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        self.stage_src = self.tmp / "stage_src"
        self.stage_src.mkdir()
        (self.stage_src / "instructions.py").write_text("INSTRUCTIONS = 1\n")
        (self.stage_src / "rewards.py").write_text("def preflight():\n    pass\ndef verify_constraints(r, g):\n    return [True]\n")
        self.stage_files = {"ifevalg/instructions.py": self.stage_src / "instructions.py", "rewards.py": self.stage_src / "rewards.py"}
        kw = dict(
            clock=self.clock, remote_layout=self.remote_layout, stage_files=self.stage_files, pull_margin_s=57.0, eval_reserve_s=0.0, poll_seconds=0.05,
            tick_interval=0.1, sleep=lambda s: time.sleep(0.02), vllm_port=self.vllm_port, serve_timeout_s=30.0, bootstrap_timeout_s=120.0, min_job_seconds=1.0,
        )  # fmt: skip
        kw.update(self.trainer_kw)
        self.trainer = tr.GlimmerTrainer(self.manager, SHAPE, self.scorer, self.gen_factory, self.layout, HOSTKIT, BASE, **kw)

    def _kill_stragglers(self):
        import subprocess

        subprocess.run(["pkill", "-f", str(self.tmp)], capture_output=True)  # fake vLLM servers and ssh forwards started under this temp dir

    def gen_factory(self, endpoint):
        self.generators.append(endpoint)
        return object()

    def make_job(self, recipe_files=None, job_id="j-0001", trial="t-0001", max_minutes=60.0, entrypoint=("bash", "run.sh")) -> TrainJob:
        files = recipe_files if recipe_files is not None else {"run.sh": GOOD_RECIPE, "config.yaml": b"a: 1\n"}
        rdir = self.layout.trial_recipe_dir(trial)
        rdir.mkdir(parents=True, exist_ok=True)
        tmp_tar = rdir / "tmp.tar.gz"
        with tarfile.open(tmp_tar, "w:gz") as t:
            for name, data in files.items():
                ti = tarfile.TarInfo(name)
                ti.size, ti.mode = len(data), 0o755
                import io

                t.addfile(ti, io.BytesIO(data))
        sha = hashlib.sha256(tmp_tar.read_bytes()).hexdigest()
        final = rdir / f"{sha}.tar.gz"
        tmp_tar.rename(final)
        req = ScoreRequest(
            "sc-0001", Phase.SEARCH, "a-evo-1", Split.DEV, trial, None, BASE, None, (0, 1), GenerationParams.recommended(2000), "sv1-fakefakefakefake", None, None
        )  # fmt: skip
        return TrainJob(job_id, trial, FileRef(final.relative_to(self.layout.root).as_posix(), sha, final.stat().st_size), entrypoint, max_minutes, SHAPE, req)

    def run_job(self, job=None, meter=None):
        return self.trainer.run_job(job or self.make_job(), meter or self.meter)

    def assert_no_rental_left(self):
        self.assertEqual(self.provider.handles(), set(), "every path must delete the rental")
        self.assertTrue(all(r["state"] in ("deleted", "not_created") for r in self.manager.read()["rentals"]))


class HappyPath(Harness):
    def test_protocol(self):
        self.assertIsInstance(self.trainer, Trainer)

    def test_a_good_job_returns_a_verified_result(self):
        job = self.make_job()
        res = self.run_job(job)
        self.assertIsInstance(res, TrainResult)
        self.assertEqual(res.status, JobStatus.OK, res.failure)
        self.assertEqual((res.job_id, res.trial_id, res.adapter_id), ("j-0001", "t-0001", "ad-0001"))
        # the adapter: exactly the two behaviour-defining files plus our own adapter.json, hash-verified under the lab root
        adir = self.layout.adapter_dir("ad-0001")
        self.assertEqual(sorted(p.name for p in adir.iterdir()), ["adapter.json", "adapter_config.json", "adapter_model.safetensors"])
        verify_adapter_dir(res.adapter, root=self.layout.root)
        self.assertEqual(read_record(self.layout.adapter_json_path("ad-0001"), type(res.adapter)), res.adapter)
        # training data the contamination scan will read
        self.assertEqual(res.train_data.path, f"trials/t-0001/train_data/j-0001.jsonl")
        self.assertEqual(self.layout.file_ref(self.layout.trial_train_data_path("t-0001", "j-0001")), res.train_data)
        self.assertIn("assistant", (self.layout.root / res.train_data.path).read_text())
        # the dev score was written by the (fake) scorer through the real layout, for THIS adapter
        self.assertEqual(res.dev_score_id, "sc-0001")
        rec = read_record(self.layout.score_record_path("sc-0001"), ScoreRecord)
        self.assertEqual(rec.adapter_sha256, res.adapter.sha256)
        (request, _gen, model), = self.scorer.calls
        self.assertEqual(request.adapter, res.adapter)
        self.assertEqual(request.request_id, job.dev_request.request_id)
        self.assertEqual(model, adapter_model_name(res.adapter))
        self.assertIsNone(request.arm)
        # the rental is gone and the usage is the rental ledger's
        self.assert_no_rental_left()
        self.assertGreaterEqual(res.gpu.minutes, 1.0)
        self.assertEqual(len(res.gpu.rental_ids), 1)
        self.assertTrue(res.log and (self.layout.root / res.log.path).read_text())
        self.assertTrue(set(res.timings) <= {"download_s", "load_s", "generation_s", "train_s", "export_s", "eval_s"})
        self.assertIn("eval_s", res.timings)
        self.assertIn("load_s", res.timings)

    def test_the_endpoint_is_a_control_host_loopback_port(self):
        self.run_job()
        (ep,) = self.generators
        self.assertRegex(ep.base_url, r"^http://127\.0\.0\.1:\d+$")
        self.assertEqual(ep.base_name, "muse-glimmer")

    def test_the_vllm_command_has_the_lora_flags_and_the_verified_hash(self):
        res = self.run_job()
        argv = json.loads((self.venv_vllm / "bin" / "vllm.argv").read_text())
        self.assertIn("--enable-lora", argv)
        self.assertEqual(argv[argv.index("--max-lora-rank") + 1], "32")
        self.assertEqual(argv[argv.index("--lora-modules") + 1], f"{adapter_model_name(res.adapter)}={self.work}/out/adapter")

    def test_gpu_ticks_are_charged_to_the_meter_with_the_documented_call_ids(self):
        res = self.run_job()
        rid = res.gpu.rental_ids[0]
        self.assertIn(f"gpu:{rid}:1", self.meter.charges)
        self.assertIn(f"gpu:{rid}:1:usd", self.meter.charges)
        self.assertAlmostEqual(self.meter.charges[f"gpu:{rid}:1:usd"][1], 1.31 / 60, places=6)
        self.assertGreaterEqual(self.meter.usage().gpu_minutes, 1.0)

    def test_the_rental_cap_is_the_smallest_of_job_meter_and_dollars(self):
        small = BudgetSpec(researcher_tokens=1, gpu_minutes=100.0, dollars=2.0, wall_seconds=100.0)  # 2 dollars buy 91.6 minutes at 1.31/h
        meter = FakeMeter(small)
        self.trainer.pull_margin_s = 0.0
        self.run_job(self.make_job(max_minutes=500.0), meter)
        row = self.manager.read()["rentals"][0]
        minutes = (row["cap_epoch"] - row["create_attempt_epoch"]) / 60
        self.assertLessEqual(minutes, 2.0 / 1.31 * 60 + 1)
        self.assertGreater(minutes, 60)

    def test_the_lease_cap_is_rounded_down_to_whole_minutes_the_allowance_covers(self):
        from nightshift.lab.model_server import _gpu_cap_minutes

        # The manager arms the lease in whole minutes rounded UP, so 1.1 remaining minutes must not become a 2-minute lease.
        meter = FakeMeter(BudgetSpec(researcher_tokens=1, gpu_minutes=1.1, dollars=50.0, wall_seconds=100.0))
        self.assertEqual(_gpu_cap_minutes(self.manager, SHAPE, 60.0, meter), 1.0)
        # the same for a dollar-derived limit: 1.31/h buys 1.6 minutes for $0.035
        meter = FakeMeter(BudgetSpec(researcher_tokens=1, gpu_minutes=100.0, dollars=0.035, wall_seconds=100.0))
        self.assertEqual(_gpu_cap_minutes(self.manager, SHAPE, 60.0, meter), 1.0)
        meter = FakeMeter(BudgetSpec(researcher_tokens=1, gpu_minutes=0.9, dollars=50.0, wall_seconds=100.0))
        with self.assertRaises(BudgetExhausted):
            _gpu_cap_minutes(self.manager, SHAPE, 60.0, meter)

    def test_what_goes_to_the_rental_is_public_data_and_pinned_files_only(self):
        self.run_job()
        uploaded = sorted(p.name for p in (self.work / "data").iterdir())
        self.assertEqual(uploaded, ["dev.jsonl", "train.jsonl"])
        self.assertEqual((self.work / "data" / "train.jsonl").read_text(), self.layout.train_data("a-evo-1").read_text())
        self.assertEqual((self.opt / "rewards.py").read_text(), self.stage_files["rewards.py"].read_text())
        self.assertTrue((self.opt / "ifevalg" / "instructions.py").exists())
        everything = "\n".join(self.fs.commands())
        for banned in ("selection", "IFBench_test", "private/assignments", "private/test", "split_salt", "THUNDER"):
            self.assertNotIn(banned, everything)
        self.assertEqual([p.name for p in (self.opt / "stage").rglob("*") if p.is_file()].count("selection.jsonl"), 0)

    def test_no_secret_reaches_the_rental(self):
        os.environ["THUNDER_API_KEY"] = "tok-must-not-travel"
        self.addCleanup(os.environ.pop, "THUNDER_API_KEY", None)
        self.run_job()
        blob = "\n".join(self.fs.commands()) + json.dumps(self.fs.calls())
        self.assertNotIn("tok-must-not-travel", blob)
        for f in list(self.opt.rglob("*")) + list(self.work.rglob("*")):
            if f.is_file() and f.stat().st_size < 1_000_000:
                self.assertNotIn(b"tok-must-not-travel", f.read_bytes(), str(f))

    def test_a_download_token_reaches_bootstrap_only_never_the_recipe(self):
        canary = "hf_canary_must_not_reach_the_recipe"
        rl = self.remote_layout
        self.trainer.remote_layout = RemoteLayout(rl.opt, rl.work, rl.models, tuple(rl.env) + (("HF_TOKEN", canary),))
        self.assertNotIn("HF_TOKEN", self.trainer.remote_layout.script_env())
        self.assertEqual(self.trainer.remote_layout.script_env(download_token=True)["HF_TOKEN"], canary)
        recipe = b"env > seen_env.txt\n" + GOOD_RECIPE
        res = self.run_job(self.make_job({"run.sh": recipe, "config.yaml": b"a: 1\n"}))
        self.assertEqual(res.status, JobStatus.OK)
        self.assertNotIn(canary, (self.work / "recipe" / "seen_env.txt").read_text())
        self.assertEqual([c for c in self.fs.commands() if canary in c and "bootstrap.sh" not in c], [], "only the bootstrap command carries the token")

    def test_a_watchdog_is_armed_before_the_job_and_the_money_ledger_has_the_trial(self):
        res = self.run_job()
        self.assertEqual(len(self.launcher.calls), 1)
        self.assertEqual(self.manager.usage_for("t-0001").rental_ids, res.gpu.rental_ids)


class JobFailures(Harness):
    def test_failing_recipe(self):
        res = self.run_job(self.make_job({"run.sh": b"echo broken >&2; exit 3\n"}))
        self.assertEqual(res.status, JobStatus.FAILED)
        self.assertIn("recipe_failed", res.failure)
        self.assertIn("broken", res.failure)
        self.assertIsNone(res.adapter)
        self.assertIsNone(res.dev_score_id)
        self.assertEqual(self.scorer.calls, [])
        self.assertFalse(self.layout.adapter_dir("ad-0001").exists())
        self.assert_no_rental_left()
        self.assertGreaterEqual(res.gpu.minutes, 1.0, "failed jobs still report the money they cost")

    def test_recipe_past_its_deadline_is_a_timeout(self):
        self.trainer.pull_margin_s = 45.0  # a 60 s cap leaves about 15 s for the recipe, enough slack for slow fake uploads on a loaded machine
        t0 = time.time()
        res = self.run_job(self.make_job({"run.sh": b"sleep 120\n"}, max_minutes=1.0))
        self.assertEqual(res.status, JobStatus.TIMEOUT, res.failure)
        self.assertLess(time.time() - t0, 60)
        self.assert_no_rental_left()

    def test_no_adapter_and_no_training_data_are_failures(self):
        res = self.run_job(self.make_job({"run.sh": b"echo nothing\n"}))
        self.assertEqual(res.status, JobStatus.FAILED)
        self.assertIn("no_adapter", res.failure)
        body = b'mkdir -p "$GLIMMER_OUT/adapter"; echo c > "$GLIMMER_OUT/adapter/adapter_config.json"; echo w > "$GLIMMER_OUT/adapter/adapter_model.safetensors"\n'
        res = self.run_job(self.make_job({"run.sh": body}, job_id="j-0002"))
        self.assertEqual(res.status, JobStatus.FAILED)
        self.assertIn("no_train_data", res.failure, "a job that bypassed the helper has no training data and fails")

    def test_hostile_recipe_tarball_is_refused_on_the_rental(self):
        job = self.make_job({"run.sh": b"echo hi\n"})
        # replace the archive on disk by one with a symlink member (the control API would have refused it; run_job.sh re-checks)
        path = self.layout.root / job.recipe.path
        with tarfile.open(path, "w:gz") as t:
            ti = tarfile.TarInfo("evil")
            ti.type, ti.linkname = tarfile.SYMTYPE, "/etc/passwd"
            t.addfile(ti)
        bad = hashlib.sha256(path.read_bytes()).hexdigest()
        job2 = TrainJob(job.job_id, job.trial_id, FileRef(job.recipe.path, bad, path.stat().st_size), job.entrypoint, 60.0, SHAPE, job.dev_request)
        res = self.run_job(job2)
        self.assertEqual(res.status, JobStatus.FAILED)
        self.assertIn("bad_recipe", res.failure)

    def test_adapter_with_a_too_large_rank_is_not_installed(self):
        self.remote_layout_env("FAKE_ADAPTER_MODE", "bad_rank")
        res = self.run_job()
        self.assertEqual(res.status, JobStatus.FAILED)
        self.assertIn("not usable", res.failure)
        self.assertFalse(self.layout.adapter_dir("ad-0001").exists())
        self.assertEqual([p.name for p in self.layout.adapter_dir("ad-0001").parent.iterdir()], [], "no half-installed temp directory")
        self.assert_no_rental_left()

    def test_adapter_with_non_lora_tensors_or_out_of_range_offsets_is_not_installed(self):
        for i, mode in enumerate(("not_lora", "oob")):
            self.remote_layout_env("FAKE_ADAPTER_MODE", mode)
            res = self.run_job(self.make_job(job_id=f"j-01{i}0"))
            self.assertEqual(res.status, JobStatus.FAILED, mode)
            self.assertIn("not usable", res.failure)
        self.assertEqual([p.name for p in self.layout.adapter_dir("x").parent.iterdir()], [])

    def remote_layout_env(self, key, value):
        env = dict(self.remote_layout.env)
        env[key] = value
        self.trainer.remote_layout = RemoteLayout(self.remote_layout.opt, self.remote_layout.work, self.remote_layout.models, tuple(env.items()))

    def test_a_hostile_adapter_fetch_is_a_failed_job_not_an_install(self):
        # the rental returns a symlink where the model weights should be (e.g. swapped after run_job.sh looked)
        self.fs.add_rule("-cf - -- adapter_model.safetensors", stdout=make_tar_bytes([("adapter_model.safetensors", "symlink", "/etc/passwd")]))
        res = self.run_job()
        self.assertEqual(res.status, JobStatus.FAILED)
        self.assertIn("refuses", res.failure)
        self.assertFalse(self.layout.adapter_dir("ad-0001").exists())
        self.assertEqual(self.scorer.calls, [])
        self.assert_no_rental_left()

    def test_extra_members_in_the_adapter_fetch_are_refused(self):
        self.fs.add_rule("-cf - -- adapter_model.safetensors", stdout=make_tar_bytes([("adapter_model.safetensors", "file", b"x"), ("pwn.sh", "file", b"#!/bin/sh")]))
        res = self.run_job()
        self.assertEqual(res.status, JobStatus.FAILED)
        self.assertFalse(self.layout.adapter_dir("ad-0001").exists())

    def test_dev_scoring_failure_keeps_the_adapter_but_has_no_score(self):
        self.scorer.error = ScorerError("the endpoint died")
        res = self.run_job()
        self.assertEqual(res.status, JobStatus.FAILED)
        self.assertIn("dev scoring failed", res.failure)
        self.assertIsNotNone(res.adapter)
        self.assertIsNone(res.dev_score_id)
        self.assert_no_rental_left()

    def test_an_access_denied_from_the_scorer_propagates_after_the_teardown(self):
        self.scorer.error = AccessDenied("not allowed")
        with self.assertRaises(AccessDenied):
            self.run_job()
        self.assert_no_rental_left()


class Infrastructure(Harness):
    def test_recipe_tarball_must_match_its_hash(self):
        job = self.make_job()
        bad = TrainJob(job.job_id, job.trial_id, FileRef(job.recipe.path, "0" * 64, job.recipe.size_bytes), job.entrypoint, 60.0, SHAPE, job.dev_request)
        with self.assertRaisesRegex(TrainerError, "does not match"):
            self.run_job(bad)
        self.assertEqual(self.provider.calls_of("create"), [])

    def test_recipe_symlink_is_refused(self):
        job = self.make_job()
        real = self.layout.root / job.recipe.path
        moved = real.with_name("moved.tar.gz")
        real.rename(moved)
        real.symlink_to(moved)
        with self.assertRaises(TrainerError):
            self.run_job(job)
        self.assertEqual(self.provider.calls_of("create"), [])

    def test_stage_file_with_the_wrong_pin_is_refused_before_renting(self):
        self.trainer.stage_pins = {"rewards.py": "0" * 64}
        with self.assertRaisesRegex(TrainerError, "refusing to stage"):
            self.run_job()
        self.assertEqual(self.provider.calls_of("create"), [])

    def test_missing_stage_file_is_refused_before_renting(self):
        self.trainer.stage_files = {"ifevalg/instructions.py": self.tmp / "nope.py"}
        with self.assertRaisesRegex(TrainerError, "cannot stage"):
            self.run_job()
        self.assertEqual(self.provider.calls_of("create"), [])

    def test_unsafe_stage_names_are_refused(self):
        for rel in ("../x.py", "/abs.py", "a b.py"):
            self.trainer.stage_files = {rel: self.stage_src / "rewards.py"}
            with self.assertRaises(TrainerError):
                self.run_job()

    def test_public_data_must_exist(self):
        self.layout.dev_data("a-evo-1").unlink()
        with self.assertRaises(TrainerError):
            self.run_job()
        self.assertEqual(self.provider.calls_of("create"), [])

    def test_create_rejected_is_an_infrastructure_fault_and_leaks_nothing(self):
        self.provider.create_mode = "reject"
        with self.assertRaisesRegex(TrainerError, "could not rent"):
            self.run_job()
        self.assert_no_rental_left()

    def test_bootstrap_failure_deletes_the_rental(self):
        for step in ("machine", "uv", "axolotl_env", "vllm_env", "verify_env", "weights"):
            (self.opt / f".bootstrap.{step}.done").unlink()  # the real bootstrap.sh now runs and fails: no nvidia-smi here
        env = {k: v for k, v in os.environ.items()}
        self.trainer.bootstrap_timeout_s = 60.0
        with self.assertRaisesRegex(TrainerError, "bootstrap|could not rent"):
            self.run_job()
        self.assert_no_rental_left()

    def test_a_rental_that_cannot_be_deleted_is_reported_and_left_to_the_watchdog(self):
        self.provider.fail_delete_n = 50
        with self.assertRaisesRegex(TrainerError, "may still bill"):
            self.run_job()
        self.assertEqual(self.manager.read()["rentals"][0]["state"], "live")
        self.assertEqual(len(alive_watchdogs(self.manager.sessions_dir / self.manager.read()["rentals"][0]["session"])), 1)
        self.provider.fail_delete_n = 0
        self.manager.delete(self.manager.read()["rentals"][0]["session"], "cleanup")
        self.assert_no_rental_left()

    def test_provider_outage_during_the_job_is_infrastructure_not_a_failed_job(self):
        self.provider.fail_list_n = 10**6
        with self.assertRaises(TrainerError):
            self.run_job()
        self.provider.fail_list_n = 0


class Budget(Harness):
    def test_no_gpu_room_returns_budget_exhausted_without_renting(self):
        spent = BudgetSpec(researcher_tokens=1, gpu_minutes=0.5, dollars=50.0, wall_seconds=10.0)
        res = self.run_job(meter=FakeMeter(spent))
        self.assertEqual(res.status, JobStatus.BUDGET_EXHAUSTED)
        self.assertEqual(self.provider.calls_of("create"), [])
        self.assertEqual(res.gpu.minutes, 0.0)

    def test_a_cap_crossed_mid_job_deletes_the_rental_at_once(self):
        tight = BudgetSpec(researcher_tokens=1, gpu_minutes=2.0, dollars=50.0, wall_seconds=10.0)
        meter = FakeMeter(tight)
        self.trainer.pull_margin_s = 0.0
        self.trainer.min_job_seconds = 0.0
        slow = self.make_job({"run.sh": b"sleep 20\n"}, max_minutes=60.0)
        self.clock.offset = 0.0
        sleeper = self.trainer.sleep

        def fast_forward(s):  # each poll of the job moves the budget clock 40 s ahead
            self.clock.offset += 40.0
            sleeper(s)

        self.trainer.sleep = fast_forward
        t0 = time.time()
        res = self.trainer.run_job(slow, meter)
        self.assertEqual(res.status, JobStatus.BUDGET_EXHAUSTED, res.failure)
        self.assertLess(time.time() - t0, 30)
        self.assert_no_rental_left()
        self.assertIn("deleted", [r["state"] for r in self.manager.read()["rentals"]][0])
        self.assertGreaterEqual(meter.usage().gpu_minutes, 2.0)
        self.assertEqual(self.scorer.calls, [])

    def test_the_first_tick_is_booked_at_create_even_for_an_instant_failure(self):
        res = self.run_job(self.make_job({"run.sh": b"exit 1\n"}))
        self.assertGreaterEqual(self.meter.usage().gpu_minutes, 1.0)
        self.assertEqual(res.status, JobStatus.FAILED)

    def test_ticker_books_every_due_tick_once(self):
        from nightshift.lab.model_server import GpuTicker

        now = [1000.0]
        meter = FakeMeter(self.budget)
        fired = []
        t = GpuTicker(meter, "r1", 6.0, lambda: now[0], 1000.0, lambda: fired.append(1))
        t.poll_once()
        self.assertEqual(t.booked, 1)
        now[0] += 125  # 3 minutes elapsed in total -> 3 ticks due
        t.poll_once()
        t.poll_once()
        self.assertEqual(t.booked, 3)
        self.assertEqual(sorted(k for k in meter.charges if k.endswith(":usd")), ["gpu:r1:1:usd", "gpu:r1:2:usd", "gpu:r1:3:usd"])
        self.assertAlmostEqual(meter.usage().gpu_usd, 0.3, places=6)
        self.assertEqual(fired, [])
        t.finish(1000.0 + 301)  # deleted at 5 min 1 s -> 6 billed minutes
        self.assertEqual(t.booked, 6)

    def test_ticker_fires_once_and_still_books_both_halves_when_a_cap_is_crossed(self):
        from nightshift.lab.model_server import GpuTicker

        meter = FakeMeter(BudgetSpec(researcher_tokens=1, gpu_minutes=1.0, dollars=50.0, wall_seconds=1.0))
        fired = []
        t = GpuTicker(meter, "r1", 6.0, lambda: 1000.0, 1000.0, lambda: fired.append(1))
        t.poll_once()
        t.poll_once()
        self.assertEqual(fired, [1])
        self.assertIsInstance(t.exhausted, BudgetExhausted)
        self.assertIn("gpu:r1:1:usd", meter.charges, "the dollar half is booked even though the minutes half crossed the cap")

    def test_a_meter_that_cannot_record_is_treated_as_exhausted(self):
        from nightshift.lab.model_server import GpuTicker

        class Broken(FakeMeter):
            def charge(self, *a, **k):
                raise OSError("disk full")

        fired = []
        t = GpuTicker(Broken(self.budget), "r1", 1.0, lambda: 1000.0, 1000.0, lambda: fired.append(1))
        t.poll_once()
        self.assertEqual(fired, [1])
        self.assertIsNotNone(t.exhausted)


class ModelServerTests(Harness):
    def make_adapter(self, adapter_id="ad-9000"):
        d = self.layout.adapter_dir(adapter_id)
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text('{"peft_type": "LORA", "r": 32}')
        (d / "adapter_model.safetensors").write_bytes(b"weights for scoring")
        return adapter_ref_from_dir(d, adapter_id=adapter_id)

    def server(self, **kw):
        args = dict(
            clock=self.clock, remote_layout=self.remote_layout, sleep=lambda s: time.sleep(0.02), poll_seconds=0.05, tick_interval=0.1,
            vllm_port=self.vllm_port, serve_timeout_s=30.0,
        )  # fmt: skip
        args.update(kw)
        return RentalModelServer(self.manager, SHAPE, HOSTKIT, self.layout, **args)

    def spec(self, adapters=(), minutes=30.0):
        return ServeSpec(BASE, tuple(adapters), 8300, minutes, "unit test", "t-0001")

    def test_protocol_and_a_full_session(self):
        from nightshift.lab.contracts import ModelServer, ServeSession

        ms = self.server()
        self.assertIsInstance(ms, ModelServer)
        ad = self.make_adapter()
        with ms.serve(self.spec([ad]), self.meter) as session:
            self.assertIsInstance(session, ServeSession)
            ep = session.endpoint
            self.assertEqual(ep.base_name, "muse-glimmer")
            self.assertEqual(ep.model_for(ad), adapter_model_name(ad))
            self.assertEqual(ep.model_for(None), "muse-glimmer")
            self.assertRegex(ep.base_url, r"^http://127\.0\.0\.1:\d+$")
            from urllib.request import urlopen

            listed = json.loads(urlopen(ep.base_url + "/v1/models", timeout=5).read())["data"]
            self.assertEqual({m["id"] for m in listed}, {"muse-glimmer", adapter_model_name(ad)})
            # the adapter files on the "rental" are the control host's verified bytes
            rdir = self.models / "adapters" / "ad-9000"
            self.assertEqual((rdir / "adapter_model.safetensors").read_bytes(), b"weights for scoring")
            self.assertEqual(len(self.provider.handles()), 1)
        self.assertEqual(self.provider.handles(), set())

    def test_close_returns_usage_and_is_idempotent(self):
        session = self.server().serve(self.spec(), self.meter)
        usage = session.close()
        self.assertGreaterEqual(usage.minutes, 1.0)
        self.assertEqual(session.close(), usage)
        self.assert_no_rental_left()
        self.assertEqual(len(self.provider.calls_of("delete")), 1)

    def test_each_session_rents_a_fresh_machine(self):
        ms = self.server()
        a, b = ms.serve(self.spec(), self.meter), ms.serve(self.spec(), self.meter)
        self.assertNotEqual(a.lease.rental.session, b.lease.rental.session)
        a.close()
        b.close()
        self.assertEqual(len(self.provider.calls_of("create")), 2)

    def test_no_gpu_room_raises_budget_exhausted_without_renting(self):
        with self.assertRaises(BudgetExhausted):
            self.server().serve(self.spec(), FakeMeter(BudgetSpec(researcher_tokens=1, gpu_minutes=0.0, dollars=5.0, wall_seconds=1.0)))
        self.assertEqual(self.provider.calls_of("create"), [])

    def test_a_tampered_control_host_adapter_is_never_uploaded_and_the_rental_is_deleted(self):
        ad = self.make_adapter()
        (self.layout.adapter_dir("ad-9000") / "adapter_model.safetensors").write_bytes(b"tampered after it was recorded")
        with self.assertRaises(Exception) as cm:
            self.server().serve(self.spec([ad]), self.meter)
        self.assertIn("listing sha256", str(cm.exception))
        self.assertFalse((self.models / "adapters").exists())
        self.assert_no_rental_left()

    def test_a_server_that_never_comes_up_deletes_the_rental(self):
        (self.venv_vllm / "bin" / "vllm").write_text("#!/usr/bin/env bash\nexit 1\n")
        with self.assertRaisesRegex(TrainerError, "did not list"):
            self.server(serve_timeout_s=2.0).serve(self.spec(), self.meter)
        self.assert_no_rental_left()

    def test_serving_needs_the_layout(self):
        ms = RentalModelServer(self.manager, SHAPE, HOSTKIT)
        with self.assertRaises(TrainerError):
            ms.serve(self.spec(), self.meter)

    def test_budget_exhausted_during_a_session_deletes_the_rental(self):
        tight = BudgetSpec(researcher_tokens=1, gpu_minutes=2.0, dollars=50.0, wall_seconds=10.0)
        meter = FakeMeter(tight)
        session = self.server().serve(self.spec(minutes=60.0), meter)
        self.clock.offset += 120.0  # the budget clock runs ahead: the next tick crosses the one-minute allowance
        for _ in range(100):
            if self.provider.handles() == set():
                break
            time.sleep(0.05)
        self.assertEqual(self.provider.handles(), set(), "the ticker deletes the rental when a cap is crossed")
        self.assertIsNotNone(session.lease.exhausted)
        session.close()


    def test_a_failed_teardown_keeps_the_ticker_and_retries_it_after_the_budget_is_crossed(self):
        tight = BudgetSpec(researcher_tokens=1, gpu_minutes=2.0, dollars=50.0, wall_seconds=10.0)
        session = self.server().serve(self.spec(minutes=60.0), FakeMeter(tight))
        self.provider.fail_delete_n = 10**6  # a provider outage: every deletion attempt fails
        self.clock.offset += 120.0
        for _ in range(200):
            if len(self.provider.calls_of("delete")) >= 3:
                break
            time.sleep(0.05)
        self.assertGreaterEqual(len(self.provider.calls_of("delete")), 3)
        self.assertEqual(len(self.provider.handles()), 1, "the outage defeated the deletion; the rental still bills")
        self.assertFalse(session.lease.closed)
        self.provider.fail_delete_n = 0  # the outage ends: the ticker must still be watching and delete the rental
        for _ in range(200):
            if self.provider.handles() == set():
                break
            time.sleep(0.05)
        self.assertEqual(self.provider.handles(), set(), "the budget callback fires again after a failed teardown")
        session.close()

    def test_the_requested_base_reaches_the_rental_and_a_different_one_is_refused(self):
        from nightshift.lab.model_server import acquire_lease, serve_on_lease

        other = BaseModelRef("meta-models/Muse-Glimmer-30B", "b" * 40)
        session = self.server().serve(ServeSpec(other, (), 8300, 30.0, "unit test", "t-0001"), self.meter)
        env = session.lease.layout_remote.script_env()
        self.assertEqual((env["GLIMMER_REPO"], env["GLIMMER_REVISION"]), (other.repo_id, other.revision))
        with self.assertRaisesRegex(TrainerError, "not the requested"):
            serve_on_lease(session.lease, self.spec())  # BASE: the rental holds `other`
        session.close()
        lease = acquire_lease(
            self.manager, SHAPE, session="s-nobase", trial_id="t-0001", item="x", max_minutes=30.0, meter=self.meter, hostkit_dir=HOSTKIT,
            root=self.layout.root, layout_remote=self.remote_layout, clock=self.clock, sleep=lambda s: time.sleep(0.02), poll_seconds=0.05,
            tick_interval=0.1, vllm_port=self.vllm_port, bootstrap_timeout_s=60.0,
        )  # fmt: skip
        try:
            with self.assertRaisesRegex(TrainerError, "unrecorded checkpoint"):
                serve_on_lease(lease, self.spec())
        finally:
            lease.close("test")


class ConcurrentCreates(Harness):
    def test_a_second_trial_waits_for_the_create_in_flight_instead_of_failing(self):
        import threading

        from nightshift.lab.model_server import create_when_settled

        gate, entered = threading.Event(), threading.Event()
        self.provider.on_create = lambda: (entered.set(), gate.wait(10))
        out = {}

        def first():
            out["a"] = self.manager.create("s-a", SHAPE, 1.0, "t-0001", "train")

        t = threading.Thread(target=first)
        t.start()
        self.assertTrue(entered.wait(5))
        self.provider.on_create = None
        waits = []

        def sleep(s):
            waits.append(s)
            gate.set()  # the first create settles while the second waits
            time.sleep(0.05)

        b = create_when_settled(self.manager, "s-b", SHAPE, 1.0, "t-0002", "train", wait_s=30, clock=self.clock, sleep=sleep, poll_seconds=0.01)
        t.join(5)
        self.assertTrue(waits, "the second create had to wait")
        self.assertEqual((out["a"].state, b.state), ("live", "live"))
        self.assertEqual(len(self.provider.calls_of("create")), 2)

    def test_a_stuck_unsettled_rental_is_not_waited_for_forever(self):
        from nightshift.lab.gpu_rentals import RentalRefused
        from nightshift.lab.model_server import create_when_settled

        self.provider.create_mode = "ambiguous_not_created"
        with self.assertRaises(Exception):
            self.manager.create("s-a", SHAPE, 1.0, "t-0001", "train")
        self.provider.create_mode = "ok"
        t = [0.0]
        with self.assertRaises(RentalRefused) as cm:
            create_when_settled(self.manager, "s-b", SHAPE, 1.0, "t-0002", "train", wait_s=5, clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s), poll_seconds=1.0)
        self.assertEqual(cm.exception.code, "unsettled")

    def test_the_money_rule_is_not_retried(self):
        from nightshift.lab.gpu_rentals import RentalRefused
        from nightshift.lab.model_server import create_when_settled

        self.manager.set_hard_stop(0.5, "tiny")
        calls = []
        with self.assertRaises(RentalRefused) as cm:
            create_when_settled(self.manager, "s-a", SHAPE, 1.0, "t-0001", "train", wait_s=30, clock=self.clock, sleep=lambda s: calls.append(s), poll_seconds=0.01)
        self.assertEqual((cm.exception.code, calls), ("refused", []))


class FakeTrainerTests(unittest.TestCase):
    """The double owner 5 and 6 build worlds from must honour the same contract as the real trainer."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.layout = LabLayout(Path(self._td.name) / "lab")
        self.clock = FakeClock()
        self.meter = FakeMeter(BudgetSpec(researcher_tokens=10, gpu_minutes=30.0, dollars=10.0, wall_seconds=60.0))

    def job(self):
        req = ScoreRequest("sc-0001", Phase.SEARCH, "a-evo-1", Split.DEV, "t-0001", None, BASE, None, (0, 1), GenerationParams.recommended(1000), "sv1-x", None, None)
        return TrainJob("j-0001", "t-0001", FileRef("trials/t-0001/recipe/" + "a" * 64 + ".tar.gz", "a" * 64, 1), ("bash", "run.sh"), 20.0, SHAPE, req)

    def test_ok_result_is_complete_and_consistent_with_the_layout(self):
        from tests.support.lab_fake_gpu import FakeTrainer

        ft = FakeTrainer(self.layout, clock=self.clock, dev_score=0.62)
        self.assertIsInstance(ft, Trainer)
        res = ft.run_job(self.job(), self.meter)
        self.assertEqual(res.status, JobStatus.OK)
        verify_adapter_dir(res.adapter, root=self.layout.root)
        rec = read_record(self.layout.score_record_path(res.dev_score_id), ScoreRecord)
        self.assertAlmostEqual(rec.headline_mean(), 0.62)
        self.assertEqual(rec.adapter_sha256, res.adapter.sha256)
        self.assertEqual(self.layout.file_ref(self.layout.trial_train_data_path("t-0001", "j-0001")), res.train_data)
        self.assertEqual(self.meter.usage().gpu_minutes, 10.0)

    def test_budget_and_failure_modes(self):
        from tests.support.lab_fake_gpu import FakeTrainer

        ft = FakeTrainer(self.layout, clock=self.clock, gpu_minutes=100.0)
        res = ft.run_job(self.job(), FakeMeter(BudgetSpec(researcher_tokens=1, gpu_minutes=5.0, dollars=10.0, wall_seconds=1.0)))
        self.assertEqual(res.status, JobStatus.BUDGET_EXHAUSTED)
        ft2 = FakeTrainer(self.layout, clock=self.clock)
        ft2.fail_with = JobStatus.TIMEOUT
        self.assertEqual(ft2.run_job(self.job(), self.meter).status, JobStatus.TIMEOUT)


class AdapterValidation(unittest.TestCase):
    def test_config(self):
        ok = json.dumps({"peft_type": "LORA", "r": 32, "lora_alpha": 16, "target_modules": ["q_proj"]}).encode()
        self.assertEqual(tr.validate_adapter_config(ok)["r"], 32)
        bad = {
            "not lora": {"peft_type": "PREFIX_TUNING", "r": 8},
            "rank 0": {"peft_type": "LORA", "r": 0},
            "rank 64": {"peft_type": "LORA", "r": 64},
            "rank bool": {"peft_type": "LORA", "r": True},
            "dora": {"peft_type": "LORA", "r": 8, "use_dora": True},
            "modules_to_save": {"peft_type": "LORA", "r": 8, "modules_to_save": ["lm_head"]},
            "alpha text": {"peft_type": "LORA", "r": 8, "lora_alpha": "x"},
            "target type": {"peft_type": "LORA", "r": 8, "target_modules": 5},
        }
        for name, cfg in bad.items():
            with self.subTest(name), self.assertRaises(TrainerError):
                tr.validate_adapter_config(json.dumps(cfg).encode())
        for raw in (b"not json", b"[1]", b'{"r": NaN}', b'{"peft_type": "LORA", "r": 8, "r": 9}'):
            with self.subTest(raw=raw), self.assertRaises(TrainerError):
                tr.validate_adapter_config(raw)

    def test_safetensors(self):
        import struct

        def write(name, header, body=b"\x00" * 64):
            p = Path(tempfile.mkdtemp()) / "m.safetensors"
            h = json.dumps(header).encode()
            p.write_bytes(struct.pack("<Q", len(h)) + h + body)
            return p

        good = write("x", {"base_model.model.layers.0.q_proj.lora_A.weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]}, "__metadata__": {"format": "pt"}})
        self.assertEqual(tr.validate_safetensors(good), 1)
        for name, header in {
            "not lora": {"model.embed.weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]}},
            "outside": {"a.lora_A.weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 6400]}},
            "negative": {"a.lora_A.weight": {"dtype": "F32", "shape": [1], "data_offsets": [-4, 4]}},
            "empty": {},
            "bad offsets": {"a.lora_A.weight": {"dtype": "F32", "shape": [1], "data_offsets": "x"}},
        }.items():
            with self.subTest(name), self.assertRaises(TrainerError):
                tr.validate_safetensors(write("x", header))
        p = Path(tempfile.mkdtemp()) / "short"
        p.write_bytes(b"123")
        with self.assertRaises(TrainerError):
            tr.validate_safetensors(p)
        p.write_bytes(struct.pack("<Q", 10**12) + b"x" * 20)
        with self.assertRaises(TrainerError):
            tr.validate_safetensors(p)

    def test_timing_names_are_mapped_to_the_canonical_keys(self):
        got = tr.map_timings({"generation_s": 5, "train": 7, "train_seconds": 3, "download": 1.5, "export_s": 2, "mystery": 9, "eval_s": -1, "load_s": True, "x": "1"})
        self.assertEqual(got, {"generation_s": 5.0, "train_s": 10.0, "download_s": 1.5, "export_s": 2.0})
        self.assertEqual(tr.map_timings("nope"), {})
        self.assertEqual(tr.map_timings({"train_s": float("nan")}), {})

    def test_default_stage_files_point_at_the_third_party_checkout(self):
        layout = LabLayout(Path("/lab"))
        files = tr.default_stage_files(layout)
        self.assertEqual(files["ifevalg/instructions.py"], Path("/lab/third_party/open-instruct/open_instruct/IFEvalG/instructions.py"))
        self.assertEqual(sum(1 for k in files if k.startswith("ifevalg/")), 3)


if __name__ == "__main__":
    unittest.main()
