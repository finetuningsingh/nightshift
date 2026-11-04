import hashlib
import io
import json
import tarfile
import tempfile
import threading
import unittest
from unittest import mock

from nightshift.lab import control_api
from nightshift.lab.contracts import JobStatus, LabKind, Phase, Split
from nightshift.lab.control_api import ControlApi, RecipeRejected, entrypoint_script, parse_entrypoint, validate_recipe_tarball
from nightshift.lab.metering_proxy import LabHttpServer
from tests.support.lab_fake_runtime import ControlClient, FakeLedger, FakeMeter, TrialFixture, make_recipe

TOKEN = "control-token-0123456789abcdef"


def tar_with(members: list) -> bytes:
    """members: (TarInfo, bytes|None) pairs -> gzip tar."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for ti, data in members:
            tar.addfile(ti, io.BytesIO(data) if data is not None else None)
    return buf.getvalue()


def reg(name: str, data: bytes = b"x") -> tuple:
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    return ti, data


def special(name: str, type_: bytes, linkname: str = "") -> tuple:
    ti = tarfile.TarInfo(name)
    ti.type, ti.linkname = type_, linkname
    return ti, None


class ApiCase(unittest.TestCase):
    final = False

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fx = TrialFixture(self._tmp.name, final=self.final)
        self.spec = self.fx.spec("t-0001")
        self.ledger = FakeLedger("trial")
        self.meter = FakeMeter(self.spec.budget, self.ledger, trial_id="t-0001", clock=self.fx.clock)
        self.hook_calls = []
        self.hook_result = None
        self.api = ControlApi(self.fx.trainer, self.meter, self.fx.layout, self.spec, TOKEN, self.ledger, clock=self.fx.clock, on_job_finished=self.hook)
        self.server = LabHttpServer([self.api]).start()
        self.addCleanup(self.server.stop)
        self.addCleanup(lambda: self.api.close(wait=False))
        self.client = ControlClient(f"http://127.0.0.1:{self.server.port}/control", TOKEN)

    def hook(self, job, result, record):
        self.hook_calls.append((job, result, record))
        return self.hook_result

    def submit_ok(self, **kw):
        status, doc = self.client.submit(make_recipe(), **kw)
        self.assertEqual((status, doc["status"]), (200, "queued"), doc)
        return doc["job_id"]


class AuthTests(ApiCase):
    def test_every_endpoint_refuses_a_wrong_token_and_touches_nothing(self):
        for method, path, data in (("GET", "/budget", None), ("POST", "/submit", make_recipe()), ("GET", "/status", None), ("GET", "/dev-score", None), ("GET", "/pointer", None)):
            for tok in ("", "wrong", TOKEN[:-1], TOKEN + "0"):
                status, doc = self.client.request(method, path, data=data, token=tok)
                self.assertEqual(status, 401, (path, tok))
        self.assertEqual(self.fx.trainer.jobs, [])
        self.assertEqual(self.ledger.events, [])
        self.assertFalse(self.fx.layout.trial_recipe_dir("t-0001").exists())

    def test_unknown_routes_and_methods(self):
        self.assertEqual(self.client.request("GET", "/nope")[0], 404)
        self.assertEqual(self.client.request("GET", "/submit")[0], 404)
        self.assertEqual(self.client.request("POST", "/budget", data=b"")[0], 404)


class BudgetAndStatusTests(ApiCase):
    def test_budget_left(self):
        status, doc = self.client.request("GET", "/budget")
        self.assertEqual(status, 200)
        self.assertEqual(doc, {"researcher_tokens_left": 2_000_000, "gpu_minutes_left": 120.0, "dollars_left": 25.0, "wall_seconds_left": 7200.0})
        self.meter.charge(__import__("nightshift.lab.contracts", fromlist=["Resource"]).Resource.GPU_MINUTES, 30.0, "c1")
        self.fx.clock.advance(100)
        _s, doc = self.client.request("GET", "/budget")
        self.assertEqual(doc["gpu_minutes_left"], 90.0)
        self.assertEqual(doc["wall_seconds_left"], 7100.0)

    def test_status_without_jobs_and_unknown_job(self):
        self.assertEqual(self.client.status(), (200, {"state": "none", "n_jobs": 0}))
        self.assertEqual(self.client.status("t-0001-j9")[0], 404)


class JobLifecycleTests(ApiCase):
    def test_submit_runs_one_job_and_serves_its_dev_score(self):
        tar = make_recipe()
        status, doc = self.client.submit(tar)
        self.assertEqual((status, doc["status"]), (200, "queued"))
        job_id = doc["job_id"]
        self.assertEqual(job_id, "t-0001-j1")
        res = self.client.wait(job_id)
        self.assertEqual(res["state"], "ok")
        self.assertEqual(res["adapter_id"], "ad-0001")
        self.assertAlmostEqual(res["dev_score"], 0.45)
        self.assertEqual(res["gpu_minutes_used"], 5.0)
        # what the trainer received
        job = self.fx.trainer.jobs[0]
        self.assertEqual(job.entrypoint, ("bash", "run.sh"))
        self.assertEqual(job.shape, self.spec.shape)
        req = job.dev_request
        self.assertEqual((req.split, req.arm, req.adapter, req.phase, req.final_request_id), (Split.DEV, None, None, Phase.SEARCH, None))
        self.assertEqual((req.seeds, req.params, req.scorer_version, req.assignment_id, req.trial_id), (self.spec.score_seeds, self.spec.score_params, self.spec.scorer_version, "a-evo-1", "t-0001"))
        self.assertEqual(req.request_id, "t-0001-dev1")
        # the recipe is stored by hash, beside the trial, and the TrainJob refers to it
        sha = hashlib.sha256(tar).hexdigest()
        self.assertEqual(job.recipe.sha256, sha)
        self.assertEqual(job.recipe.path, f"trials/t-0001/recipe/{sha}.tar.gz")
        self.assertEqual((self.fx.layout.root / job.recipe.path).read_bytes(), tar)
        # ledger events
        self.assertEqual([k for k in self.ledger.kinds() if k != LabKind.BUDGET_CHARGED], [LabKind.GPU_JOB_STARTED, LabKind.GPU_JOB_FINISHED])
        self.assertEqual(self.ledger.of(LabKind.GPU_JOB_STARTED)[0], job)
        self.assertEqual(self.ledger.of(LabKind.GPU_JOB_FINISHED)[0].adapter_id, "ad-0001")
        # dev score and pointer
        s, ds = self.client.request("GET", "/dev-score", {"adapter_id": "ad-0001"})
        self.assertEqual(s, 200)
        self.assertEqual((ds["metric"], ds["n_prompts"]), ("prompt_all", 20))
        self.assertAlmostEqual(ds["dev_score"], 0.45)
        s, ptr = self.client.request("GET", "/pointer", {"adapter_id": "ad-0001"})
        self.assertEqual((s, ptr["recipe_sha256"]), (200, sha))
        # the runner's view
        ok = self.api.ok_adapters()
        self.assertEqual(list(ok), ["ad-0001"])
        self.assertEqual(ok["ad-0001"][1].score_id, "t-0001-dev1")
        # status without an id is the latest job
        _s, latest = self.client.status()
        self.assertEqual((latest["job_id"], latest["n_jobs"]), (job_id, 1))

    def test_hook_receives_job_result_and_the_dev_record(self):
        self.submit_ok()
        self.client.wait("t-0001-j1")
        (job, result, record), = self.hook_calls
        self.assertEqual((job.job_id, result.status, record.score_id), ("t-0001-j1", JobStatus.OK, "t-0001-dev1"))

    def test_hook_rejection_withholds_the_adapter_and_its_dev_score(self):
        self.hook_result = "contamination scan hit"
        self.submit_ok()
        res = self.client.wait("t-0001-j1")
        self.assertEqual(res["state"], "failed")
        self.assertEqual(res["failure"], "contamination scan hit")
        self.assertNotIn("adapter_id", res)
        self.assertNotIn("dev_score", res)
        self.assertEqual(self.client.request("GET", "/dev-score", {"adapter_id": "ad-0001"})[0], 404)
        self.assertEqual(self.client.request("GET", "/pointer", {"adapter_id": "ad-0001"})[0], 404)
        self.assertEqual(self.api.ok_adapters(), {})

    def test_hook_that_raises_rejects_the_job(self):
        def boom(job, result, record):
            raise RuntimeError("scanner exploded with /secret/path")

        self.api._hook = boom
        self.submit_ok()
        res = self.client.wait("t-0001-j1")
        self.assertEqual(res["state"], "failed")
        self.assertNotIn("/secret/path", json.dumps(res))
        self.assertEqual(self.api.ok_adapters(), {})

    def test_dev_scores_exist_only_for_this_trials_own_adapters(self):
        self.submit_ok()
        self.client.wait("t-0001-j1")
        for aid in ("ad-9999", "../etc", "", "ad-0001x"):
            self.assertEqual(self.client.request("GET", "/dev-score", {"adapter_id": aid})[0], 404, aid)
        # a second trial's API does not know the first trial's adapter even though the files exist on the same lab root
        spec2 = self.fx.spec("t-0002")
        api2 = ControlApi(self.fx.trainer, FakeMeter(spec2.budget, clock=self.fx.clock, trial_id="t-0002"), self.fx.layout, spec2, "other-token-0123456789", FakeLedger(), clock=self.fx.clock)
        server2 = LabHttpServer([api2]).start()
        self.addCleanup(server2.stop)
        c2 = ControlClient(f"http://127.0.0.1:{server2.port}/control", "other-token-0123456789")
        self.assertEqual(c2.request("GET", "/dev-score", {"adapter_id": "ad-0001"})[0], 404)
        self.assertEqual(self.client.request("GET", "/dev-score", {"adapter_id": "ad-0001"}, token="other-token-0123456789")[0], 401)

    def test_trainer_exception_fails_the_job_and_the_worker_survives(self):
        self.fx.trainer.raise_exc = RuntimeError("rental exploded")
        j1 = self.submit_ok()
        res = self.client.wait(j1)
        self.assertEqual(res["state"], "failed")
        self.assertIn("rental exploded", res["failure"])
        finished = self.ledger.of(LabKind.GPU_JOB_FINISHED)
        self.assertEqual(finished[0].status, JobStatus.FAILED)
        self.fx.trainer.raise_exc = None
        j2 = self.submit_ok()
        self.assertEqual(self.client.wait(j2)["state"], "ok")

    def test_failed_and_timeout_and_budget_results_map_to_states(self):
        for status, state in ((JobStatus.FAILED, "failed"), (JobStatus.TIMEOUT, "timeout"), (JobStatus.BUDGET_EXHAUSTED, "budget_exhausted")):
            self.fx.trainer.fail_status = status
            res = self.client.wait(self.submit_ok())
            self.assertEqual(res["state"], state)
            self.assertIn("failure", res)
            self.assertNotIn(str(self.fx.layout.root), res["failure"])  # control-host paths are not shown to researchers
            self.assertIn("<lab>", res["failure"])

    def test_gpu_budget_exhaustion_inside_a_job(self):
        from nightshift.lab.contracts import BudgetSpec

        spec = self.fx.spec("t-0003", budget=BudgetSpec(2_000_000, 3.0, 25.0, 7200.0))
        led = FakeLedger()
        meter = FakeMeter(spec.budget, led, trial_id="t-0003", clock=self.fx.clock)
        api = ControlApi(self.fx.trainer, meter, self.fx.layout, spec, TOKEN, led, clock=self.fx.clock)
        server = LabHttpServer([api]).start()
        self.addCleanup(server.stop)
        self.addCleanup(lambda: api.close(wait=False))
        c = ControlClient(f"http://127.0.0.1:{server.port}/control", TOKEN)
        _s, doc = c.submit(make_recipe(), max_gpu_minutes=50)
        res = c.wait(doc["job_id"])
        self.assertEqual(res["state"], "budget_exhausted")
        self.assertEqual(self.fx.trainer.jobs[-1].max_gpu_minutes, 3.0)  # capped to what the trial had left
        # nothing is left: the next submission is refused before it costs anything
        status, doc = c.submit(make_recipe())
        self.assertEqual((status, doc["status"]), (422, "refused"))
        self.assertIn("budget exhausted", doc["reason"])

    def test_job_cap_is_the_smaller_of_request_gpu_left_and_wall_left(self):
        self.submit_ok(max_gpu_minutes=17.5)
        self.client.wait("t-0001-j1")
        self.assertEqual(self.fx.trainer.jobs[0].max_gpu_minutes, 17.5)
        self.fx.clock.advance(7200 - 600)  # 10 minutes of wall time left
        self.submit_ok()
        self.client.wait("t-0001-j2")
        self.assertAlmostEqual(self.fx.trainer.jobs[1].max_gpu_minutes, 10.0)

    def test_jobs_run_one_at_a_time_in_order(self):
        gate = threading.Event()
        self.fx.trainer.block = gate
        ids = [self.submit_ok() for _ in range(3)]
        self.assertTrue(self.fx.trainer.entered.wait(5))
        states = [self.client.status(j)[1]["state"] for j in ids]
        self.assertEqual(states, ["running", "queued", "queued"])
        gate.set()
        for j in ids:
            self.assertEqual(self.client.wait(j)["state"], "ok")
        self.assertEqual(self.fx.trainer.max_parallel, 1)
        self.assertEqual([j.job_id for j in self.fx.trainer.jobs], ids)
        self.assertEqual(sorted(self.api.ok_adapters()), ["ad-0001", "ad-0002", "ad-0003"])

    def test_close_cancels_queued_jobs_waits_for_the_running_one_and_refuses_new_ones(self):
        gate = threading.Event()
        self.fx.trainer.block = gate
        j1, j2 = self.submit_ok(), self.submit_ok()
        self.assertTrue(self.fx.trainer.entered.wait(5))
        threading.Timer(0.3, gate.set).start()
        self.assertTrue(self.api.close(wait=True, timeout=10))
        self.assertEqual(self.client.status(j1)[1]["state"], "ok")  # the running job was allowed to finish
        st = self.client.status(j2)[1]
        self.assertEqual(st["state"], "failed")
        self.assertEqual(len(self.fx.trainer.jobs), 1)
        status, doc = self.client.submit(make_recipe())
        self.assertEqual((status, doc["status"]), (422, "refused"))

    def test_dev_record_that_does_not_belong_to_the_job_is_not_trusted(self):
        trainer = self.fx.trainer
        orig = trainer.run_job

        def lying(job, meter):
            res = orig(job, meter)
            # point the result at somebody else's dev score
            return res.__class__(res.job_id, res.trial_id, res.status, res.adapter_id, res.adapter, "t-0001-dev1", res.gpu, res.timings, res.log, res.train_data, res.failure)

        trainer.run_job = lying
        self.submit_ok()
        self.client.wait("t-0001-j1")
        # first job is fine (its own record) ...
        self.assertEqual(list(self.api.ok_adapters()), ["ad-0001"])
        # ... the second claims the first job's score id, which carries another adapter's hash
        self.submit_ok()
        res = self.client.wait("t-0001-j2")
        self.assertEqual(res["state"], "failed")
        self.assertEqual(list(self.api.ok_adapters()), ["ad-0001"])


    def test_dev_record_of_another_trial_is_not_trusted(self):
        self.fx.trainer.dev_trial_override = "t-9999"
        self.submit_ok()
        res = self.client.wait("t-0001-j1")
        self.assertEqual(res["state"], "failed")
        self.assertEqual(self.api.ok_adapters(), {})


class EntrypointTests(unittest.TestCase):
    def test_script_is_found_behind_an_interpreter(self):
        self.assertEqual(entrypoint_script(parse_entrypoint("bash run.sh")), "run.sh")
        self.assertEqual(entrypoint_script(parse_entrypoint("python3 -u ./cfg/train.py --epochs 3")), "cfg/train.py")
        self.assertEqual(entrypoint_script(parse_entrypoint("./go.sh")), "go.sh")
        for bad in ("bash ../x.sh", "bash /etc/passwd", "bash", "", "bash   "):
            with self.assertRaises(RecipeRejected, msg=bad):
                entrypoint_script(parse_entrypoint(bad))
        with self.assertRaises(RecipeRejected):
            parse_entrypoint("bash 'unterminated")
        with self.assertRaises(RecipeRejected):
            parse_entrypoint(" ".join(["a"] * 21))


class RecipeRulesTests(ApiCase):
    def assertRefused(self, data: bytes, reason_part: str = "", entrypoint: str = None):
        status, doc = self.client.submit(data, entrypoint=entrypoint)
        self.assertEqual((status, doc["status"], doc["job_id"]), (422, "refused", ""), doc)
        self.assertIn(reason_part, doc["reason"])
        self.assertEqual(self.fx.trainer.jobs, [])
        self.assertEqual(self.api.n_jobs(), 0)
        d = self.fx.layout.trial_recipe_dir("t-0001")
        self.assertTrue(not d.exists() or list(d.iterdir()) == [])  # nothing stored for a refused recipe
        self.assertEqual(self.ledger.events, [])

    def test_symbolic_link_hard_link_and_devices_are_refused(self):
        run = reg("run.sh")
        self.assertRefused(tar_with([run, special("link", tarfile.SYMTYPE, "/etc/passwd")]), "regular")
        self.assertRefused(tar_with([run, special("hl", tarfile.LNKTYPE, "run.sh")]), "regular")
        self.assertRefused(tar_with([run, special("dev", tarfile.CHRTYPE)]), "regular")
        self.assertRefused(tar_with([run, special("blk", tarfile.BLKTYPE)]), "regular")
        self.assertRefused(tar_with([run, special("fifo", tarfile.FIFOTYPE)]), "regular")

    def test_escaping_absolute_and_git_paths_are_refused(self):
        for name in ("../evil.sh", "a/../../evil", "/abs/path", ".git/config", "sub/.git/hooks/pre-commit", "a/./../b"):
            self.assertRefused(tar_with([reg("run.sh"), reg(name)]), "path")

    def test_entrypoint_must_be_a_file_in_the_recipe(self):
        self.assertRefused(make_recipe({"other.sh": b"x"}), "not a file in the recipe")
        self.assertRefused(make_recipe(), "entrypoint", entrypoint="bash ../run.sh")
        self.assertRefused(make_recipe(), "not a file in the recipe", entrypoint="bash cfg")  # a directory is not a script

    def test_not_a_gzip_tar_and_empty(self):
        self.assertRefused(b"", "empty")
        self.assertRefused(b"this is not a tarball", "not a valid")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:  # a tar that is not gzipped
            ti = tarfile.TarInfo("run.sh")
            tar.addfile(ti, io.BytesIO(b""))
        self.assertRefused(buf.getvalue(), "not a valid")

    def test_too_many_entries(self):
        members = [reg(f"d/f{i}", b"") for i in range(control_api.MAX_RECIPE_ENTRIES)] + [reg("run.sh")]
        self.assertRefused(tar_with(members), "entries")

    def test_gzip_bomb_is_cut_off_by_the_unpacked_limit(self):
        big = tarfile.TarInfo("big.bin")
        big.size = 201 * 1024 * 1024
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.addfile(tarfile.TarInfo("run.sh"), io.BytesIO(b""))
            class Zeros(io.RawIOBase):
                left = big.size
                def readable(self): return True
                def readinto(self, b):
                    n = min(len(b), self.left)
                    b[:n] = bytes(n)
                    self.left -= n
                    return n
            tar.addfile(big, io.BufferedReader(Zeros()))
        data = buf.getvalue()
        self.assertLess(len(data), 5 * 1024 * 1024)  # tiny on the wire, 201 MB unpacked
        self.assertRefused(data, "unpacks to more than")

    def test_oversized_upload_is_refused_before_it_is_read_in_full(self):
        with mock.patch.object(control_api, "MAX_RECIPE_COMPRESSED_BYTES", 100):
            status, doc = self.client.request("POST", "/submit", data=make_recipe({"run.sh": b"x" * 5000, "junk": bytes(range(256)) * 40}))
        self.assertEqual(status, 413)
        self.assertEqual(self.fx.trainer.jobs, [])

    def test_compressed_size_limit(self):
        with mock.patch.object(control_api, "MAX_RECIPE_COMPRESSED_BYTES", 200):
            with self.assertRaises(RecipeRejected):
                validate_recipe_tarball(make_recipe({"run.sh": b"x" * 5000, "junk": bytes(range(256)) * 40}), ("bash", "run.sh"))

    def test_a_good_recipe_with_directories_passes_validation(self):
        n = validate_recipe_tarball(make_recipe({"run.sh": b"x", "a/b/c.txt": b"y"}), ("bash", "run.sh"))
        self.assertGreaterEqual(n, 2)

    def test_bad_max_gpu_minutes(self):
        for v in ("abc", "-3", "0", "nan", "inf"):
            status, doc = self.client.request("POST", "/submit", {"max_gpu_minutes": v}, make_recipe())
            self.assertEqual((status, doc["status"]), (422, "refused"), v)
        self.assertEqual(self.fx.trainer.jobs, [])


class FinalPhaseTests(ApiCase):
    final = True

    def setUp(self):
        super().setUp()
        self.spec = self.fx.final_spec("f-evolved-0")
        self.api = ControlApi(self.fx.trainer, FakeMeter(self.spec.budget, clock=self.fx.clock, trial_id=self.spec.trial_id), self.fx.layout, self.spec, TOKEN, FakeLedger(), clock=self.fx.clock)
        self.server2 = LabHttpServer([self.api]).start()
        self.addCleanup(self.server2.stop)
        self.addCleanup(lambda: self.api.close(wait=False))
        self.client = ControlClient(f"http://127.0.0.1:{self.server2.port}/control", TOKEN)

    def test_dev_template_of_a_final_trial_carries_the_final_request_id(self):
        _s, doc = self.client.submit(make_recipe())
        self.client.wait(doc["job_id"])
        req = self.fx.trainer.jobs[0].dev_request
        self.assertEqual((req.phase, req.final_request_id, req.arm, req.adapter, req.split), (Phase.FINAL, "f-0001", None, None, Split.DEV))


if __name__ == "__main__":
    unittest.main()
