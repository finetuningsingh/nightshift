import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from nightshift.lab import runtime_coral as rc
from nightshift.lab.contracts import (
    AccessDenied,
    BudgetSpec,
    ContractError,
    ExitReason,
    FinalComparisonRefused,
    JobStatus,
    LabKind,
    Phase,
    RuntimeFailure,
    Split,
    TrialExecutor,
    TrialRecord,
    TrialStatus,
    read_record,
)
from tests.support.lab_fake_runtime import (
    FakeRuntime,
    LocalRemote,
    TrialFixture,
    Upstream,
    install_fake_coral,
    make_recipe,
    post_json,
)

API_KEY = "meta-secret-key-0123456789"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class TrialCase(unittest.TestCase):
    final = False
    budget = None

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name).resolve()
        self.fx = TrialFixture(self.tmp, final=self.final, budget=self.budget)
        self.layout = self.fx.layout

    def runner(self, runtime=None, **kw):
        self.runtime = runtime or FakeRuntime(self.layout, clock=self.fx.clock)
        return self.fx.runner(self.runtime, **kw)

    def kinds(self, ledger):
        return ledger.kinds()

    def master_kinds(self):
        return self.fx.master.kinds()

    def all_text(self):
        """Everything a trial leaves behind that a researcher or an auditor could read: ledgers, records, trace."""
        parts = []
        for led in [self.fx.master, *self.fx.trial_ledgers.values()]:
            parts += [json.dumps(r.to_dict()) for _k, _a, r in led.events]
        for p in self.layout.root.rglob("*.json"):
            parts.append(p.read_text())
        return "\n".join(parts)


class HappyPathTests(TrialCase):
    def test_a_search_trial_end_to_end(self):
        runner = self.runner()
        self.assertIsInstance(runner, TrialExecutor)
        spec = self.fx.spec("t-0001")
        rec = runner.run(spec)
        self.assertEqual(rec.status, TrialStatus.COMPLETED)
        self.assertIsNone(rec.failure)
        # the submission is the runner's: control-host dev score, true recipe hash
        self.assertEqual(rec.submission.adapter_id, "ad-0001")
        self.assertAlmostEqual(rec.submission.dev_score, 0.45)
        self.assertEqual(rec.adapter.adapter_id, "ad-0001")
        self.assertEqual((rec.dev_score_id, rec.selection_score_id), ("t-0001-dev1", "t-0001-sel"))
        self.assertEqual(rec.initial_context_sha256, self.runtime.handles[0].initial_context_sha256)
        self.assertIsNotNone(rec.trace)
        self.assertIsNotNone(rec.notes)
        self.assertEqual(rec.environment["runtime_exit"], "finished")
        self.assertEqual(rec.environment["isolation"], "fake")
        self.assertEqual(len(rec.environment["pack_tree_sha256"]), 64)
        self.assertEqual(rec.budget_used.gpu_minutes, 5.0)
        self.assertGreater(rec.budget_used.gpu_usd, 0)
        # master ledger: exactly the lab-level story, in order
        self.assertEqual(
            self.master_kinds(),
            [LabKind.TRIAL_STARTED, LabKind.SCORE_RECORDED, LabKind.CONTAMINATION_SCANNED, LabKind.SCORE_RECORDED, LabKind.SCORE_COST, LabKind.TRIAL_FINISHED],
        )
        started, dev, scan, sel, cost, finished = [r for _k, _a, r in self.fx.master.events]
        self.assertEqual(started, spec)
        self.assertEqual((dev.score_id, dev.split), ("t-0001-dev1", Split.DEV))
        self.assertTrue(scan.passed)
        self.assertEqual((sel.score_id, sel.split, sel.assignment_id, sel.phase), ("t-0001-sel", Split.SELECTION, "a-evo-1", Phase.SEARCH))
        self.assertEqual((cost.score_id, cost.gpu.minutes), ("t-0001-sel", 3.0))
        self.assertEqual(finished, rec)
        # trial ledger: the jobs and the budget, sealed with an anchor that the record carries
        tl = self.fx.trial_ledgers["t-0001"]
        self.assertIn(LabKind.GPU_JOB_STARTED, tl.kinds())
        self.assertIn(LabKind.GPU_JOB_FINISHED, tl.kinds())
        self.assertIn(LabKind.BUDGET_CHARGED, tl.kinds())
        self.assertTrue(tl.sealed and tl.closed)
        self.assertEqual(rec.trial_ledger_anchor.seq, len(tl.events))
        # files
        self.assertEqual(read_record(self.layout.trial_record_path("t-0001"), TrialRecord), rec)
        self.assertTrue(self.layout.trial_spec_path("t-0001").is_file())
        self.assertTrue((self.layout.trial_pack_dir("t-0001") / "task.yaml").is_file())
        # what the scorer was asked
        (req,) = self.fx.scorer.requests
        self.assertEqual((req.split, req.phase, req.arm, req.assignment_id, req.trial_id), (Split.SELECTION, Phase.SEARCH, None, "a-evo-1", "t-0001"))
        self.assertEqual((req.seeds, req.params, req.scorer_version, req.base, req.adapter), (spec.score_seeds, spec.score_params, spec.scorer_version, spec.base, rec.adapter))
        # the serving session was closed
        self.assertTrue(all(s.closed for s in self.fx.model_server.sessions))
        # the scan saw the job's own files
        (call,) = self.fx.scanner.calls
        self.assertEqual((call["trial_id"], call["job_id"]), ("t-0001", "t-0001-j1"))
        self.assertEqual(call["recipe"].path.startswith("trials/t-0001/recipe/"), True)
        # the runtime was started with loopback endpoints and the trial token, which was registered as a secret
        ctx = self.runtime.contexts[0]
        self.assertTrue(ctx.llm_base_url.startswith("http://127.0.0.1:"))
        self.assertEqual(ctx.llm_base_url.split(":")[-1].split("/")[0], ctx.control_endpoint.split(":")[-1].split("/")[0])
        self.assertIn(ctx.llm_token, self.fx.registered)
        self.assertIn(API_KEY, self.fx.registered)
        self.assertTrue(self.runtime.handles[0].closed)

    def test_nothing_secret_is_written_anywhere(self):
        rec = self.runner().run(self.fx.spec("t-0001"))
        token = self.runtime.contexts[0].llm_token
        text = self.all_text()
        self.assertNotIn(token, text)
        self.assertNotIn(API_KEY, text)
        self.assertEqual(rec.status, TrialStatus.COMPLETED)

    def test_the_best_control_host_dev_score_wins_not_the_researchers_claim(self):
        self.fx.trainer.score_for = lambda n: {1: 0.9, 2: 0.2}[n]

        def script(ctx, client, handle):
            adapters = []
            for _ in range(2):
                _s, doc = client.submit(make_recipe())
                adapters.append(client.wait(doc["job_id"])["adapter_id"])
            # the researcher claims the worse adapter is the better one, and writes a flattering dev score
            handle.commit_submission(adapters[1], dev_score=0.99, recipe_sha256="f" * 64, at=1_793_000_100)
            handle.commit_submission(adapters[0], dev_score=0.0, recipe_sha256="e" * 64, at=1_793_000_200)

        rec = self.runner(FakeRuntime(self.layout, script, clock=self.fx.clock)).run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.COMPLETED)
        self.assertEqual(rec.submission.adapter_id, "ad-0001")
        self.assertAlmostEqual(rec.submission.dev_score, 0.9)
        self.assertNotEqual(rec.submission.recipe_sha256, "e" * 64)  # replaced by the hash of the recipe that really produced it
        self.assertEqual(rec.environment["submission_recipe_mismatch"], "true")

    def test_ties_go_to_the_earliest_commit_time(self):
        self.fx.trainer.score_for = lambda n: 0.5

        def script(ctx, client, handle):
            adapters = []
            for _ in range(3):
                _s, doc = client.submit(make_recipe())
                adapters.append(client.wait(doc["job_id"])["adapter_id"])
            handle.commit_submission(adapters[2], at=1_793_000_300)
            handle.commit_submission(adapters[1], at=1_793_000_100)  # earliest
            handle.commit_submission(adapters[0], at=1_793_000_200)

        rec = self.runner(FakeRuntime(self.layout, script, clock=self.fx.clock)).run(self.fx.spec("t-0001"))
        self.assertEqual(rec.submission.adapter_id, "ad-0002")


class AccountingCloseTests(TrialCase):
    def test_the_metering_proxy_stops_booking_before_the_usage_is_read_and_the_ledger_sealed(self):
        from unittest import mock

        from tests.support.lab_fake_runtime import FakeLedger
        from nightshift.lab.metering_proxy import MeteringProxy

        order = []
        real_close, real_seal = MeteringProxy.close, FakeLedger.seal
        with mock.patch.object(MeteringProxy, "close", lambda self: (order.append("proxy"), real_close(self))[1]), mock.patch.object(
            FakeLedger, "seal", lambda self: (order.append("seal"), real_seal(self))[1]
        ):
            rec = self.runner().run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.COMPLETED)
        self.assertEqual(order[:2], ["proxy", "seal"])


class SubmissionLineageTests(TrialCase):
    def test_a_submission_naming_another_trials_adapter_is_ignored(self):
        first = self.runner().run(self.fx.spec("t-0001"))
        foreign = first.adapter.adapter_id

        def script(ctx, client, handle):
            handle.commit_submission(foreign, dev_score=1.0)  # exists on disk, scored well: but not produced by THIS trial

        rec = self.runner(FakeRuntime(self.layout, script, clock=self.fx.clock)).run(self.fx.spec("t-0002"))
        self.assertEqual(rec.status, TrialStatus.NO_SUBMISSION)
        self.assertIsNone(rec.adapter)
        self.assertEqual(rec.environment["ignored_submissions"], foreign)
        self.assertEqual(len(self.fx.scorer.requests), 1)  # only the first trial was ever scored

    def test_own_adapter_wins_over_a_foreign_one_in_the_same_trial(self):
        first = self.runner().run(self.fx.spec("t-0001"))

        def script(ctx, client, handle):
            _s, doc = client.submit(make_recipe())
            own = client.wait(doc["job_id"])["adapter_id"]
            handle.commit_submission(first.adapter.adapter_id, at=1_793_000_100)
            handle.commit_submission(own, at=1_793_000_200)

        rec = self.runner(FakeRuntime(self.layout, script, clock=self.fx.clock)).run(self.fx.spec("t-0002"))
        self.assertEqual(rec.status, TrialStatus.COMPLETED)
        self.assertNotEqual(rec.submission.adapter_id, first.adapter.adapter_id)


class StatusMappingTests(TrialCase):
    def test_no_attempts_at_all_finished_is_no_submission(self):
        rec = self.runner(FakeRuntime(self.layout, lambda ctx, c, h: None, clock=self.fx.clock)).run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.NO_SUBMISSION)
        self.assertIsNotNone(rec.failure)
        self.assertIsNone(rec.selection_score_id)
        self.assertEqual(self.fx.scorer.requests, [])

    def test_crashed_with_nothing_submitted_is_failed(self):
        def script(ctx, client, handle):
            raise RuntimeError("opencode died")

        rec = self.runner(FakeRuntime(self.layout, script, clock=self.fx.clock)).run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.FAILED)
        self.assertEqual(rec.environment["runtime_exit"], "crashed")
        self.assertIn("crashed", rec.failure)

    def test_crashed_after_a_submission_is_scored_normally_and_the_crash_is_noted(self):
        def script(ctx, client, handle):
            FakeRuntime.default_script(ctx, client, handle)
            raise RuntimeError("opencode died after submitting")

        rec = self.runner(FakeRuntime(self.layout, script, clock=self.fx.clock)).run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.COMPLETED)
        self.assertEqual(rec.environment["runtime_exit"], "crashed")

    def test_crashed_with_only_foreign_submissions_is_no_submission(self):
        def script(ctx, client, handle):
            handle.commit_submission("ad-7777")
            raise RuntimeError("died")

        rec = self.runner(FakeRuntime(self.layout, script, clock=self.fx.clock)).run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.NO_SUBMISSION)

    def test_failed_jobs_leave_no_adapter_to_submit(self):
        self.fx.trainer.fail_status = JobStatus.FAILED
        rec = self.runner().run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.NO_SUBMISSION)
        self.assertEqual(self.fx.trial_ledgers["t-0001"].of(LabKind.GPU_JOB_FINISHED)[0].status, JobStatus.FAILED)

    def test_runtime_that_cannot_start_is_failed_and_cleaned_up(self):
        runtime = FakeRuntime(self.layout, start_error=RuntimeFailure("ssh: connection refused"), clock=self.fx.clock)
        rec = self.runner(runtime).run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.FAILED)
        self.assertIn("failed to start", rec.failure)
        self.assertIsNone(rec.initial_context_sha256)
        self.assertEqual(self.master_kinds(), [LabKind.TRIAL_STARTED, LabKind.TRIAL_FINISHED])
        self.assertTrue(self.fx.trial_ledgers["t-0001"].closed)

    def test_pack_checkout_failure_is_failed_before_anything_is_launched(self):
        self.fx.packs.fail = True
        runtime = FakeRuntime(self.layout, clock=self.fx.clock)
        rec = self.runner(runtime).run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.FAILED)
        self.assertIn("checkout", rec.failure)
        self.assertEqual(runtime.contexts, [])

    def test_harvest_failure_is_failed(self):
        runtime = FakeRuntime(self.layout, clock=self.fx.clock)
        runtime_start = runtime.start

        def start(ctx):
            h = runtime_start(ctx)
            h.collect_error = RuntimeFailure("nothing to harvest")
            return h

        runtime.start = start
        rec = self.runner(runtime).run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.FAILED)
        self.assertIn("harvest", rec.failure)
        self.assertEqual(self.fx.scorer.requests, [])


class BudgetStopTests(TrialCase):
    budget = BudgetSpec(researcher_tokens=6000, gpu_minutes=120.0, dollars=25.0, wall_seconds=7200.0)

    def llm_script(self, handle_wait=True):
        upstream = self.upstream

        def script(ctx, client, handle):
            FakeRuntime.default_script(ctx, client, handle)  # a scored adapter exists first
            url = ctx.llm_base_url + "/chat/completions"
            for _ in range(20):
                status, _h, _d = post_json(url, ctx.llm_token, {"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2500})
                if status == 429:
                    break
            handle.stop_event.wait(20)  # a researcher that never stops by itself

        return script

    def test_token_cap_stops_the_runtime_and_the_scored_adapter_counts(self):
        self.upstream = Upstream().start()
        self.addCleanup(self.upstream.stop)
        self.upstream.queue.extend([Upstream.Reply(usage=(300, 200, 0))] * 30)
        runtime = FakeRuntime(self.layout, self.llm_script(), clock=self.fx.clock)
        rec = self.runner(runtime, upstream_base_url=self.upstream.base_url).run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.BUDGET_EXHAUSTED)
        self.assertEqual(runtime.handles[0].stopped_with, [ExitReason.STOPPED_BUDGET])
        self.assertEqual(rec.environment["runtime_exit"], "stopped_budget")
        self.assertIsNotNone(rec.adapter)
        self.assertEqual(rec.selection_score_id, "t-0001-sel")
        self.assertLessEqual(rec.budget_used.researcher_tokens, 6000)
        self.assertGreater(len(self.upstream.requests), 0)  # calls before the cap were forwarded with the Meta key
        self.assertEqual(self.upstream.requests[0]["headers"]["authorization"], "Bearer " + API_KEY)

    def test_a_researcher_that_exits_after_a_refused_reservation_still_ends_budget_exhausted(self):
        self.upstream = Upstream().start()
        self.addCleanup(self.upstream.stop)
        self.upstream.queue.extend([Upstream.Reply(usage=(300, 200, 0))] * 30)
        refused = threading.Event()

        def script(ctx, client, handle):
            FakeRuntime.default_script(ctx, client, handle)
            url = ctx.llm_base_url + "/chat/completions"
            for _ in range(20):
                status, _h, _d = post_json(url, ctx.llm_token, {"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2500})
                if status == 429:
                    break
            refused.set()  # the researcher exits right after the refusal, without waiting to be stopped

        runtime = FakeRuntime(self.layout, script, clock=self.fx.clock)
        orig = runtime.start

        def start(ctx):
            h = orig(ctx)
            real_poll = h.poll

            def poll():  # the monitor's first look comes after the refusal, so its budget check never runs
                refused.wait(20)
                return real_poll()

            h.poll = poll
            return h

        runtime.start = start
        rec = self.runner(runtime, upstream_base_url=self.upstream.base_url).run(self.fx.spec("t-0001"))
        self.assertLess(rec.budget_used.researcher_tokens, 6000)
        self.assertEqual(rec.status, TrialStatus.BUDGET_EXHAUSTED)
        self.assertIsNotNone(rec.adapter)

    def test_wall_time_stops_the_runtime(self):
        clock = self.fx.clock

        def script(ctx, client, handle):
            FakeRuntime.default_script(ctx, client, handle)
            clock.advance(10_000)  # past the 7200 s wall budget
            handle.stop_event.wait(20)

        runtime = FakeRuntime(self.layout, script, clock=clock)
        rec = self.runner(runtime).run(self.fx.spec("t-0001"))
        self.assertEqual(runtime.handles[0].stopped_with, [ExitReason.STOPPED_WALL])
        self.assertEqual(rec.status, TrialStatus.BUDGET_EXHAUSTED)

    def test_budget_stop_without_a_scored_adapter_is_no_submission(self):
        clock = self.fx.clock

        def script(ctx, client, handle):
            clock.advance(10_000)
            handle.stop_event.wait(20)

        runtime = FakeRuntime(self.layout, script, clock=clock)
        rec = self.runner(runtime).run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.NO_SUBMISSION)
        self.assertEqual(runtime.handles[0].stopped_with, [ExitReason.STOPPED_WALL])

    def test_gpu_budget_ended_by_a_job_marks_the_trial_budget_exhausted(self):
        budget = BudgetSpec(researcher_tokens=2_000_000, gpu_minutes=7.0, dollars=25.0, wall_seconds=7200.0)
        self.tearDown_fx = TrialFixture(self.tmp / "b", budget=budget)
        fx = self.tearDown_fx
        runtime = FakeRuntime(fx.layout, clock=fx.clock)

        def script(ctx, client, handle):
            adapters = []
            for _ in range(3):  # 5 minutes each against 7 available: job 1 ok, job 2 runs out of GPU minutes
                _s, doc = client.submit(make_recipe())
                if doc.get("status") != "queued":
                    break
                res = client.wait(doc["job_id"])
                if res["state"] == "ok":
                    adapters.append(res["adapter_id"])
            handle.commit_submission(adapters[0])

        runtime.script = script
        rec = fx.runner(runtime).run(fx.spec("t-0001", budget=budget))
        self.assertEqual(rec.status, TrialStatus.BUDGET_EXHAUSTED)
        self.assertEqual(rec.adapter.adapter_id, "ad-0001")


class ContaminationTests(TrialCase):
    def test_a_hit_stops_the_runtime_at_once_and_the_adapter_is_never_scored(self):
        self.fx.trainer.train_text = "Please answer. SELECTION-CANARY appears in this training row."
        def script(ctx, client, handle):
            client.submit(make_recipe())
            handle.commit_submission("ad-0001")  # the researcher submits it anyway
            handle.stop_event.wait(20)

        runtime = FakeRuntime(self.layout, script, clock=self.fx.clock)
        rec = self.runner(runtime).run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.CONTAMINATED)
        self.assertEqual(runtime.handles[0].stopped_with, [ExitReason.STOPPED_CONTAMINATED])
        self.assertEqual(rec.environment["runtime_exit"], "stopped_contaminated")
        self.assertIsNone(rec.adapter)
        self.assertIsNone(rec.selection_score_id)
        self.assertEqual(self.fx.scorer.requests, [])  # not on selection, not on anything
        self.assertEqual(self.fx.model_server.sessions, [])  # no scoring rental was even rented
        scan = [r for k, _a, r in self.fx.master.events if k == LabKind.CONTAMINATION_SCANNED]
        self.assertEqual([s.passed for s in scan], [False])
        self.assertEqual(scan[0].n_exact_hits, 1)
        self.assertEqual(self.master_kinds()[-1], LabKind.TRIAL_FINISHED)

    def test_a_scan_that_cannot_run_withholds_the_adapter(self):
        self.fx.scanner.raise_exc = OSError("disk error")
        rec = self.runner().run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.NO_SUBMISSION)
        self.assertEqual(self.fx.scorer.requests, [])
        self.assertIn("disk error", rec.environment["scan_error"])

    def test_clean_data_passes_and_every_job_is_scanned(self):
        def script(ctx, client, handle):
            for _ in range(2):
                _s, doc = client.submit(make_recipe())
                res = client.wait(doc["job_id"])
            handle.commit_submission(res["adapter_id"])

        rec = self.runner(FakeRuntime(self.layout, script, clock=self.fx.clock)).run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.COMPLETED)
        self.assertEqual([c["job_id"] for c in self.fx.scanner.calls], ["t-0001-j1", "t-0001-j2"])


class ScoringFailureTests(TrialCase):
    def test_one_scorer_failure_is_recorded_and_retried_once_with_the_same_seeds(self):
        self.fx.scorer.fail_next = 1
        rec = self.runner().run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.COMPLETED)
        self.assertEqual(rec.selection_score_id, "t-0001-sel2")
        first, second = self.fx.scorer.requests
        self.assertEqual((first.request_id, second.request_id), ("t-0001-sel", "t-0001-sel2"))
        self.assertEqual((first.seeds, first.params, first.adapter), (second.seeds, second.params, second.adapter))
        fails = [r for k, _a, r in self.fx.master.events if k == LabKind.SCORE_FAILED]
        self.assertEqual(len(fails), 1)
        self.assertEqual((fails[0].request_id, fails[0].result_revealed, fails[0].split, fails[0].adapter_sha256), ("t-0001-sel", False, Split.SELECTION, rec.adapter.sha256))
        self.assertIsNone(rec.failure)
        self.assertTrue(all(s.closed for s in self.fx.model_server.sessions))

    def test_two_failures_leave_a_failed_trial_with_the_adapter_kept(self):
        self.fx.scorer.fail_next = 2
        rec = self.runner().run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.FAILED)
        self.assertIsNotNone(rec.adapter)
        self.assertEqual(rec.submission.adapter_id, rec.adapter.adapter_id)
        self.assertIsNone(rec.selection_score_id)
        self.assertIn("twice", rec.failure)
        self.assertEqual(len([1 for k, _a, _r in self.fx.master.events if k == LabKind.SCORE_FAILED]), 2)
        self.assertEqual(len(self.fx.scorer.requests), 2)  # exactly one retry, never a third attempt
        self.assertTrue(all(s.closed for s in self.fx.model_server.sessions))

    def test_a_guard_refusal_is_not_retried_and_not_recorded_as_an_infrastructure_failure(self):
        def guard(request):
            raise AccessDenied("selection already scored for this adapter")

        self.fx.scorer.guard = guard
        rec = self.runner().run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.FAILED)
        self.assertIn("refused", rec.failure)
        self.assertEqual(len(self.fx.scorer.requests), 1)
        self.assertNotIn(LabKind.SCORE_FAILED, self.master_kinds())  # it was never a scoring attempt that "failed without a number"

    def test_adapter_that_fails_verification_is_not_scored(self):
        def script(ctx, client, handle):
            FakeRuntime.default_script(ctx, client, handle)
            (self.layout.adapter_dir("ad-0001") / "adapter_model.safetensors").write_bytes(b"tampered after the job")

        rec = self.runner(FakeRuntime(self.layout, script, clock=self.fx.clock)).run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.FAILED)
        self.assertIn("verification", rec.failure)
        self.assertEqual(self.fx.scorer.requests, [])
        self.assertIsNone(rec.adapter)


class FinalPhaseTests(TrialCase):
    final = True

    def test_final_trial_scores_heldout_selection_and_the_test_split(self):
        spec = self.fx.final_spec("f-evolved-0")
        rec = self.runner().run(spec)
        self.assertEqual(rec.status, TrialStatus.COMPLETED)
        sel, test = self.fx.scorer.requests
        self.assertEqual((sel.split, sel.assignment_id, sel.request_id), (Split.SELECTION, "a-held-1", "f-evolved-0-sel"))
        self.assertEqual((test.split, test.assignment_id, test.request_id), (Split.TEST, "ifbench_test", "f-evolved-0-test"))
        for r in (sel, test):
            self.assertEqual((r.phase, r.arm.value, r.final_request_id, r.trial_id), (Phase.FINAL, "evolved", "f-0001", "f-evolved-0"))
        records = [r for k, _a, r in self.fx.master.events if k == LabKind.SCORE_RECORDED]
        self.assertEqual([r.split for r in records], [Split.DEV, Split.SELECTION, Split.TEST])
        self.assertEqual(rec.selection_score_id, "f-evolved-0-sel")
        # the dev score of a final trial is a final-phase dev request
        self.assertEqual(records[0].final_request_id, "f-0001")

    def test_a_refused_test_score_ends_the_trial_failed_without_retry(self):
        def guard(request):
            if request.split is Split.TEST:
                raise FinalComparisonRefused("no FinalRequest covers this trial")

        self.fx.scorer.guard = guard
        rec = self.runner().run(self.fx.final_spec("f-evolved-1"))
        self.assertEqual(rec.status, TrialStatus.FAILED)
        self.assertEqual(rec.selection_score_id, "f-evolved-1-sel")  # the held-out score was produced; the test score was refused
        self.assertEqual(len(self.fx.scorer.requests), 2)
        self.assertNotIn(LabKind.SCORE_FAILED, self.master_kinds())

    def test_a_final_trial_needs_a_matching_final_request(self):
        spec = self.fx.final_spec("f-evolved-0")
        runner = self.runner()
        good = self.fx.state.finals
        for bad in ([], [r for r in good if r.arm.value == "v0"]):  # none at all; only the other arm's
            self.fx.state.finals = bad
            with self.assertRaises(ContractError):
                runner.run(spec)
        import dataclasses
        from nightshift.lab.contracts import Split

        self.fx.state.finals = [dataclasses.replace(r, splits=(Split.SELECTION,)) for r in good]  # the test split is not unlocked
        with self.assertRaises(ContractError):
            runner.run(spec)
        self.fx.state.finals = [dataclasses.replace(r, analysis_sha256="9" * 64) for r in good]  # bound to another analysis
        with self.assertRaises(ContractError):
            runner.run(spec)
        from nightshift.lab.contracts import PackRef

        self.fx.state.finals = [dataclasses.replace(r, pack=PackRef("packs/pack.git", "d" * 40)) for r in good]  # another pack commit
        with self.assertRaises(ContractError):
            runner.run(spec)
        self.assertEqual(self.master_kinds(), [])
        self.fx.state.finals = good
        self.assertEqual(runner.run(spec).status, TrialStatus.COMPLETED)
        v0 = runner.run(self.fx.final_spec("f-v0-0"))
        self.assertEqual(v0.spec.final_request_id, "f-0002")

    def test_a_final_trial_must_be_planned(self):
        spec = self.fx.final_spec("f-evolved-0")
        import dataclasses

        wrong_seed = dataclasses.replace(spec, seed=spec.seed + 1)
        with self.assertRaises(ContractError):
            self.runner().run(wrong_seed)
        self.assertEqual(self.master_kinds(), [])


class RefusalTests(TrialCase):
    def assertRefusedBeforeSpending(self, spec, exc=ContractError, runner=None):
        runner = runner or self.runner()
        with self.assertRaises(exc):
            runner.run(spec)
        self.assertEqual(self.master_kinds(), [])
        self.assertFalse(self.layout.trial_dir(spec.trial_id).exists())
        self.assertEqual(self.runtime.contexts, [])
        self.assertEqual(self.fx.trial_ledgers, {})

    def test_budget_that_is_not_the_frozen_budget(self):
        self.assertRefusedBeforeSpending(self.fx.spec("t-0001", budget=BudgetSpec(1, 1.0, 1.0, 1.0)))

    def test_pack_that_was_never_committed(self):
        from nightshift.lab.contracts import PackRef

        self.assertRefusedBeforeSpending(self.fx.spec("t-0001", pack=PackRef("packs/pack.git", "c" * 40)))
        self.assertRefusedBeforeSpending(self.fx.spec("t-0002", pack_version="v9"))

    def test_scorer_version_mismatch(self):
        self.assertRefusedBeforeSpending(self.fx.spec("t-0001", scorer_version="sv1-someotherscorer"))

    def test_wrong_assignment_role_and_revisions(self):
        self.assertRefusedBeforeSpending(self.fx.spec("t-0001", assignment_id="a-held-1"))  # a search trial on a held-out assignment
        self.assertRefusedBeforeSpending(self.fx.spec("t-0002", data_revisions={"train_sha256": "9" * 64, "dev_sha256": "2" * 64, "selection_sha256": "3" * 64}))

    def test_final_phase_before_search_is_closed(self):
        # the non-final fixture has no frozen final analysis and no lab.search_closed
        spec = TrialFixture(self.tmp / "other", final=True).final_spec("f-evolved-0")
        self.assertRefusedBeforeSpending(spec)

    def test_a_trial_id_is_used_once(self):
        runner = self.runner()
        runner.run(self.fx.spec("t-0001"))
        with self.assertRaises(FileExistsError):
            runner.run(self.fx.spec("t-0001"))

    def test_the_runner_refuses_to_run_unconfigured(self):
        for missing in ({"contamination_scan": None}, {"api_key": None}, {"upstream_base_url": ""}):
            self.runtime = FakeRuntime(self.layout, clock=self.fx.clock)
            with self.assertRaises(RuntimeFailure):
                self.fx.runner(self.runtime, **missing).run(self.fx.spec("t-0001"))
        self.assertEqual(self.master_kinds(), [])

    def test_the_frozen_budget_file_is_checked_against_the_ledger(self):
        # a rewritten budget.json no longer matches the hash the ledger recorded
        self.layout.budget_path.write_text(self.layout.budget_path.read_text().replace("120.0", "999.0"))
        self.assertRefusedBeforeSpending(self.fx.spec("t-0001"))


class ConcurrencyTests(TrialCase):
    def test_two_trials_run_concurrently_against_one_shared_ledger(self):
        # each trial's researcher waits until BOTH are running, so the trials really overlap
        barrier = threading.Barrier(2, timeout=30)

        probes = {}

        def script(ctx, client, handle):
            _s, doc = client.submit(make_recipe())
            res = client.wait(doc["job_id"])
            barrier.wait()  # both trials are running, so both servers are live: probe them with the OTHER trial's token
            other = next(c for c in runtime.contexts if c.spec.trial_id != ctx.spec.trial_id).llm_token
            tid = ctx.spec.trial_id
            probes[tid] = {
                "control_other": client.request("GET", "/status", token=other)[0],
                "control_own": client.request("GET", "/status")[0],
                "proxy_other": post_json(ctx.llm_base_url + "/chat/completions", other, {"model": "m", "messages": []})[0],
                "proxy_own": post_json(ctx.llm_base_url + "/chat/completions", ctx.llm_token, {"model": "m", "messages": []})[0],
            }
            barrier.wait()
            handle.commit_submission(res["adapter_id"])

        runtime = FakeRuntime(self.layout, script, clock=self.fx.clock)
        runner = self.fx.runner(runtime)
        self.runtime = runtime
        out = {}

        def go(tid):
            out[tid] = runner.run(self.fx.spec(tid))

        threads = [threading.Thread(target=go, args=(t,)) for t in ("t-0001", "t-0002")]
        [t.start() for t in threads]
        [t.join(60) for t in threads]
        self.assertEqual({k: v.status for k, v in out.items()}, {"t-0001": TrialStatus.COMPLETED, "t-0002": TrialStatus.COMPLETED})
        a, b = out["t-0001"].adapter.adapter_id, out["t-0002"].adapter.adapter_id
        self.assertNotEqual(a, b)
        for tid, rec in out.items():
            led = self.fx.trial_ledgers[tid]
            for _k, _actor, payload in led.events:
                self.assertEqual(getattr(payload, "trial_id", tid), tid)  # no cross-talk between trial ledgers
            self.assertEqual(rec.selection_score_id, f"{tid}-sel")
        ctxs = {c.spec.trial_id: c for c in runtime.contexts}
        self.assertNotEqual(ctxs["t-0001"].llm_token, ctxs["t-0002"].llm_token)
        self.assertNotEqual(ctxs["t-0001"].llm_base_url, ctxs["t-0002"].llm_base_url)
        self.assertEqual(self.master_kinds().count(LabKind.TRIAL_STARTED), 2)
        self.assertEqual(self.master_kinds().count(LabKind.TRIAL_FINISHED), 2)
        self.assertEqual(self.master_kinds().count(LabKind.SCORE_RECORDED), 4)
        # each trial's token opens only its own control API and proxy (probed while both servers were live)
        self.assertEqual(set(probes), {"t-0001", "t-0002"})
        for tid, p in probes.items():
            self.assertEqual((p["control_other"], p["proxy_other"]), (401, 401), tid)
            self.assertEqual(p["control_own"], 200, tid)
            self.assertNotEqual(p["proxy_own"], 401, tid)  # its own token passes authentication (the request itself is rejected or forwarded)


class InternalErrorTests(TrialCase):
    def test_a_bug_inside_a_trial_still_ends_in_a_recorded_failed_trial(self):
        runner = self.runner()

        def boom(*a, **kw):
            raise ValueError(f"unexpected {API_KEY}")

        self.fx.scorer.score = boom
        self.fx.scorer.fail_next = 0
        rec = runner.run(self.fx.spec("t-0001"))
        # a scorer exception is an infrastructure failure (retried once); the key never appears in the record
        self.assertEqual(rec.status, TrialStatus.FAILED)
        self.assertNotIn(API_KEY, json.dumps(rec.to_dict()))
        self.assertNotIn(API_KEY, self.all_text())

    def test_exception_in_the_runtime_poll_is_recorded(self):
        runtime = FakeRuntime(self.layout, clock=self.fx.clock)
        orig = runtime.start

        def start(ctx):
            h = orig(ctx)
            h.poll = lambda: (_ for _ in ()).throw(RuntimeError("poll exploded"))
            return h

        runtime.start = start
        rec = self.runner(runtime).run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.FAILED)
        self.assertIn("internal error", rec.failure)
        self.assertEqual(self.master_kinds()[-1], LabKind.TRIAL_FINISHED)
        self.assertTrue(self.fx.trial_ledgers["t-0001"].closed)

    def test_an_internal_error_waits_for_the_running_training_job_before_sealing_the_trial(self):
        runtime = FakeRuntime(self.layout, clock=self.fx.clock)
        orig = runtime.start
        trainer = self.fx.trainer
        trainer.block = threading.Event()
        self.addCleanup(trainer.block.set)

        def start(ctx):
            h = orig(ctx)

            def poll():
                if trainer.entered.is_set():  # the researcher's job is on the GPU when the runner hits its bug
                    raise RuntimeError("poll exploded")
                return None

            h.poll = poll
            return h

        runtime.start = start
        runner = self.runner(runtime)
        out = {}
        t = threading.Thread(target=lambda: out.setdefault("rec", runner.run(self.fx.spec("t-0001"))), daemon=True)
        t.start()
        self.assertTrue(trainer.entered.wait(10))
        t.join(1.0)
        self.assertTrue(t.is_alive(), "the trial finished while its training job was still running")
        self.assertNotIn(LabKind.TRIAL_FINISHED, self.master_kinds())
        self.assertFalse(self.fx.trial_ledgers["t-0001"].sealed)
        trainer.block.set()
        t.join(30)
        rec = out["rec"]
        self.assertEqual(rec.status, TrialStatus.FAILED)
        self.assertIn("internal error", rec.failure)
        self.assertEqual(rec.budget_used.gpu_minutes, 5.0)  # the job's minutes are in the usage that was recorded
        tl = self.fx.trial_ledgers["t-0001"]
        self.assertEqual(rec.trial_ledger_anchor.seq, len(tl.events))  # nothing was appended after the seal
        self.assertTrue(tl.sealed and tl.closed)


    def test_a_job_that_outlives_the_drain_timeout_fails_the_trial_and_scores_nothing(self):
        trainer = self.fx.trainer
        trainer.block = threading.Event()
        self.addCleanup(trainer.block.set)

        def script(ctx, client, handle):
            client.submit(make_recipe())  # the researcher leaves without waiting for its job
            trainer.entered.wait(10)

        runtime = FakeRuntime(self.layout, script, clock=self.fx.clock)
        rec = self.runner(runtime, job_drain_timeout=0.05).run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.FAILED)
        self.assertIn("still running", rec.failure)
        self.assertEqual(rec.environment["job_still_running"], "true")
        self.assertEqual(self.fx.scorer.requests, [])
        self.assertIsNone(rec.adapter)

    def test_an_interrupt_waits_for_the_running_training_job_before_the_trial_ledger_is_closed(self):
        runtime = FakeRuntime(self.layout, clock=self.fx.clock)
        orig = runtime.start
        trainer = self.fx.trainer
        trainer.block = threading.Event()
        self.addCleanup(trainer.block.set)

        def start(ctx):
            h = orig(ctx)

            raised = []

            def poll():
                if trainer.entered.is_set() and not raised:  # one Ctrl-C, as a person sends it
                    raised.append(1)
                    raise KeyboardInterrupt()
                return None

            h.poll = poll
            return h

        runtime.start = start
        runner = self.runner(runtime)
        out = {}

        def go():
            try:
                runner.run(self.fx.spec("t-0001"))
            except BaseException as exc:  # noqa: BLE001
                out["exc"] = exc

        t = threading.Thread(target=go, daemon=True)
        t.start()
        self.assertTrue(trainer.entered.wait(10))
        t.join(1.0)
        self.assertTrue(t.is_alive(), "the interrupted trial returned while its training job was still running")
        self.assertFalse(self.fx.trial_ledgers["t-0001"].closed)
        trainer.block.set()
        t.join(30)
        self.assertIsInstance(out["exc"], KeyboardInterrupt)
        tl = self.fx.trial_ledgers["t-0001"]
        self.assertTrue(tl.closed)
        self.assertIn(LabKind.GPU_JOB_FINISHED, tl.kinds())  # the job's charges and result landed before the ledger closed


# --------------------------------------------------------------------------------------------------
class RealRuntimeIntegrationTests(TrialCase):
    """TrialRunner + CoralRuntime + the real glimmer tool + ControlApi + MeteringProxy, with a local 'researcher host' and a fake `coral`."""

    def setUp(self):
        super().setUp()
        self.coral_home = self.tmp / "coral"
        install_fake_coral(self.coral_home, {"mode": "researcher"})
        self.trials_root = self.tmp / "srv" / "trials"
        self.trials_root.mkdir(parents=True)
        self.remote = LocalRemote(self.tmp / "home")
        self.addCleanup(lambda: [t.close() for t in self.remote.tunnels])
        for aid in ("a-evo-1",):
            self.layout.train_data(aid).parent.mkdir(parents=True, exist_ok=True)
            self.layout.train_data(aid).write_text('{"key": "t1", "prompt": "p"}\n')
            self.layout.dev_data(aid).write_text('{"key": "d1", "prompt": "q"}\n')

    def test_full_trial_through_the_real_runtime_tool_and_control_api(self):
        runtime = rc.CoralRuntime(self.layout, self.remote, str(self.coral_home), {"coral_commit": "0" * 40, "opencode_version": "1.19.3"}, free_port, clock=self.fx.clock, remote_trials_root=str(self.trials_root))
        self.runtime = runtime
        rec = self.fx.runner(runtime, poll_seconds=0.05).run(self.fx.spec("t-0001"))
        self.assertEqual(rec.status, TrialStatus.COMPLETED, rec.failure)
        self.assertEqual(rec.submission.adapter_id, "ad-0001")
        self.assertEqual(rec.environment["isolation"], "shared_user_serial")
        self.assertEqual(rec.environment["runtime_exit"], "finished")
        self.assertEqual(rec.environment["grader_modified"], "false")
        self.assertEqual(rec.selection_score_id, "t-0001-sel")
        self.assertEqual(rec.budget_used.gpu_minutes, 5.0)
        # the committed submission.json is what the harvest read, and the trace is in the lab root
        self.assertTrue((self.layout.root / rec.trace.path).is_file())
        self.assertFalse((self.trials_root / "t-0001").exists())  # the researcher host's trial directory was deleted
        self.assertNotIn(API_KEY, self.all_text())


if __name__ == "__main__":
    unittest.main()
