"""score_session (owner 1): serve, score, always tear down, report cost."""

import tempfile
import unittest
from pathlib import Path

from nightshift.lab.contracts import (
    AccessDenied,
    Arm,
    BudgetExhausted,
    GpuError,
    GpuUsage,
    Phase,
    ScorerError,
    Split,
    Verifier,
)
from nightshift.lab.score_session import PROMPT_TOKEN_ALLOWANCE, score_adapter, score_many, serve_spec_for
from nightshift.lab.scorer import ScoreService
from tests.support.lab_fake_generator import FakeGenerator, FakeModelServer, FakeVerifierBackend
from tests.test_lab_scorer import BASE, EVO, HELD, PARAMS, Lab, adapter
from nightshift.lab.contracts import BaseModelRef, ScoreRequest


class SessionCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.lab = Lab(Path(self.tmp.name))
        self.guard_calls = []
        self.ev = FakeVerifierBackend(Verifier.IFEVALG)
        self.svc = ScoreService(self.lab.layout, self.lab.manifest, {Verifier.IFEVALG: self.ev, Verifier.IFBENCH_UPSTREAM: FakeVerifierBackend(Verifier.IFBENCH_UPSTREAM)},
                                guard=self.guard_calls.append, clock=lambda: 1_790_000_000.0)
        self.server = FakeModelServer()
        self.gens = []
        def factory(endpoint):
            g = FakeGenerator()
            g.endpoint = endpoint
            self.gens.append(g)
            return g
        self.factory = factory

    def request(self, **kw):
        base = dict(request_id="sc-0001", phase=Phase.SEARCH, assignment_id=EVO, split=Split.SELECTION, trial_id="t-0001", arm=Arm.V0, base=BASE,
                    adapter=adapter(), seeds=(0, 1), params=PARAMS, scorer_version=self.svc.scorer_version, reason=None, final_request_id=None)
        base.update(kw)
        return ScoreRequest(**base)


class ScoreAdapterTests(SessionCase):
    def test_serves_base_and_adapter_scores_with_the_adapter_name_and_closes_once(self):
        meter = object()
        rec, usage = score_adapter(self.request(), self.server, self.svc, self.factory, meter=meter)
        spec = self.server.specs[0]
        self.assertEqual(spec.base, BASE)
        self.assertEqual(spec.adapters, (adapter(),))
        self.assertEqual(spec.trial_id, "t-0001")
        self.assertEqual(spec.max_model_len, PARAMS.max_tokens + PROMPT_TOKEN_ALLOWANCE)
        self.assertIs(self.server.meters[0], meter)
        self.assertEqual(self.server.sessions[0].close_calls, 1)
        self.assertEqual(usage, self.server.usage)
        self.assertEqual(self.gens[0].models_seen, ["lora-" + adapter().sha256[:8]] * 2)
        self.assertEqual(rec.adapter_sha256, adapter().sha256)
        self.assertEqual(rec.score_id, "sc-0001")

    def test_stock_model_is_served_without_adapters_and_scored_under_the_base_name(self):
        rec, usage = score_adapter(self.request(arm=Arm.STOCK, adapter=None, trial_id=None), self.server, self.svc, self.factory)
        self.assertEqual(self.server.specs[0].adapters, ())
        self.assertIsNone(self.server.specs[0].trial_id)
        self.assertEqual(set(self.gens[0].models_seen), {"glimmer-base"})
        self.assertIsNone(rec.adapter_sha256)

    def test_a_scorer_failure_still_closes_the_session_and_carries_the_cost(self):
        self.gens.clear()
        def failing_factory(endpoint):
            return FakeGenerator(errors={"a-evo-1-s0": "timeout"})
        with self.assertRaises(ScorerError) as cm:
            score_adapter(self.request(), self.server, self.svc, failing_factory)
        self.assertEqual(self.server.sessions[0].close_calls, 1)
        self.assertEqual(cm.exception.gpu_usage, self.server.usage)
        self.assertFalse(self.lab.layout.score_record_path("sc-0001").exists())

    def test_an_interrupt_still_closes_the_session_and_carries_the_cost(self):
        def interrupted(model, prompt, params, seed):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt) as cm:
            score_adapter(self.request(), self.server, self.svc, lambda endpoint: FakeGenerator(responses=interrupted))
        self.assertEqual(self.server.sessions[0].close_calls, 1)
        self.assertEqual(cm.exception.gpu_usage, self.server.usage)  # billed, so the caller can still record lab.score_cost

    def test_a_refused_request_never_rents_a_gpu(self):
        with self.assertRaises(AccessDenied):
            score_adapter(self.request(assignment_id=HELD), self.server, self.svc, self.factory)
        self.assertEqual(self.server.specs, [])
        def deny(request):
            raise AccessDenied("search not closed")
        svc = ScoreService(self.lab.layout, self.lab.manifest, {Verifier.IFEVALG: self.ev}, guard=deny, clock=lambda: 1.0)
        with self.assertRaises(AccessDenied):
            score_adapter(self.request(scorer_version=svc.scorer_version), self.server, svc, self.factory)
        self.assertEqual(self.server.specs, [])
        score_adapter(self.request(), self.server, self.svc, self.factory)
        with self.assertRaises(ScorerError):  # a score is not re-rolled: refused before the second rental
            score_adapter(self.request(), self.server, self.svc, self.factory)
        self.assertEqual(len(self.server.specs), 1)

    def test_serve_failure_propagates_and_scores_nothing(self):
        for err in (BudgetExhausted("gpu_minutes", 10, 10), GpuError("no capacity")):
            server = FakeModelServer(fail=err)
            with self.assertRaises(type(err)):
                score_adapter(self.request(), server, self.svc, self.factory)
        self.assertEqual(self.gens, [])

    def test_close_failure_after_a_scorer_failure_keeps_the_original_error(self):
        class BadClose(FakeModelServer):
            def serve(inner, spec, meter=None):
                session = super().serve(spec, meter)
                def close():
                    session.close_calls += 1
                    raise GpuError("delete failed")
                session.close = close
                return session
        def failing_factory(endpoint):
            return FakeGenerator(raise_systemic="endpoint died")
        server = BadClose()
        with self.assertRaises(ScorerError) as cm:
            score_adapter(self.request(), server, self.svc, failing_factory)
        self.assertIn("endpoint died", str(cm.exception))
        self.assertTrue(any("delete failed" in n for n in getattr(cm.exception, "__notes__", [])))
        self.assertEqual(server.sessions[0].close_calls, 1)

    def test_close_failure_on_success_is_not_swallowed(self):
        class BadClose(FakeModelServer):
            def serve(inner, spec, meter=None):
                session = super().serve(spec, meter)
                def close():
                    raise GpuError("delete failed")
                session.close = close
                return session
        with self.assertRaises(GpuError):
            score_adapter(self.request(), BadClose(), self.svc, self.factory)


class ScoreManyTests(SessionCase):
    def test_stock_and_adapters_share_one_server(self):
        other = adapter("3")
        reqs = [
            self.request(request_id="sc-1", arm=Arm.STOCK, adapter=None, trial_id=None),
            self.request(request_id="sc-2", adapter=adapter()),
            self.request(request_id="sc-3", adapter=other, arm=Arm.EVOLVED),
            self.request(request_id="sc-4", adapter=adapter(), split=Split.DEV),
        ]
        records, usage = score_many(reqs, self.server, self.svc, self.factory)
        self.assertEqual(len(self.server.specs), 1)
        self.assertEqual(self.server.sessions[0].close_calls, 1)
        self.assertEqual([a.sha256 for a in self.server.specs[0].adapters], [adapter().sha256, other.sha256])
        self.assertEqual([r.score_id for r in records], ["sc-1", "sc-2", "sc-3", "sc-4"])
        self.assertEqual(len(self.gens), 1)
        models = {c[0] for c in self.gens[0].calls}
        self.assertEqual(models, {"glimmer-base", "lora-" + adapter().sha256[:8], "lora-" + other.sha256[:8]})
        self.assertIsNone(self.server.specs[0].trial_id)  # requests of different trials

    def test_a_batch_scoring_one_logical_score_twice_is_refused_before_a_gpu_is_rented(self):
        reqs = [self.request(request_id="sc-1"), self.request(request_id="sc-2")]  # same adapter, assignment, phase, split; different ids
        with self.assertRaises(ScorerError):
            score_many(reqs, self.server, self.svc, self.factory)
        self.assertEqual(self.server.specs, [])
        self.assertFalse(self.lab.layout.score_record_path("sc-1").exists())

    def test_a_second_request_for_a_scored_logical_score_is_refused_by_the_scorer(self):
        score_adapter(self.request(request_id="sc-1"), self.server, self.svc, self.factory)
        with self.assertRaises(ScorerError):  # the ledger has not been appended to, so the guard cannot see sc-1
            score_adapter(self.request(request_id="sc-2"), self.server, self.svc, self.factory)
        self.assertFalse(self.lab.layout.score_record_path("sc-2").exists())

    def test_failure_in_the_middle_closes_and_keeps_earlier_records(self):
        class Flaky(FakeGenerator):
            def generate(inner, model, prompts, params, seed):
                if model.startswith("lora-") and model.endswith(other.sha256[:8]):
                    raise ScorerError("endpoint died")
                return super().generate(model, prompts, params, seed)
        other = adapter("3")
        reqs = [self.request(request_id="sc-1"), self.request(request_id="sc-2", adapter=other)]
        with self.assertRaises(ScorerError):
            score_many(reqs, self.server, self.svc, lambda ep: Flaky())
        self.assertEqual(self.server.sessions[0].close_calls, 1)
        self.assertTrue(self.lab.layout.score_record_path("sc-1").exists())
        self.assertFalse(self.lab.layout.score_record_path("sc-2").exists())

    def test_requests_must_share_a_base_and_must_not_be_empty(self):
        other_base = BaseModelRef("meta-models/Muse-Glimmer-30B", "c" * 40)
        with self.assertRaises(ScorerError):
            serve_spec_for([self.request(), self.request(request_id="sc-2", base=other_base)])
        with self.assertRaises(ScorerError):
            score_many([], self.server, self.svc, self.factory)

    def test_max_model_len_uses_the_largest_max_tokens(self):
        from dataclasses import replace
        big = replace(PARAMS, max_tokens=6000)
        spec = serve_spec_for([self.request(), self.request(request_id="sc-2", params=big)])
        self.assertEqual(spec.max_model_len, 6000 + PROMPT_TOKEN_ALLOWANCE)
        self.assertEqual(serve_spec_for([self.request()], max_model_len=9999).max_model_len, 9999)


if __name__ == "__main__":
    unittest.main()
