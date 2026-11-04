"""InspectGenerator: the score's model calls as an Inspect AI task. Needs inspect_ai (skipped without it); no network, no GPU."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import inspect_ai  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False

from nightshift.lab.contracts import GenerationParams, GenPrompt, ScoreRecord, ScorerError, Split, read_record
from tests.test_lab_scorer import ScorerTestBase

PARAMS = GenerationParams.recommended(2000)


def sample(sid, text="answer", stop="stop", tokens=7, error=None):
    out = SimpleNamespace(completion=text, stop_reason=stop, usage=SimpleNamespace(output_tokens=tokens))
    return SimpleNamespace(id=sid, output=out, error=error)


@unittest.skipUnless(HAVE, "inspect_ai is not installed")
class MappingTests(unittest.TestCase):
    def test_results_follow_prompt_order_and_map_finish_reasons(self):
        from nightshift.lab.inspect_generator import results_from_samples

        prompts = [GenPrompt("a", "x"), GenPrompt("b", "y"), GenPrompt("c", "z")]
        got = results_from_samples([sample("c", "C\x00", "max_tokens"), sample("a", "A"), sample("b", "B", "content_filter")], prompts)
        self.assertEqual([g.prompt_id for g in got], ["a", "b", "c"])
        self.assertEqual([g.text for g in got], ["A", "B", "C"])  # NUL removed
        self.assertEqual([g.finish_reason for g in got], ["stop", "content_filter", "length"])
        self.assertEqual([g.truncated for g in got], [False, False, True])
        self.assertEqual(got[0].completion_tokens, 7)

    def test_a_failed_sample_becomes_a_generation_error_with_empty_text(self):
        from nightshift.lab.inspect_generator import results_from_samples

        got = results_from_samples([sample("a", error=SimpleNamespace(message="HTTP 500"))], [GenPrompt("a", "x")])
        self.assertEqual((got[0].text, got[0].error), ("", "HTTP 500"))

    def test_samples_that_do_not_match_the_prompts_are_refused(self):
        from nightshift.lab.inspect_generator import results_from_samples

        with self.assertRaises(ScorerError):
            results_from_samples([sample("a")], [GenPrompt("a", "x"), GenPrompt("b", "y")])
        with self.assertRaises(ScorerError):
            results_from_samples([sample("a"), sample("a")], [GenPrompt("a", "x")])

    def test_prompt_digest_depends_on_ids_texts_and_order(self):
        from nightshift.lab.inspect_generator import prompts_digest

        a, b = GenPrompt("a", "x"), GenPrompt("b", "y")
        self.assertEqual(prompts_digest([a, b]), prompts_digest([a, b]))
        self.assertNotEqual(prompts_digest([a, b]), prompts_digest([b, a]))
        self.assertNotEqual(prompts_digest([a, b]), prompts_digest([a, GenPrompt("b", "y ")]))


def mock_model(text="answer", stop_reason="stop"):
    from inspect_ai.model import ModelOutput, get_model

    def outputs():
        while True:
            out = ModelOutput.from_content("mockllm/model", text)
            out.choices[0].stop_reason = stop_reason
            yield out

    return lambda served: get_model("mockllm/model", custom_outputs=outputs())


@unittest.skipUnless(HAVE, "inspect_ai is not installed")
class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.prompts = [GenPrompt(f"k{i}", f"prompt {i} with {{braces}}") for i in range(3)]

    def gen(self, **kw):
        from nightshift.lab.inspect_generator import InspectGenerator

        return InspectGenerator("http://127.0.0.1:1", log_dir=Path(self.tmp.name) / "logs", workers=2, model_factory=mock_model(**kw))

    def test_generations_come_back_in_order_and_the_log_is_hashed(self):
        from inspect_ai.log import read_eval_log

        g = self.gen()
        got = g.generate("served", self.prompts, PARAMS, 3)
        self.assertEqual([r.prompt_id for r in got], ["k0", "k1", "k2"])
        self.assertTrue(all(r.text == "answer" and r.finish_reason == "stop" and r.error is None for r in got))
        ev = g.drain_evidence()
        self.assertEqual(g.drain_evidence(), {})  # drained
        self.assertEqual((ev["generator"], ev["inspect_task"]), ("inspect", "glimmer_generate@1"))
        path = Path(self.tmp.name) / "logs" / ev["inspect_log_seed3"]
        self.assertEqual(ev["inspect_log_sha256_seed3"], hashlib.sha256(path.read_bytes()).hexdigest())
        log = read_eval_log(str(path))
        self.assertEqual(log.eval.task_version, "1")
        self.assertEqual(log.eval.metadata["n_prompts"], 3)
        self.assertEqual(len(log.eval.metadata["prompts_sha256"]), 64)
        cfg = log.plan.config
        self.assertEqual((cfg.temperature, cfg.top_p, cfg.max_tokens, cfg.seed), (PARAMS.temperature, PARAMS.top_p, PARAMS.max_tokens, 3))
        self.assertEqual(cfg.extra_body, {"top_k": PARAMS.top_k, "chat_template_kwargs": {"reasoning_strength": PARAMS.reasoning_strength}})
        first = log.samples[0].messages
        self.assertEqual((first[0].role, first[0].content), ("system", PARAMS.system_prompt))
        self.assertIn("{braces}", first[1].content)  # prompt text reaches the model unchanged

    def test_a_length_stop_is_a_truncated_generation(self):
        got = self.gen(stop_reason="max_tokens").generate("served", self.prompts, PARAMS, 0)
        self.assertTrue(all(r.truncated for r in got))

    def test_an_empty_prompt_list_runs_nothing(self):
        self.assertEqual(self.gen().generate("served", [], PARAMS, 0), [])
        self.assertFalse((Path(self.tmp.name) / "logs").exists())


@unittest.skipUnless(HAVE, "inspect_ai is not installed")
class ScoreServiceTests(ScorerTestBase):
    def test_a_score_made_through_inspect_records_the_log_hashes_in_its_pins(self):
        from nightshift.lab.inspect_generator import InspectGenerator

        gen = InspectGenerator("http://127.0.0.1:1", log_dir=self.layout.private_dir / "inspect_logs", workers=2, model_factory=mock_model("OK"))
        rec = self.svc.score(self.request(split=Split.DEV), gen, "glimmer-base")
        on_disk = read_record(self.layout.score_record_path("sc-0001"), ScoreRecord)
        self.assertEqual(rec, on_disk)
        pins = rec.verifier_pins
        self.assertEqual((pins["generator"], pins["inspect_task"]), ("inspect", "glimmer_generate@1"))
        for seed in (0, 1):
            log = self.layout.private_dir / "inspect_logs" / pins[f"inspect_log_seed{seed}"]
            self.assertEqual(pins[f"inspect_log_sha256_seed{seed}"], hashlib.sha256(log.read_bytes()).hexdigest())
        self.assertIn("prompts_sha256", pins)  # the service's own pin is still there
        json.dumps(pins)

    def test_the_plain_http_generator_states_itself_in_the_record(self):
        from nightshift.lab.generator import HttpGenerator

        class Offline(HttpGenerator):
            def generate(self, model, prompts, params, seed):
                from tests.support.lab_fake_generator import FakeGenerator

                return FakeGenerator().generate(model, prompts, params, seed)

        rec = self.svc.score(self.request(split=Split.DEV), Offline("http://127.0.0.1:1"), "glimmer-base")
        self.assertEqual(rec.verifier_pins["generator"], "http")
        self.assertNotIn("inspect_task", rec.verifier_pins)

    def test_a_generator_without_evidence_adds_no_pins(self):
        rec = self.svc.score(self.request(split=Split.DEV), self.gen, "glimmer-base")
        self.assertNotIn("generator", rec.verifier_pins)


if __name__ == "__main__":
    unittest.main()
