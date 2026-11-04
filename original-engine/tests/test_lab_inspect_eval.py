import asyncio
import unittest
from types import SimpleNamespace

try:
    import inspect_ai  # noqa: F401
    HAVE = True
except ImportError:  # the lab does not need Inspect AI; its task is optional
    HAVE = False

from nightshift.lab.contracts import GenerationParams


class Backend:
    def score(self, rows, responses, mode="both"):
        self.seen = (rows, responses)
        ok = responses[0] == "good"
        return [SimpleNamespace(passed_strict=ok, fraction=1.0 if ok else 0.5, instruction_ids=("a:b",), strict=(ok,), error=False)]


@unittest.skipUnless(HAVE, "inspect_ai is not installed")
class InspectEvalTest(unittest.TestCase):
    def run_scorer(self, completion, stop_reason):
        from nightshift.lab import inspect_eval as ie

        backend = Backend()
        score_fn = ie.ifevalg_scorer(backend)
        state = SimpleNamespace(metadata={"row": {"key": "k", "ground_truth": "g"}}, output=SimpleNamespace(completion=completion, stop_reason=stop_reason))
        return asyncio.run(score_fn(state, None)).value, backend

    def test_truncated_generation_is_scored_as_empty(self):
        value, backend = self.run_scorer("good", "max_tokens")
        self.assertEqual(backend.seen[1], [""])
        self.assertEqual(value, {"prompt_all": 0, "fraction": 0.5, "truncated": 1})

    def test_complete_generation_is_scored_as_is(self):
        value, backend = self.run_scorer("good", "stop")
        self.assertEqual(value, {"prompt_all": 1, "fraction": 1.0, "truncated": 0})

    def test_task_carries_decoding_version_and_pins_the_dataset(self):
        import hashlib
        import tempfile
        from pathlib import Path
        from unittest import mock

        from nightshift.lab import inspect_eval as ie
        from nightshift.lab.contracts import LabLayout

        row = b'{"key":"k","prompt":"p","ground_truth":"g","instruction_ids":["a:b"]}\n'
        manifest = SimpleNamespace(get=lambda aid: SimpleNamespace(data_revisions={"dev_sha256": hashlib.sha256(row).hexdigest()}))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(ie, "_manifest", lambda layout: manifest):
            layout = LabLayout(Path(tmp))
            d = layout.dev_data("E1")
            d.parent.mkdir(parents=True)
            d.write_bytes(row)
            task = ie.build_task(layout, "E1", Backend(), {}, GenerationParams.recommended(6000))
            self.assertEqual(task.config.extra_body, {"top_k": 64, "chat_template_kwargs": {"reasoning_strength": "high"}})
            self.assertEqual(task.version, ie.TASK_VERSION)
            d.write_bytes(row.replace(b'"p"', b'"changed"'))
            with self.assertRaisesRegex(Exception, "changed after the freeze"):
                ie.load_dev_rows(layout, "E1")


if __name__ == "__main__":
    unittest.main()
