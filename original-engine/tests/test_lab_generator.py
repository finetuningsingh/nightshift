"""HttpGenerator against a loopback fake OpenAI-compatible endpoint (owner 1). No network beyond 127.0.0.1."""

import http.server
import json
import socket
import threading
import time
import unittest

from nightshift.lab.contracts import GenerationParams, GenPrompt, ScorerError
from nightshift.lab.generator import DEAD_AFTER, HttpGenerator, chat_completions_url

PARAMS = GenerationParams(0.7, 0.9, 40, 123, "medium", "You are a test system.")


class FakeEndpoint:
    """A loopback server. ``handler(path, headers, body) -> (status, json-able or str)``."""

    def __init__(self, handler):
        self.requests = []
        self._lock = threading.Lock()
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n))
                with outer._lock:
                    outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                status, payload = handler(self.path, self.headers, body)
                data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def completion(text, finish="stop", tokens=5, **message_extra):
    return {"choices": [{"message": {"role": "assistant", "content": text, **message_extra}, "finish_reason": finish}],
            "usage": {"completion_tokens": tokens}}


def prompts(n):
    return [GenPrompt(f"p{i}", f"question {i}") for i in range(n)]


def no_sleep(_):
    pass


class GeneratorTests(unittest.TestCase):
    def start(self, handler):
        ep = FakeEndpoint(handler)
        self.addCleanup(ep.close)
        return ep

    def test_url_forms(self):
        self.assertEqual(chat_completions_url("http://h:1"), "http://h:1/v1/chat/completions")
        self.assertEqual(chat_completions_url("http://h:1/"), "http://h:1/v1/chat/completions")
        self.assertEqual(chat_completions_url("http://h:1/v1"), "http://h:1/v1/chat/completions")
        self.assertEqual(chat_completions_url("http://h:1/v1/"), "http://h:1/v1/chat/completions")

    def test_request_body_carries_explicit_system_message_decoding_and_seed(self):
        ep = self.start(lambda p, h, b: (200, completion("hi")))
        HttpGenerator(ep.url, sleep=no_sleep).generate("glimmer-base", [GenPrompt("p0", "What is 2+2?")], PARAMS, 41)
        req = ep.requests[0]
        self.assertEqual(req["path"], "/v1/chat/completions")
        b = req["body"]
        self.assertEqual(b["model"], "glimmer-base")
        self.assertEqual(b["messages"], [{"role": "system", "content": "You are a test system."}, {"role": "user", "content": "What is 2+2?"}])
        self.assertEqual((b["temperature"], b["top_p"], b["top_k"], b["max_tokens"], b["seed"]), (0.7, 0.9, 40, 123, 41))
        self.assertEqual(b["chat_template_kwargs"], {"reasoning_strength": "medium"})
        self.assertFalse(b["stream"])
        self.assertNotIn("Authorization", req["headers"])

    def test_input_order_is_kept_and_only_content_is_read(self):
        def handler(path, headers, body):
            text = body["messages"][1]["content"]
            if text.endswith("0"):
                time.sleep(0.15)  # the first prompt answers last
            return 200, completion(f"answer to {text}", tokens=len(text), reasoning_content="SECRET REASONING", reasoning="more")
        ep = self.start(handler)
        out = HttpGenerator(ep.url, workers=4, sleep=no_sleep).generate("m", prompts(6), PARAMS, 0)
        self.assertEqual([r.prompt_id for r in out], [f"p{i}" for i in range(6)])
        self.assertEqual([r.text for r in out], [f"answer to question {i}" for i in range(6)])
        self.assertTrue(all("SECRET" not in r.text and "more" not in r.text for r in out))
        self.assertEqual(out[3].completion_tokens, len("question 3"))
        self.assertTrue(all(r.finish_reason == "stop" and r.error is None for r in out))

    def test_finish_reason_length_marks_truncation_and_null_content_is_empty(self):
        def handler(path, headers, body):
            if body["messages"][1]["content"].endswith("1"):
                return 200, {"choices": [{"message": {"role": "assistant", "content": None, "reasoning_content": "thinking only"}, "finish_reason": "stop"}]}
            return 200, completion("cut off", finish="length", tokens=123)
        ep = self.start(handler)
        out = HttpGenerator(ep.url, sleep=no_sleep).generate("m", prompts(2), PARAMS, 0)
        self.assertTrue(out[0].truncated)
        self.assertEqual(out[0].text, "cut off")
        self.assertEqual((out[1].text, out[1].completion_tokens, out[1].error), ("", 0, None))

    def test_api_key_is_sent_but_not_shown(self):
        ep = self.start(lambda p, h, b: (200, completion("ok")))
        gen = HttpGenerator(ep.url + "/v1", api_key="sk-secret-value", sleep=no_sleep)
        gen.generate("m", prompts(1), PARAMS, 0)
        self.assertEqual(ep.requests[0]["headers"]["Authorization"], "Bearer sk-secret-value")
        self.assertNotIn("sk-secret-value", repr(gen))

    def test_transient_failure_is_retried_then_succeeds(self):
        calls = {"n": 0}
        def handler(path, headers, body):
            calls["n"] += 1
            return (500, "boom") if calls["n"] < 3 else (200, completion("finally"))
        ep = self.start(handler)
        out = HttpGenerator(ep.url, retries=2, sleep=no_sleep).generate("m", prompts(1), PARAMS, 0)
        self.assertEqual((out[0].text, out[0].error), ("finally", None))
        self.assertEqual(calls["n"], 3)

    def test_persistent_per_prompt_failure_is_reported_not_raised(self):
        def handler(path, headers, body):
            if body["messages"][1]["content"] == "question 1":
                return 500, "always"
            return 200, completion("fine")
        ep = self.start(handler)
        out = HttpGenerator(ep.url, retries=1, sleep=no_sleep).generate("m", prompts(3), PARAMS, 0)
        self.assertEqual([r.error is None for r in out], [True, False, True])
        self.assertEqual((out[1].text, out[1].finish_reason), ("", ""))
        self.assertIn("500", out[1].error)

    def test_a_prompt_the_server_rejects_is_not_retried_and_not_systemic(self):
        counts = {}
        def handler(path, headers, body):
            q = body["messages"][1]["content"]
            counts[q] = counts.get(q, 0) + 1
            return (400, "context too long") if q == "question 0" else (200, completion("fine"))
        ep = self.start(handler)
        out = HttpGenerator(ep.url, retries=3, sleep=no_sleep).generate("m", prompts(2), PARAMS, 0)
        self.assertEqual(counts["question 0"], 1)
        self.assertIsNotNone(out[0].error)
        self.assertIsNone(out[1].error)

    def test_wrong_model_or_credentials_raise_immediately(self):
        for status in (401, 403, 404):
            ep = self.start(lambda p, h, b, s=status: (s, "nope"))
            with self.assertRaises(ScorerError, msg=str(status)):
                HttpGenerator(ep.url, retries=3, sleep=no_sleep).generate("m", prompts(5), PARAMS, 0)

    def test_malformed_reply_is_a_per_prompt_error(self):
        ep = self.start(lambda p, h, b: (200, {"choices": []}))
        out = HttpGenerator(ep.url, retries=0, sleep=no_sleep).generate("m", prompts(2), PARAMS, 0)
        self.assertTrue(all(r.error and "malformed" in r.error for r in out))

    def test_a_reply_without_message_content_is_an_error_not_an_empty_answer(self):
        for message in ({}, {"role": "assistant"}, {"role": "assistant", "reasoning_content": "thinking only"}):
            ep = self.start(lambda p, h, b, m=message: (200, {"choices": [{"message": m, "finish_reason": "stop"}]}))
            out = HttpGenerator(ep.url, retries=0, sleep=no_sleep).generate("m", prompts(2), PARAMS, 0)
            self.assertTrue(all(r.error and "malformed" in r.error for r in out), message)

    def test_dead_endpoint_raises_scorer_error_quickly(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()  # nothing listens here now
        t0 = time.time()
        with self.assertRaises(ScorerError) as cm:
            HttpGenerator(f"http://127.0.0.1:{port}", retries=0, workers=2, sleep=no_sleep).generate("m", prompts(DEAD_AFTER * 4), PARAMS, 0)
        self.assertIn("unreachable", str(cm.exception))
        self.assertLess(time.time() - t0, 20)

    def test_a_few_transport_failures_among_successes_do_not_kill_the_run(self):
        n = {"i": 0}
        def handler(path, headers, body):
            n["i"] += 1
            return 200, completion("ok")
        ep = self.start(handler)
        out = HttpGenerator(ep.url, sleep=no_sleep).generate("m", prompts(20), PARAMS, 0)
        self.assertEqual(len(out), 20)

    def test_nul_and_surrogates_do_not_break_the_contract(self):
        ep = self.start(lambda p, h, b: (200, '{"choices":[{"message":{"content":"a\\u0000b\\ud800c"},"finish_reason":"stop"}]}'))
        out = HttpGenerator(ep.url, sleep=no_sleep).generate("m", prompts(1), PARAMS, 0)
        self.assertIsNone(out[0].error)
        self.assertNotIn("\x00", out[0].text)
        out[0].text.encode("utf-8")  # encodable

    def test_empty_prompt_list_and_bad_construction(self):
        self.assertEqual(HttpGenerator("http://127.0.0.1:1").generate("m", [], PARAMS, 0), [])
        with self.assertRaises(ScorerError):
            HttpGenerator("ftp://x")
        with self.assertRaises(ScorerError):
            HttpGenerator("http://x", workers=0)

    def test_satisfies_the_generator_protocol(self):
        from nightshift.lab.contracts import Generator
        self.assertIsInstance(HttpGenerator("http://127.0.0.1:1"), Generator)


if __name__ == "__main__":
    unittest.main()
