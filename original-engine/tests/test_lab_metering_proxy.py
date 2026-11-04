import json
import math
import socket
import threading
import time
import unittest
import urllib.request

from nightshift.lab.contracts import BudgetSpec
from nightshift.lab.metering_proxy import (
    DEFAULT_MAX_TOKENS,
    LabHttpServer,
    MeteringProxy,
    estimate_prompt_tokens,
    request_max_tokens,
    usage_from,
)
from tests.support.lab_fake_runtime import FakeClock, FakeMeter, Upstream, post_json

API_KEY = "meta-secret-key-0123456789"
TOKEN = "trial-token-abcdefghijklmnop"


def body(**kw):
    b = {"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "hello there"}], "max_tokens": 100}
    b.update(kw)
    return b


class ProxyCase(unittest.TestCase):
    spec = BudgetSpec(researcher_tokens=1_000_000, gpu_minutes=60.0, dollars=50.0, wall_seconds=3600.0)

    def setUp(self):
        self.clock = FakeClock()
        self.upstream = Upstream().start()
        self.addCleanup(self.upstream.stop)
        self.meter = FakeMeter(self.spec, clock=self.clock)
        self.make_proxy()

    def make_proxy(self, meter=None, upstream_url=None):
        self.proxy = MeteringProxy(meter or self.meter, upstream_url or self.upstream.base_url, API_KEY, TOKEN, clock=self.clock)
        self.server = LabHttpServer([self.proxy]).start()
        self.addCleanup(self.server.stop)
        self.url = f"http://127.0.0.1:{self.server.port}/v1/chat/completions"

    def post(self, b=None, token=TOKEN, **kw):
        return post_json(self.url, token, b if b is not None else body(), **kw)


class ForwardingTests(ProxyCase):
    def test_non_stream_settles_actual_usage_and_uses_the_meta_key_not_the_trial_token(self):
        status, headers, data = self.post()
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["choices"][0]["message"]["content"], "ok")
        up = self.upstream.requests[0]
        self.assertEqual(up["headers"]["authorization"], "Bearer " + API_KEY)
        self.assertNotIn(TOKEN, json.dumps(up["headers"]))
        self.assertEqual(up["path"], "/v1/chat/completions")
        usage = self.meter.usage()
        self.assertEqual(usage.researcher_tokens, 11 + 7)  # actual usage replaced the reservation
        self.assertEqual(self.meter.reservations(), {})
        self.assertAlmostEqual(usage.researcher_usd, 11 * 1e-6 + 7 * 4e-6)
        self.assertEqual(self.proxy.stats()["settled"], 1)
        self.assertIsNone(self.proxy.exhausted)

    def test_reservation_is_booked_before_the_request_is_forwarded(self):
        seen = {}

        def on_request(rec):
            seen["reserved"] = self.meter.reservations()
            seen["usage"] = self.meter.usage().researcher_tokens

        self.upstream.queue.append(Upstream.Reply(on_request=on_request))
        self.post(body(max_tokens=500))
        self.assertEqual(len(seen["reserved"]), 1)
        (tokens, _usd), = seen["reserved"].values()
        self.assertGreaterEqual(tokens, 500)  # estimate + max_tokens, worst case
        self.assertGreaterEqual(seen["usage"], 500)
        self.assertEqual(self.meter.reserve_calls[0][2], 500)

    def test_prompt_estimate_is_request_bytes_over_three_and_max_tokens_defaults_to_16000(self):
        raw_body = json.dumps({"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "x" * 3000}]}).encode()
        self.post(raw=raw_body)
        model, estimate, max_tokens, call_id = self.meter.reserve_calls[0]
        self.assertEqual(model, "muse-spark-1.3")
        self.assertEqual(estimate, math.ceil(len(raw_body) / 3))
        self.assertEqual(max_tokens, DEFAULT_MAX_TOKENS)
        self.assertEqual(estimate_prompt_tokens(9), 3)
        self.assertEqual(request_max_tokens({"max_completion_tokens": 77}), 77)
        self.assertEqual(request_max_tokens({"max_tokens": True}), DEFAULT_MAX_TOKENS)
        self.assertEqual(request_max_tokens({"max_tokens": -4}), DEFAULT_MAX_TOKENS)

    def test_reservation_covers_every_requested_completion(self):
        self.post(body(max_tokens=100, n=10))
        self.assertEqual(self.meter.reserve_calls[0][2], 1000)
        self.assertEqual(self.upstream.requests[0]["body"]["max_tokens"], 100)
        self.assertEqual(self.upstream.requests[0]["body"]["n"], 10)

    def test_invalid_n_is_refused_before_anything_is_reserved_or_forwarded(self):
        for n in (0, -1, 1.5, "3", True, None, [2]):
            status, _h, _d = self.post(body(n=n))
            self.assertEqual(status, 400, n)
        self.assertEqual(self.meter.reserve_calls, [])
        self.assertEqual(self.upstream.requests, [])

    def test_forwarded_output_limit_is_exactly_the_reserved_one(self):
        raw_body = json.dumps({"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "hi"}]}).encode()
        self.post(raw=raw_body)  # no limit asked for: upstream is told the default, not left unbounded
        self.assertEqual(self.upstream.requests[-1]["body"]["max_tokens"], DEFAULT_MAX_TOKENS)
        self.assertNotIn("max_completion_tokens", self.upstream.requests[-1]["body"])
        self.post(body(max_tokens=50, max_completion_tokens=99999))  # both given: neither can exceed what was reserved
        up = self.upstream.requests[-1]["body"]
        self.assertEqual((up["max_tokens"], up["max_completion_tokens"]), (50, 50))
        self.post({"model": "m", "messages": [], "max_tokens": -4, "max_completion_tokens": 0})  # invalid limits fall back to the default
        up = self.upstream.requests[-1]["body"]
        self.assertEqual((up["max_tokens"], up["max_completion_tokens"]), (DEFAULT_MAX_TOKENS, DEFAULT_MAX_TOKENS))

    def test_stream_forces_include_usage_and_settles_on_the_usage_chunk(self):
        self.upstream.queue.append(Upstream.Reply(text="one two three", usage=(20, 9, 5)))
        status, headers, data = self.post(body(stream=True, stream_options={"include_usage": False, "other": 1}))
        self.assertEqual(status, 200)
        self.assertIn(b"[DONE]", data)
        self.assertIn(b"one ", data)
        self.assertEqual(self.upstream.requests[0]["body"]["stream_options"], {"include_usage": True, "other": 1})
        self.assertEqual(self.meter.usage().researcher_tokens, 29)
        self.assertEqual(self.meter.reservations(), {})

    def test_stream_options_added_when_absent_and_not_added_to_non_stream_requests(self):
        self.post(body(stream=True))
        self.assertEqual(self.upstream.requests[-1]["body"]["stream_options"], {"include_usage": True})
        self.post(body())
        self.assertNotIn("stream_options", self.upstream.requests[-1]["body"])

    def test_stream_without_usage_chunk_keeps_the_worst_case_reservation(self):
        self.upstream.queue.append(Upstream.Reply(stream_usage=False))
        status, _h, data = self.post(body(stream=True, max_tokens=4000))
        self.assertEqual(status, 200)
        (tokens, _usd), = self.meter.reservations().values()
        self.assertGreaterEqual(tokens, 4000)
        self.assertEqual(self.proxy.stats()["unsettled"], 1)
        self.assertGreaterEqual(self.meter.usage().researcher_tokens, 4000)

    def test_non_stream_response_without_usage_keeps_the_reservation(self):
        self.upstream.queue.append(Upstream.Reply(usage=None))
        self.post(body(max_tokens=900))
        self.assertEqual(len(self.meter.reservations()), 1)
        self.assertEqual(self.proxy.stats()["unsettled"], 1)

    def test_upstream_dropping_mid_stream_without_usage_keeps_the_reservation(self):
        self.upstream.queue.append(Upstream.Reply(text="a b c d", drop_after_chunks=2))
        self.post(body(stream=True, max_tokens=1234))
        self.assertEqual(len(self.meter.reservations()), 1)

    def test_client_disconnect_mid_stream_keeps_the_reservation(self):
        self.upstream.queue.append(Upstream.Reply(text=" ".join(["w"] * 400), delay_per_chunk=0.02))
        data = json.dumps(body(stream=True, max_tokens=777)).encode()
        s = socket.create_connection(("127.0.0.1", self.server.port))
        s.sendall(b"POST /v1/chat/completions HTTP/1.0\r\nAuthorization: Bearer " + TOKEN.encode() + b"\r\nContent-Type: application/json\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data)
        s.recv(64)
        s.close()  # hang up long before the usage chunk
        for _ in range(200):
            if self.proxy.stats()["unsettled"] >= 1:
                break
            time.sleep(0.05)
        self.assertEqual(self.proxy.stats()["unsettled"], 1)
        self.assertEqual(self.proxy.stats()["settled"], 0)
        (tokens, _usd), = self.meter.reservations().values()
        self.assertGreaterEqual(tokens, 777)

    def test_upstream_http_error_is_relayed_and_settled_at_zero(self):
        self.upstream.queue.append(Upstream.Reply(status=429, headers={"Retry-After": "7"}))
        status, headers, data = self.post(body(max_tokens=3000))
        self.assertEqual(status, 429)
        self.assertEqual(headers.get("retry-after"), "7")
        self.assertEqual(self.meter.reservations(), {})
        self.assertEqual(self.meter.usage().researcher_tokens, 0)
        self.assertEqual(self.proxy.stats()["settled_zero"], 1)
        self.assertIsNone(self.proxy.exhausted)  # an upstream rate limit is not our budget

    def test_unreachable_upstream_gives_502_and_settles_at_zero(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        self.meter = FakeMeter(self.spec, clock=self.clock)
        self.make_proxy(meter=self.meter, upstream_url=f"http://127.0.0.1:{port}/v1")
        status, _h, _d = self.post()
        self.assertEqual(status, 502)
        self.assertEqual(self.meter.usage().researcher_tokens, 0)
        self.assertEqual(self.meter.reservations(), {})

    def test_concurrent_calls_are_all_settled(self):
        results = []

        def one():
            results.append(self.post(body(max_tokens=50))[0])

        threads = [threading.Thread(target=one) for _ in range(12)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(results, [200] * 12)
        self.assertEqual(self.meter.usage().researcher_tokens, 12 * 18)
        self.assertEqual(self.meter.reservations(), {})
        self.assertEqual(self.proxy.stats()["settled"], 12)


class ShutdownTests(ProxyCase):
    def test_stop_waits_for_a_request_still_waiting_on_upstream_so_it_settles_before_accounting_closes(self):
        arrived, release = threading.Event(), threading.Event()
        self.upstream.queue.append(Upstream.Reply(on_request=lambda rec: (arrived.set(), release.wait(10))))
        results = []
        client = threading.Thread(target=lambda: results.append(self.post(body())[0]), daemon=True)
        client.start()
        self.assertTrue(arrived.wait(10))
        self.assertEqual(self.server.active_requests, 1)
        threading.Timer(0.4, release.set).start()
        self.server.stop()
        self.assertEqual(self.meter.reservations(), {})  # settled before stop() returned
        self.assertEqual(self.meter.usage().researcher_tokens, 11 + 7)
        self.assertEqual(self.server.active_requests, 0)
        client.join(10)
        self.assertEqual(results, [200])

    def test_stop_gives_up_after_the_drain_timeout_and_reports_the_straggler(self):
        arrived, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        self.upstream.queue.append(Upstream.Reply(on_request=lambda rec: (arrived.set(), release.wait(10))))
        threading.Thread(target=lambda: self.post(body()), daemon=True).start()
        self.assertTrue(arrived.wait(10))
        started = time.monotonic()
        self.server.stop(drain_timeout=0.2)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(self.server.active_requests, 1)


    def test_after_close_a_late_response_neither_settles_nor_changes_the_meter(self):
        arrived, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        self.upstream.queue.append(Upstream.Reply(on_request=lambda rec: (arrived.set(), release.wait(10))))
        results = []
        client = threading.Thread(target=lambda: results.append(self.post(body())[0]), daemon=True)
        client.start()
        self.assertTrue(arrived.wait(10))
        self.server.stop(drain_timeout=0.1)
        reserved = self.meter.usage().researcher_tokens
        self.assertEqual(len(self.meter.reservations()), 1)
        self.proxy.close()  # what the trial runner does before it reads the usage and seals the ledger
        release.set()
        client.join(10)
        self.assertEqual(results, [200])  # the client still gets its answer
        self.assertEqual(self.meter.usage().researcher_tokens, reserved)  # but the booked reservation stands
        self.assertEqual(len(self.meter.reservations()), 1)
        self.assertEqual(self.proxy.stats()["dropped_after_close"], 1)
        self.assertEqual(self.proxy.stats()["settled"], 0)


class RefusalTests(ProxyCase):
    def test_wrong_or_missing_token_is_401_and_nothing_is_forwarded_or_reserved(self):
        for token in ("", "nope", TOKEN + "x", TOKEN[:-1]):
            status, _h, data = self.post(token=token)
            self.assertEqual(status, 401, token)
            self.assertNotIn(API_KEY.encode(), data)
        self.assertEqual(self.upstream.requests, [])
        self.assertEqual(self.meter.reserve_calls, [])
        self.assertEqual(self.proxy.stats()["rejected_auth"], 4)

    def test_other_paths_and_methods_are_refused(self):
        status, _h, _d = post_json(f"http://127.0.0.1:{self.server.port}/v1/completions", TOKEN, body())
        self.assertEqual(status, 404)
        req = urllib.request.Request(self.url, headers={"Authorization": "Bearer " + TOKEN})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 405)
        self.assertEqual(self.upstream.requests, [])

    def test_malformed_bodies_are_400_and_never_reserved(self):
        for raw in (b"not json", b'{"model": "m", "messages": [], "x": NaN}', b'{"messages": []}', b'{"model": "m"}', b"[]"):
            status, _h, _d = self.post(raw=raw)
            self.assertEqual(status, 400, raw)
        self.assertEqual(self.meter.reserve_calls, [])
        self.assertEqual(self.upstream.requests, [])

    def test_chunked_sizes_that_are_not_plain_hex_are_refused_without_reading_the_payload(self):
        import io
        from types import SimpleNamespace

        from nightshift.lab.metering_proxy import BadRequest, read_request_body

        for size in (b"-1", b"+5", b"0x5", b"1_0", b"zz"):
            rfile = io.BytesIO(size + b"\r\n" + b"x" * 1000)
            h = SimpleNamespace(headers={"Transfer-Encoding": "chunked"}, rfile=rfile)
            with self.assertRaises(BadRequest, msg=size):
                read_request_body(h, 10)
            self.assertLess(rfile.tell(), 20, size)  # `-1` used to reach read(-1): everything up to EOF, past the 10-byte limit

    def test_oversized_body_is_413(self):
        proxy = MeteringProxy(self.meter, self.upstream.base_url, API_KEY, TOKEN, clock=self.clock, max_request_bytes=200)
        server = LabHttpServer([proxy]).start()
        self.addCleanup(server.stop)
        status, _h, _d = post_json(f"http://127.0.0.1:{server.port}/v1/chat/completions", TOKEN, body(messages=[{"role": "user", "content": "y" * 500}]))
        self.assertEqual(status, 413)
        self.assertEqual(self.upstream.requests, [])


class HardStopTests(ProxyCase):
    def test_429_at_the_cap_forwards_nothing_and_flags_the_runner(self):
        meter = FakeMeter(BudgetSpec(researcher_tokens=1000, gpu_minutes=60.0, dollars=50.0, wall_seconds=3600.0), clock=self.clock)
        self.make_proxy(meter=meter)
        status, _h, data = self.post(body(max_tokens=2000))  # worst case alone crosses the cap
        self.assertEqual(status, 429)
        doc = json.loads(data)["error"]
        self.assertEqual(doc["type"], "budget_exhausted")
        self.assertEqual(doc["axis"], "researcher_tokens")
        self.assertEqual(self.upstream.requests, [])
        self.assertIsNotNone(self.proxy.exhausted)
        self.assertEqual(self.proxy.exhausted.axis, "researcher_tokens")
        self.assertTrue(self.proxy.exhausted_event.is_set())
        self.assertEqual(self.proxy.stats()["rejected_budget"], 1)

    def test_spend_is_stopped_once_the_cap_is_reached_across_calls(self):
        meter = FakeMeter(BudgetSpec(researcher_tokens=500, gpu_minutes=60.0, dollars=50.0, wall_seconds=3600.0), clock=self.clock)
        self.make_proxy(meter=meter)
        self.upstream.queue.extend([Upstream.Reply(usage=(100, 100, 0))] * 5)
        codes = [self.post(body(max_tokens=100))[0] for _ in range(5)]
        self.assertEqual(codes[:2], [200, 200])
        self.assertIn(429, codes)
        n_forwarded = len(self.upstream.requests)
        self.assertLess(n_forwarded, 5)
        self.assertLessEqual(meter.usage().researcher_tokens, 500)  # the reserve-then-settle bound held with an accurate estimate
        self.assertTrue(self.proxy.exhausted_event.is_set())

    def test_settling_across_the_cap_sets_the_flag_but_still_returns_the_paid_response(self):
        meter = FakeMeter(BudgetSpec(researcher_tokens=300, gpu_minutes=60.0, dollars=50.0, wall_seconds=3600.0), clock=self.clock)
        self.make_proxy(meter=meter)
        self.upstream.queue.append(Upstream.Reply(usage=(150, 200, 0)))
        status, _h, data = self.post(body(max_tokens=100))
        self.assertEqual(status, 200)  # already paid for
        self.assertIsNotNone(self.proxy.exhausted)
        self.assertEqual(self.post(body(max_tokens=10))[0], 429)


class SecretTests(ProxyCase):
    def test_upstream_error_body_echoing_the_key_is_scrubbed(self):
        leak = json.dumps({"error": {"message": "bad key " + API_KEY}}).encode()
        self.upstream.queue.append(Upstream.Reply(status=401, raw_body=leak))
        status, headers, data = self.post()
        self.assertEqual(status, 401)
        self.assertNotIn(API_KEY.encode(), data)
        self.assertIn(b"[REDACTED]", data)
        self.assertNotIn(API_KEY, json.dumps(headers))

    def test_non_stream_response_never_carries_the_key_or_upstream_headers(self):
        _s, headers, data = self.post()
        self.assertNotIn(API_KEY, json.dumps(headers) + data.decode())
        self.assertNotIn("set-cookie", headers)

    def test_constructor_rejects_bad_upstream_and_missing_secrets(self):
        with self.assertRaises(ValueError):
            MeteringProxy(self.meter, "ftp://x/v1", API_KEY, TOKEN, clock=self.clock)
        with self.assertRaises(ValueError):
            MeteringProxy(self.meter, self.upstream.base_url, "", TOKEN, clock=self.clock)
        with self.assertRaises(ValueError):
            MeteringProxy(self.meter, self.upstream.base_url, API_KEY, "", clock=self.clock)


class UsageParsingTests(unittest.TestCase):
    def test_usage_from(self):
        u = usage_from({"usage": {"prompt_tokens": 10, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 4}}})
        self.assertEqual((u.prompt_tokens, u.completion_tokens, u.cached_tokens), (10, 3, 4))
        self.assertEqual(usage_from({"usage": {"prompt_tokens": 10, "completion_tokens": 3, "cached_tokens": 99}}).cached_tokens, 10)  # clamped
        for bad in (None, {}, {"usage": None}, {"usage": {"prompt_tokens": -1, "completion_tokens": 1}}, {"usage": {"prompt_tokens": "x", "completion_tokens": 1}},
                    {"usage": {"prompt_tokens": True, "completion_tokens": 1}}, {"usage": {"prompt_tokens": 1.5, "completion_tokens": 1}}):
            self.assertIsNone(usage_from(bad), bad)


if __name__ == "__main__":
    unittest.main()
