"""Real loopback HTTP tests for nightshift.dashboard.server."""
from __future__ import annotations

import http.client
import io
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from nightshift.dashboard import server as srv
from nightshift.dashboard.server import CSP, ReplayRefused, make_server, parse_since
from nightshift.dashboard.state_api import BANNER, SIMULATED_BANNER, build_state
from nightshift.ledger import Ledger
from nightshift.types import Kind
from nightshift.view import build_view
from tests.support.synthetic import FakeClock, RunBuilder

SEALED_CANARY = 0.314159265
ROUTES = ["/", "/app.js", "/app.css", "/api/state", "/api/events", "/api/health"]


class FakeMono:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class ServerCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.run_dir = self.root / "run"
        self.assets = self.root / "assets"
        self.assets.mkdir()
        (self.assets / "index.html").write_text("<!doctype html><title>t</title><script src=\"/app.js\"></script>", encoding="utf-8")
        (self.assets / "app.js").write_text("console.log('hi');", encoding="utf-8")
        (self.assets / "app.css").write_text("body{margin:0}", encoding="utf-8")
        self.servers: list = []

    # ---- ledger builders ---------------------------------------------------------------------------
    def build_ledger(self, backend: str = "scripted", close: bool = True) -> Ledger:
        led = Ledger.create(self.run_dir, clock=FakeClock(), fsync=False)
        rb = RunBuilder(led)
        rb.start("law", backend=backend)
        base = rb.baseline(1.0, 0.9)
        e = rb.experiment("A", "quadratic", score=0.8, train=0.7, delta=0.2, noise_margin=0.01)
        rb.replication_baseline(base, SEALED_CANARY)
        rb.replicate(e, SEALED_CANARY, sealed_delta=SEALED_CANARY)
        rb.promote(e, score=0.8)
        rb.finish()
        if close:
            led.close()
        self.addCleanup(led.close)
        return led

    # ---- server helpers ----------------------------------------------------------------------------
    def serve(self, **kw):
        kw.setdefault("assets_dir", self.assets)
        server = make_server(self.run_dir, "127.0.0.1", 0, **kw)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        self.servers.append(server)
        self.addCleanup(self._stop, server)
        self.port = server.server_address[1]
        return server

    @staticmethod
    def _stop(server) -> None:
        server.shutdown()
        server.server_close()

    def get(self, path: str, method: str = "GET", headers: dict | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request(method, path, headers=headers or {})
            r = conn.getresponse()
            body = r.read()
            return r.status, {k.lower(): v for k, v in r.getheaders()}, body
        finally:
            conn.close()

    def raw(self, text: str):
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as s:
            s.sendall(text.encode("latin-1"))
            chunks = []
            while True:
                data = s.recv(65536)
                if not data:
                    break
                chunks.append(data)
        blob = b"".join(chunks)
        head, _, body = blob.partition(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        status = int(lines[0].split()[1])
        hdrs = {}
        for ln in lines[1:]:
            k, _, v = ln.partition(":")
            hdrs[k.strip().lower()] = v.strip()
        return status, hdrs, body

    def state(self) -> dict:
        status, _, body = self.get("/api/state")
        self.assertEqual(status, 200)
        return json.loads(body)

    def assert_security_headers(self, h: dict) -> None:
        self.assertEqual(h["content-security-policy"], CSP)
        self.assertEqual(h["x-content-type-options"], "nosniff")
        self.assertEqual(h["referrer-policy"], "no-referrer")
        self.assertEqual(h["cache-control"], "no-store")
        self.assertEqual([k for k in h if k.startswith("access-control-")], [])


class HeaderAndRouteTests(ServerCase):
    def test_csp_is_the_contract_string(self):
        self.assertEqual(CSP, "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; "
                              "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")

    def test_every_route_carries_the_full_header_set(self):
        self.build_ledger()
        self.serve()
        types = {"/": "text/html", "/app.js": "text/javascript", "/app.css": "text/css", "/api/state": "application/json",
                 "/api/events": "application/json", "/api/health": "application/json"}
        for route in ROUTES:
            status, h, body = self.get(route)
            self.assertEqual(status, 200, route)
            self.assert_security_headers(h)
            self.assertTrue(h["content-type"].startswith(types[route]), (route, h["content-type"]))
            self.assertEqual(int(h["content-length"]), len(body))

    def test_static_files_are_served_from_the_assets_directory(self):
        self.build_ledger()
        self.serve()
        self.assertEqual(self.get("/app.js")[2], b"console.log('hi');")
        self.assertIn(b"<title>t</title>", self.get("/")[2])
        self.assertEqual(self.get("/app.css")[2], b"body{margin:0}")

    def test_missing_asset_is_a_404_with_headers(self):
        self.build_ledger()
        (self.assets / "app.css").unlink()
        self.serve()
        status, h, body = self.get("/app.css")
        self.assertEqual(status, 404)
        self.assert_security_headers(h)
        self.assertEqual(self.get("/api/health")[0], 200)

    def test_default_assets_directory_is_the_package_directory(self):
        self.assertEqual(srv.PACKAGE_DIR, Path(srv.__file__).resolve().parent)
        self.build_ledger()
        server = make_server(self.run_dir, "127.0.0.1", 0)
        self.addCleanup(server.server_close)
        self.assertEqual(server.assets_dir, srv.PACKAGE_DIR)

    def test_unknown_paths_are_404_with_headers(self):
        self.build_ledger()
        self.serve()
        for path in ["/index.html", "/api/state/", "/api", "/api/../app.js", "/%2e%2e/etc/passwd", "/../../etc/passwd",
                     "/APP.JS", "/app.js/", "/static/app.js", "/api/statex", "/ledger.jsonl", "/.git/config",
                     "/api/state%00", "*", "http://127.0.0.1/api/state"]:
            status, h, _ = self.raw(f"GET {path} HTTP/1.0\r\nHost: 127.0.0.1:{self.port}\r\n\r\n")[:3] \
                if path.startswith(("*", "http")) else self.get(path)
            self.assertEqual(status, 404, path)
            self.assert_security_headers(h)

    def test_query_string_does_not_change_routing(self):
        self.build_ledger()
        self.serve()
        self.assertEqual(self.get("/app.js?v=1")[0], 200)
        self.assertEqual(self.get("/api/health?x=%00")[0], 200)

    def test_head_matches_get_without_a_body(self):
        self.build_ledger()
        self.serve()
        s1, h1, b1 = self.get("/api/health")
        s2, h2, b2 = self.get("/api/health", method="HEAD")
        self.assertEqual((s1, s2), (200, 200))
        self.assertEqual(b2, b"")
        self.assertEqual(h2["content-length"], h1["content-length"])
        self.assert_security_headers(h2)

    def test_non_get_methods_are_405_with_headers_and_no_cors(self):
        self.build_ledger()
        self.serve()
        for method in ["POST", "PUT", "DELETE", "PATCH", "OPTIONS", "TRACE"]:
            status, h, _ = self.get("/api/state", method=method,
                                    headers={"Origin": f"http://127.0.0.1:{self.port}"} if method == "OPTIONS" else {})
            self.assertEqual(status, 405, method)
            self.assertEqual(h["allow"], "GET, HEAD")
            self.assert_security_headers(h)
        status, h, _ = self.raw(f"FOO /api/state HTTP/1.0\r\nHost: 127.0.0.1:{self.port}\r\n\r\n")
        self.assertEqual(status, 405)
        status, h, _ = self.raw(f"POST /api/state HTTP/1.0\r\nHost: 127.0.0.1:{self.port}\r\nContent-Length: 4\r\n\r\nabcd")
        self.assertEqual(status, 405)

    def test_garbage_request_gets_an_error_with_headers(self):
        self.build_ledger()
        self.serve()
        status, h, _ = self.raw("this is not http\r\n\r\n")
        self.assertIn(status, (400, 404, 405))
        self.assert_security_headers(h)

    def test_an_exception_while_building_state_is_a_plain_500(self):
        self.build_ledger()
        server = self.serve()
        server.state = lambda: (_ for _ in ()).throw(RuntimeError("secret internals /home/x"))
        status, h, body = self.get("/api/state")
        self.assertEqual(status, 500)
        self.assertNotIn(b"secret", body)
        self.assertNotIn(b"Traceback", body)
        self.assert_security_headers(h)

    def test_server_is_threaded_daemon_with_a_request_timeout(self):
        self.build_ledger()
        server = self.serve()
        self.assertTrue(server.daemon_threads)
        self.assertEqual(srv.Handler.timeout, 10)
        # a stalled client must not block other requests
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as idle:
            idle.sendall(b"GET /api/health HTT")
            self.assertEqual(self.get("/api/health")[0], 200)


class HostOriginTests(ServerCase):
    def setUp(self) -> None:
        super().setUp()
        self.build_ledger()
        self.serve()

    def test_allowed_host_names(self):
        for host in (f"127.0.0.1:{self.port}", f"localhost:{self.port}", f"[::1]:{self.port}", f"LOCALHOST:{self.port}"):
            self.assertEqual(self.get("/api/health", headers={"Host": host})[0], 200, host)

    def test_wrong_host_is_403_everywhere(self):
        for host in ("evil.example", "evil.example:%d" % self.port, "127.0.0.1", "127.0.0.1:1", "localhost.evil.com:%d" % self.port,
                     "127.0.0.1:%d.evil.com" % self.port, "0.0.0.0:%d" % self.port, ""):
            for route in ROUTES:
                status, h, _ = self.get(route, headers={"Host": host})
                self.assertEqual(status, 403, (host, route))
                self.assert_security_headers(h)

    def test_missing_and_duplicate_host_headers_are_403(self):
        self.assertEqual(self.raw("GET /api/health HTTP/1.0\r\n\r\n")[0], 403)
        dup = f"GET /api/health HTTP/1.0\r\nHost: 127.0.0.1:{self.port}\r\nHost: evil.example\r\n\r\n"
        self.assertEqual(self.raw(dup)[0], 403)

    def test_foreign_or_null_origin_is_403(self):
        for origin in ("http://evil.example", "null", f"http://evil.example:{self.port}", f"https://127.0.0.1:{self.port}",
                       f"http://localhost:{self.port}"):      # a different name than the Host header is also foreign
            status, h, _ = self.get("/api/state", headers={"Origin": origin})
            self.assertEqual(status, 403, origin)
            self.assert_security_headers(h)

    def test_same_origin_is_accepted_and_gets_no_cors_headers(self):
        status, h, _ = self.get("/api/state", headers={"Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(status, 200)
        self.assert_security_headers(h)
        status, _, _ = self.get("/api/state", headers={"Host": f"localhost:{self.port}", "Origin": f"http://localhost:{self.port}"})
        self.assertEqual(status, 200)


class BindTests(ServerCase):
    def test_non_loopback_needs_allow_remote(self):
        self.build_ledger()
        for host in ("0.0.0.0", "192.0.2.1", "example.com", "::"):
            with self.assertRaises(ValueError):
                make_server(self.run_dir, host, 0, assets_dir=self.assets)

    def test_allow_remote_prints_a_warning(self):
        self.build_ledger()
        err = io.StringIO()
        with redirect_stderr(err):
            server = make_server(self.run_dir, "0.0.0.0", 0, allow_remote=True, assets_dir=self.assets)
        self.addCleanup(server.server_close)
        self.assertIn("WARNING", err.getvalue())
        self.assertIn("0.0.0.0", err.getvalue())

    def test_loopback_names_need_no_flag_and_print_no_warning(self):
        self.build_ledger()
        err = io.StringIO()
        with redirect_stderr(err):
            for host in ("127.0.0.1", "localhost"):
                make_server(self.run_dir, host, 0, assets_dir=self.assets).server_close()
        self.assertEqual(err.getvalue(), "")

    def test_remote_host_check_still_requires_the_bound_port(self):
        self.build_ledger()
        server = make_server(self.run_dir, "127.0.0.1", 0, allow_remote=True, assets_dir=self.assets)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        self.addCleanup(self._stop, server)
        self.port = server.server_address[1]
        self.assertEqual(self.get("/api/health", headers={"Host": f"build-box.lan:{self.port}"})[0], 200)
        self.assertEqual(self.get("/api/health", headers={"Host": "build-box.lan:1"})[0], 403)
        self.assertEqual(self.get("/api/health", headers={"Host": f"bad host:{self.port}"})[0], 403)

    def test_ipv6_loopback(self):
        if not socket.has_ipv6:
            self.skipTest("no IPv6")
        self.build_ledger()
        try:
            server = make_server(self.run_dir, "::1", 0, assets_dir=self.assets)
        except OSError:
            self.skipTest("::1 not bindable here")
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        self.addCleanup(self._stop, server)
        port = server.server_address[1]
        conn = http.client.HTTPConnection("::1", port, timeout=5)
        conn.request("GET", "/api/health", headers={"Host": f"[::1]:{port}"})
        self.assertEqual(conn.getresponse().status, 200)
        conn.close()


class StateAndEventsTests(ServerCase):
    def test_state_is_the_built_state_and_follows_the_ledger(self):
        led = self.build_ledger(close=False)
        self.serve()
        s = self.state()
        self.assertEqual(s["run"]["run_id"], led.run_id)
        self.assertIs(s["ledger"]["chain_ok"], True)
        self.assertEqual(s["ledger"]["last_seq"], led.events()[-1].seq)
        self.assertEqual(s["run"]["mode"], "scripted")
        before = s["ledger"]["last_seq"]
        led.append(Kind.SUP_NUDGE, "supervisor", {"text": "later"})
        led.append(Kind.SUP_NUDGE, "supervisor", {"text": "later still"})
        after = self.state()
        self.assertEqual(after["ledger"]["last_seq"], before + 2)
        self.assertEqual(after["ledger"]["head_anchor"].split(":")[0], str(before + 2))

    def test_server_waits_for_a_ledger_that_does_not_exist_yet(self):
        self.serve()
        s = self.state()
        self.assertEqual(s["ledger"]["events"], 0)
        led = Ledger.create(self.run_dir, clock=FakeClock(), fsync=False)
        self.addCleanup(led.close)
        RunBuilder(led).start("late")
        self.assertEqual(self.state()["run"]["task"], "late")

    def test_a_broken_chain_is_still_served_live_but_flagged(self):
        self.build_ledger()
        path = self.run_dir / "ledger.jsonl"
        lines = path.read_text().splitlines()
        row = json.loads(lines[4])
        row["data"]["note"] = "edited"
        lines[4] = json.dumps(row, sort_keys=True)
        path.write_text("\n".join(lines) + "\n")
        self.serve()
        s = self.state()
        self.assertIs(s["ledger"]["chain_ok"], False)
        self.assertEqual(s["ledger"]["bad_seq"], 5)

    def test_server_never_writes_to_the_run_directory(self):
        self.build_ledger()
        snap = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in self.run_dir.iterdir()}
        self.serve()
        for route in ROUTES:
            self.get(route)
        self.assertEqual({p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in self.run_dir.iterdir()}, snap)

    def test_events_endpoint(self):
        led = self.build_ledger()
        self.serve()
        n = len(led.events())
        d = json.loads(self.get("/api/events")[2])
        self.assertEqual((d["since"], d["last_seq"], d["count"], d["has_more"]), (0, n, n, False))
        self.assertEqual({k for k in d["events"][0]}, {"seq", "ts", "kind", "actor", "text"})
        d = json.loads(self.get("/api/events?since=%d" % (n - 2))[2])
        self.assertEqual([e["seq"] for e in d["events"]], [n - 1, n])
        d = json.loads(self.get("/api/events?since=%d" % n)[2])
        self.assertEqual(d["events"], [])

    def test_events_do_not_expose_sealed_numbers_or_raw_data(self):
        self.build_ledger()
        self.serve()
        body = self.get("/api/events")[2].decode()
        self.assertNotIn("0.314159", body)
        self.assertNotIn("prev_hash", body)

    def test_fuzzed_since_values_never_break_the_endpoint(self):
        led = self.build_ledger()
        self.serve()
        n = len(led.events())
        fuzz = ["", "abc", "-1", "1e3", "99999999999999999999999999", "%00", "1;2", "%D9%A1%D9%A2", "%205", "5%20", "0x10", "1_0",
                "%2B3", "NaN", "inf", "-0", "1.5", "'", "%27%20OR%201%3D1", "9" * 100, "%C3%A9", "%FF%FE", "0", "00003"]
        for value in fuzz:
            status, h, body = self.get("/api/events?since=" + value)
            self.assertEqual(status, 200, value)
            d = json.loads(body)
            self.assertIsInstance(d["since"], int)
            self.assertGreaterEqual(d["since"], 0)
            self.assertLessEqual(d["count"], n)
            self.assert_security_headers(h)
        self.assertEqual(json.loads(self.get("/api/events?since=abc")[2])["since"], 0)
        self.assertEqual(json.loads(self.get("/api/events?since=00003")[2])["since"], 3)
        self.assertEqual(json.loads(self.get("/api/events?since=2&since=9")[2])["since"], 2)
        many = "&".join(f"k{i}=1" for i in range(200))
        self.assertEqual(self.get("/api/events?" + many)[0], 200)

    def test_events_are_capped_at_500(self):
        led = Ledger.create(self.run_dir, clock=FakeClock(), fsync=False)
        RunBuilder(led).start()
        for i in range(700):
            led.append(Kind.SUP_NUDGE, "supervisor", {"text": str(i)})
        led.close()
        self.serve()
        d = json.loads(self.get("/api/events")[2])
        self.assertEqual((d["count"], d["has_more"]), (500, True))
        self.assertEqual(d["events"][-1]["seq"], 500)
        d2 = json.loads(self.get("/api/events?since=500&limit=99999")[2])
        self.assertEqual((d2["count"], d2["has_more"], d2["events"][0]["seq"]), (201, False, 501))
        self.assertEqual(json.loads(self.get("/api/events?limit=7")[2])["count"], 7)

    def test_health(self):
        led = self.build_ledger()
        self.serve()
        d = json.loads(self.get("/api/health")[2])
        # the run is scripted, so health must not call it live; "live" is only how the ledger is being served
        self.assertEqual((d["ok"], d["mode"], d["origin"], d["serving"], d["run_id"], d["last_seq"]),
                         (True, "scripted", "scripted", "live", led.run_id, led.events()[-1].seq))
        self.assertEqual(d["mode"], self.state()["run"]["mode"], "health and state use one vocabulary")

    def test_health_names_the_real_origin_of_every_kind_of_run(self):
        """scripted, simulated, live (meta-api, custom host, older ledger with no endpoint record), unknown and an empty dir."""
        cases = {
            "scripted": ("scripted", {}),
            "simulated": ("meta", {"endpoint": {"kind": "loopback", "hosts": ["127.0.0.1:41873"]}}),
            "live": ("meta", {"endpoint": {"kind": "meta-api", "hosts": ["api.meta.ai"]}}),
            "custom": ("meta", {"endpoint": {"kind": "custom", "hosts": ["models.partner.example"]}}),
            "older": ("meta", {}),
            "contradiction": ("meta", {"endpoint": {"kind": "scripted", "hosts": []}}),
        }
        want = {"scripted": "scripted", "simulated": "simulated", "live": "live", "custom": "live", "older": "live",
                "contradiction": "unknown"}
        for name, (backend, config) in cases.items():
            with self.subTest(name):
                self.run_dir = self.root / f"run-{name}"
                led = Ledger.create(self.run_dir, clock=FakeClock(), fsync=False)
                self.addCleanup(led.close)
                led.append(Kind.RUN_STARTED, "engine", {
                    "task": "law", "backend": backend, "models": {}, "limits": {}, "task_hash": "t" * 12,
                    "grader_hash": "g" * 12, "objective": "minimize", "primary_metric": "m", "config": config})
                RunBuilder(led).baseline(1.0)
                led.close()
                self.serve()
                d = json.loads(self.get("/api/health")[2])
                self.assertEqual((d["mode"], d["origin"], d["serving"]), (want[name], want[name], "live"))
                self.assertEqual(d["mode"], self.state()["run"]["mode"])

    def test_health_of_a_ledger_with_no_run_is_unknown_not_live(self):
        led = Ledger.create(self.run_dir, clock=FakeClock(), fsync=False)
        led.close()
        self.serve()
        d = json.loads(self.get("/api/health")[2])
        self.assertEqual((d["mode"], d["origin"], d["serving"], d["events"]), ("unknown", "unknown", "live", 0))

    def test_parse_since(self):
        self.assertEqual([parse_since(x) for x in (None, "", "7", "007", "-7", "7.0", "٣", "9" * 16, " 7")], [0, 0, 7, 7, 0, 0, 0, 0, 0])


class ReplayTests(ServerCase):
    def build_long(self, n_events: int = 40) -> Ledger:
        led = Ledger.create(self.run_dir, clock=FakeClock(start=5000.0, step=1.0), fsync=False)
        rb = RunBuilder(led)
        rb.start("law", backend="scripted")
        rb.baseline(1.0, 0.9)
        while len(led.events()) < n_events - 1:
            rb.experiment("A", f"h{len(led.events())}", score=0.9, delta=0.0)
        rb.finish()
        led.close()
        return led

    def test_replay_state_is_a_growing_consistent_prefix(self):
        self.build_long()
        clock = FakeMono()
        self.serve(replay_speed=10.0, clock=clock)
        total = len(Ledger.read_only(self.run_dir).events())
        positions = []
        for step in range(0, 60):
            clock.t = 1000.0 + step * 0.1          # 10x: one ledger second per 0.1 s
            s = self.state()
            self.assertEqual(s["run"]["mode"], "replay")
            rep = s["replay"]
            self.assertEqual((rep["speed"], rep["total_seq"]), (10.0, total))
            self.assertEqual(rep["position_seq"], s["ledger"]["last_seq"])
            self.assertEqual(rep["source_run_id"], s["run"]["run_id"])
            self.assertIs(s["ledger"]["chain_ok"], True)
            positions.append(rep["position_seq"])
        self.assertEqual(positions, sorted(positions))
        self.assertGreater(len(set(positions)), 10)
        self.assertEqual(positions[0], 1)
        self.assertEqual(positions[-1], total)
        self.assertTrue(s["run"]["finished"])

    def test_every_replay_state_equals_build_state_of_the_same_prefix(self):
        self.build_long()
        clock = FakeMono()
        self.serve(replay_speed=10.0, clock=clock)
        events = Ledger.read_only(self.run_dir).events()
        for t in (0.0, 0.35, 1.0, 2.2, 3.9):
            clock.t = 1000.0 + t
            got = self.state()
            k = got["ledger"]["last_seq"]
            prefix = events[:k]
            want = build_state(build_view(prefix), None, events=prefix, now=prefix[-1].ts,
                               replay={"speed": 10.0, "position_seq": k, "total_seq": events[-1].seq,
                                       "source_run_id": events[0].run_id})
            self.assertEqual(got, json.loads(json.dumps(want)))
            self.assertLessEqual(len(got["experiments"]), k)

    def test_replay_at_100x_in_real_time_grows_monotonically_to_the_end(self):
        self.build_long(40)                          # ~39 ledger seconds, about 0.4 s at 100x
        self.serve(replay_speed=100.0)
        total = len(Ledger.read_only(self.run_dir).events())
        seen = []
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            seen.append(self.state()["replay"]["position_seq"])
            if seen[-1] == total:
                break
            time.sleep(0.02)
        self.assertEqual(seen[-1], total)
        self.assertEqual(seen, sorted(seen))
        self.assertGreater(len(set(seen)), 2, "the prefix should grow over several polls, not jump straight to the end")
        self.assertEqual(self.state()["replay"]["position_seq"], total)

    def test_replay_duration_rescales_the_whole_ledger(self):
        self.build_long(41)                          # span of 40 ledger seconds
        clock = FakeMono()
        self.serve(replay_duration=4.0, clock=clock)
        self.assertEqual(self.state()["replay"]["position_seq"], 1)     # the first request starts the replay clock
        clock.t = 1000.0 + 2.0
        mid = self.state()["replay"]
        evs = Ledger.read_only(self.run_dir).events()
        self.assertAlmostEqual(mid["speed"], (evs[-1].ts - evs[0].ts) / 4.0)
        self.assertTrue(1 < mid["position_seq"] < mid["total_seq"])
        clock.t = 1000.0 + 4.0
        self.assertEqual(self.state()["replay"]["position_seq"], mid["total_seq"])

    def test_replay_events_endpoint_is_prefix_limited(self):
        self.build_long()
        clock = FakeMono()
        self.serve(replay_speed=10.0, clock=clock)
        self.state()                                                    # the first request starts the replay clock
        clock.t = 1000.0 + 0.5
        d = json.loads(self.get("/api/events")[2])
        self.assertEqual(d["last_seq"], self.state()["replay"]["position_seq"])
        self.assertGreater(d["last_seq"], 1)
        self.assertLess(d["last_seq"], len(Ledger.read_only(self.run_dir).events()))
        h = json.loads(self.get("/api/health")[2])
        # a replay of a scripted run says replay for the serving mode and scripted for what it is a replay of
        self.assertEqual((h["mode"], h["origin"], h["serving"]), ("replay", "scripted", "replay"))

    def test_replay_clock_starts_at_the_first_request_not_at_construction(self):
        """A 10x replay must not be mostly played by the time a browser opens: 300 s pass before the first request."""
        self.build_long()
        clock = FakeMono()
        server = self.serve(replay_speed=10.0, clock=clock)
        total = len(Ledger.read_only(self.run_dir).events())
        self.assertFalse(server.replay_started)
        clock.t += 300.0                                   # far longer than the 39 ledger seconds the replay covers at 10x
        first = self.state()["replay"]
        self.assertTrue(server.replay_started)
        self.assertEqual((first["position_seq"], first["total_seq"]), (1, total))
        clock.t += 1.0                                     # 1 s at 10x = 10 ledger seconds
        mid = self.state()["replay"]["position_seq"]
        self.assertTrue(1 < mid < total, mid)
        clock.t += 300.0
        self.assertEqual(self.state()["replay"]["position_seq"], total)

    def test_health_and_static_requests_do_not_start_the_replay_clock(self):
        self.build_long()
        clock = FakeMono()
        server = self.serve(replay_speed=10.0, clock=clock)
        h = json.loads(self.get("/api/health")[2])
        self.assertEqual((h["serving"], h["mode"], h["origin"], h["last_seq"]), ("replay", "replay", "scripted", 1))
        for path in ("/", "/app.js", "/app.css", "/api/health"):
            self.assertEqual(self.get(path)[0], 200)
        clock.t += 100.0
        self.assertFalse(server.replay_started)
        self.assertEqual(json.loads(self.get("/api/health")[2])["last_seq"], 1)
        self.assertEqual(self.state()["replay"]["position_seq"], 1)      # the first state request is where replay begins
        self.assertTrue(server.replay_started)

    def test_the_events_endpoint_also_starts_the_replay_clock(self):
        self.build_long()
        clock = FakeMono()
        server = self.serve(replay_speed=10.0, clock=clock)
        clock.t += 100.0
        d = json.loads(self.get("/api/events")[2])
        self.assertTrue(server.replay_started)
        self.assertEqual(d["last_seq"], 1)

    def test_a_replay_of_a_scripted_or_simulated_run_keeps_its_banner_and_names_its_origin(self):
        for name, backend, config, banner, origin in (
                ("s", "scripted", {}, BANNER, "scripted"),
                ("m", "meta", {"endpoint": {"kind": "loopback", "hosts": ["127.0.0.1"]}}, SIMULATED_BANNER, "simulated"),
                ("l", "meta", {"endpoint": {"kind": "meta-api", "hosts": ["api.meta.ai"]}}, None, "live")):
            with self.subTest(name):
                self.run_dir = self.root / f"replay-{name}"
                led = Ledger.create(self.run_dir, clock=FakeClock(start=5000.0, step=1.0), fsync=False)
                led.append(Kind.RUN_STARTED, "engine", {
                    "task": "law", "backend": backend, "models": {}, "limits": {}, "task_hash": "t" * 12, "grader_hash": "g" * 12,
                    "objective": "minimize", "primary_metric": "m", "config": config})
                RunBuilder(led).baseline(1.0)
                led.close()
                clock = FakeMono()
                self.serve(replay_speed=10.0, clock=clock)
                st = self.state()
                self.assertEqual((st["run"]["mode"], st["run"]["origin"], st["banner"]), ("replay", origin, banner))
                self.assertEqual(st["replay"]["source_run_id"], st["run"]["run_id"])
                h = json.loads(self.get("/api/health")[2])
                self.assertEqual((h["mode"], h["origin"], h["serving"]), ("replay", origin, "replay"))

    def test_replay_refuses_a_broken_chain(self):
        self.build_long()
        path = self.run_dir / "ledger.jsonl"
        lines = path.read_text().splitlines()
        row = json.loads(lines[10])
        row["data"]["edited"] = True
        lines[10] = json.dumps(row, sort_keys=True)
        path.write_text("\n".join(lines) + "\n")
        with self.assertRaises(ReplayRefused) as cm:
            make_server(self.run_dir, "127.0.0.1", 0, replay_speed=10, assets_dir=self.assets)
        self.assertIn("11", str(cm.exception))

    def test_replay_refuses_a_corrupt_or_empty_ledger(self):
        self.build_long()
        path = self.run_dir / "ledger.jsonl"
        lines = path.read_text().splitlines()
        lines[3] = "{not json"
        path.write_text("\n".join(lines) + "\n")
        with self.assertRaises(ReplayRefused):
            make_server(self.run_dir, "127.0.0.1", 0, replay_speed=10, assets_dir=self.assets)
        path.write_text("")
        with self.assertRaises(ReplayRefused):
            make_server(self.run_dir, "127.0.0.1", 0, replay_speed=10, assets_dir=self.assets)

    def test_a_refused_replay_does_not_leave_the_port_bound(self):
        self.build_long()
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        path = self.run_dir / "ledger.jsonl"
        path.write_text(path.read_text().replace('"kind"', '"kind "', 1))
        with self.assertRaises(ReplayRefused):
            make_server(self.run_dir, "127.0.0.1", port, replay_speed=10, assets_dir=self.assets)
        probe = socket.socket()
        probe.bind(("127.0.0.1", port))      # would fail with EADDRINUSE if the refused server still held the port
        probe.close()

    def test_bad_replay_settings(self):
        self.build_long()
        for kw in ({"replay_speed": 0.05}, {"replay_speed": 100.5}, {"replay_speed": float("nan")}, {"replay_speed": float("inf")},
                   {"replay_speed": True}, {"replay_speed": "fast"}, {"replay_duration": 0}, {"replay_duration": -3},
                   {"replay_duration": float("nan")}, {"replay_speed": 5, "replay_duration": 5}):
            with self.assertRaises(ValueError, msg=str(kw)):
                make_server(self.run_dir, "127.0.0.1", 0, assets_dir=self.assets, **kw)
        for ok in (0.1, 1, 100, 100.0):
            make_server(self.run_dir, "127.0.0.1", 0, replay_speed=ok, assets_dir=self.assets).server_close()

    def test_replay_never_modifies_the_ledger(self):
        self.build_long()
        before = (self.run_dir / "ledger.jsonl").read_bytes()
        names = sorted(os.listdir(self.run_dir))
        self.serve(replay_speed=100.0)
        self.state()
        self.assertEqual((self.run_dir / "ledger.jsonl").read_bytes(), before)
        self.assertEqual(sorted(os.listdir(self.run_dir)), names)


class ModeOverHttpTests(ServerCase):
    """The served state labels where the model words came from, live and in replay."""

    def build_with_endpoint(self, backend: str, endpoint: dict | None) -> None:
        led = Ledger.create(self.run_dir, clock=FakeClock(start=5000.0, step=1.0), fsync=False)
        cfg = {"endpoint": endpoint} if endpoint else {}
        led.append(Kind.RUN_STARTED, "engine", {"task": "law", "backend": backend, "models": {"supervisor": "muse-spark-1.3"},
                                                "limits": {}, "task_hash": "t" * 12, "grader_hash": "g" * 12,
                                                "objective": "minimize", "primary_metric": "m", "config": cfg})
        RunBuilder(led).baseline(1.0)
        led.close()

    def state(self) -> dict:
        status, _, body = self.get("/api/state")
        self.assertEqual(status, 200)
        return json.loads(body)

    def test_a_loopback_meta_run_is_served_as_simulated_with_the_banner(self):
        self.build_with_endpoint("meta", {"kind": "loopback", "hosts": ["127.0.0.1:41873"]})
        self.serve()
        s = self.state()
        self.assertEqual((s["run"]["mode"], s["run"]["origin"], s["banner"]), ("simulated", "simulated", SIMULATED_BANNER))
        self.assertEqual(s["run"]["endpoint"], {"kind": "loopback", "hosts": ["127.0.0.1:41873"]})

    def test_a_meta_api_run_is_live_without_a_banner(self):
        self.build_with_endpoint("meta", {"kind": "meta-api", "hosts": ["api.meta.ai"]})
        self.serve()
        s = self.state()
        self.assertEqual((s["run"]["mode"], s["banner"]), ("live", None))

    def test_a_scripted_run_keeps_the_scripted_banner(self):
        self.build_with_endpoint("scripted", {"kind": "scripted", "hosts": []})
        self.serve()
        s = self.state()
        self.assertEqual((s["run"]["mode"], s["banner"]), ("scripted", BANNER))

    def test_replay_of_a_simulated_run_is_replay_mode_but_keeps_the_simulated_banner(self):
        self.build_with_endpoint("meta", {"kind": "loopback", "hosts": ["127.0.0.1:41873"]})
        self.serve(replay_speed=10.0, clock=FakeMono())
        s = self.state()
        self.assertEqual((s["run"]["mode"], s["run"]["origin"], s["banner"]), ("replay", "simulated", SIMULATED_BANNER))

    def test_the_events_feed_names_a_loopback_endpoint_in_the_run_started_line(self):
        self.build_with_endpoint("meta", {"kind": "loopback", "hosts": ["127.0.0.1:41873"]})
        self.serve()
        status, _, body = self.get("/api/events")
        rows = json.loads(body)["events"]
        self.assertIn("loopback endpoint", rows[0]["text"])


class AdversarialServerTests(ServerCase):
    def test_the_packaged_frontend_files_are_served_with_the_full_header_set_when_present(self):
        present = [n for n in ("index.html", "app.js", "app.css") if (srv.PACKAGE_DIR / n).is_file()]
        if len(present) < 3:
            self.skipTest("the frontend files are not in the package directory yet")
        self.build_ledger()
        self.serve(assets_dir=srv.PACKAGE_DIR)
        for route, name in (("/", "index.html"), ("/app.js", "app.js"), ("/app.css", "app.css")):
            status, h, body = self.get(route)
            self.assertEqual(status, 200, route)
            self.assertEqual(body, (srv.PACKAGE_DIR / name).read_bytes())
            self.assert_security_headers(h)

    def test_many_concurrent_readers_and_a_live_writer_always_see_a_consistent_chain(self):
        led = self.build_ledger(close=False)
        self.serve()
        stop = threading.Event()
        problems: list = []

        def write() -> None:
            i = 0
            while not stop.is_set() and i < 1500:
                led.append(Kind.SUP_NUDGE, "supervisor", {"text": f"n{i}"})
                i += 1
                time.sleep(0.0005)

        def read() -> None:
            last = 0
            for _ in range(25):
                try:
                    s = self.state()
                except Exception as exc:  # noqa: BLE001 - any failure is a finding
                    problems.append(repr(exc))
                    return
                if s["ledger"]["chain_ok"] is not True or s["ledger"]["last_seq"] < last:
                    problems.append((last, s["ledger"]))
                last = s["ledger"]["last_seq"]
        writer = threading.Thread(target=write)
        writer.start()
        readers = [threading.Thread(target=read) for _ in range(4)]
        for t in readers:
            t.start()
        for t in readers:
            t.join(60)
        stop.set()
        writer.join(10)
        self.assertEqual(problems, [])

    def test_a_request_that_cannot_be_parsed_still_gets_the_security_headers(self):
        self.build_ledger()
        self.serve()
        for text in ("GET /" + "a" * 70000 + " HTTP/1.0\r\nHost: 127.0.0.1:%d\r\n\r\n" % self.port,
                     "GET / HTTP/1.0\r\nHost: 127.0.0.1:%d\r\n" % self.port + "X: y\r\n" * 200 + "\r\n",
                     "GET /api/health HTTP/9.9\r\nHost: x\r\n\r\n"):
            status, h, _ = self.raw(text)
            self.assertIn(status, (400, 414, 431, 505))
            self.assert_security_headers(h)


class RequestTimeoutTests(ServerCase):
    def test_an_idle_client_is_disconnected_when_the_request_timeout_expires(self):
        self.build_ledger()
        old = srv.Handler.timeout
        srv.Handler.timeout = 0.4            # same mechanism as the production 10 s, short enough to wait for
        self.addCleanup(setattr, srv.Handler, "timeout", old)
        self.serve()
        started = time.monotonic()
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as idle:
            idle.sendall(b"GET /api/health HTT")
            data = idle.recv(65536)           # the server gives up and closes: EOF, not an answer to half a request
        self.assertEqual(data, b"")
        self.assertLess(time.monotonic() - started, 4)
        self.assertEqual(self.get("/api/health")[0], 200)
        self.assertEqual(old, 10)


if __name__ == "__main__":
    unittest.main()
