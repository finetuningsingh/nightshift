import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from nightshift.lab.contracts import (
    CreateAmbiguous,
    CreateRejected,
    GpuProvider,
    OwnershipError,
    ProviderUnavailable,
    Shape,
    owner_tag_of_key,
)
from nightshift.lab.gpu_thunder import ThunderProvider

FIX = Path(__file__).parent / "fixtures" / "lab" / "thunder"
TOKEN = "tok-SECRET-VALUE-12345"
OURS = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOURKEYOURKEYOURKEYOURKEYOURKEYOURKEYOURKEYxx lab-test"
SHAPE = Shape("a100", 1, 12, 300)


def fixture(name: str):
    return json.loads((FIX / name).read_text())


class FakeThunder:
    """A loopback Thunder REST server. Routes return queued/static replies; every request is recorded."""

    def __init__(self):
        self.requests = []
        self.routes = {}  # (method, path) -> (status, body) or callable(req) -> (status, body) or Exception-name string
        outer = self

        class H(BaseHTTPRequestHandler):
            def _serve(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                req = {"method": self.command, "path": self.path, "headers": dict(self.headers), "body": json.loads(body) if body else None}
                outer.requests.append(req)
                route = outer.routes.get((self.command, self.path))
                if route is None:
                    route = (404, {"code": 404, "error": "not_found", "message": "no route"})
                if callable(route):
                    route = route(req)
                if route == "hang":
                    time.sleep(2)
                    return
                if route == "drop":
                    self.connection.close()
                    return
                status, payload = route
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = do_POST = _serve

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def of(self, method):
        return [r for r in self.requests if r["method"] == method]


class ThunderTests(unittest.TestCase):
    def setUp(self):
        self.srv = FakeThunder()
        self.addCleanup(self.srv.close)
        patcher = mock.patch.dict(os.environ, {"LAB_TEST_THUNDER_TOKEN": TOKEN})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.p = ThunderProvider("LAB_TEST_THUNDER_TOKEN", self.srv.url, timeout=1.0)
        self.srv.routes[("GET", "/instances/list")] = (200, fixture("list_two_instances.json"))
        self.srv.routes[("GET", "/v2/pricing")] = (200, fixture("pricing.json"))
        self.srv.routes[("GET", "/v2/status")] = (200, fixture("status.json"))

    # protocol and construction ---------------------------------------------------------------------
    def test_is_a_gpu_provider(self):
        self.assertIsInstance(self.p, GpuProvider)

    def test_token_must_travel_over_https_or_loopback(self):
        with self.assertRaises(ProviderUnavailable):
            ThunderProvider("LAB_TEST_THUNDER_TOKEN", "http://api.thundercompute.com:8443")
        ThunderProvider("LAB_TEST_THUNDER_TOKEN", "https://api.thundercompute.com:8443")
        with self.assertRaises(ProviderUnavailable):
            ThunderProvider("not an env name")

    def test_repr_has_no_token(self):
        self.assertNotIn(TOKEN, repr(self.p))

    # list ------------------------------------------------------------------------------------------
    def test_list_parses_rows_and_tags(self):
        rows = {i.handle: i for i in self.p.list_instances()}
        self.assertEqual(sorted(rows), ["ab12cd34", "od510vbf"])
        ours = rows["ab12cd34"]
        self.assertEqual(ours.status, "RUNNING")
        self.assertEqual((ours.host, ours.port, ours.user), ("203.0.113.11", 31023, "ubuntu"))
        self.assertEqual(ours.owner_tag, owner_tag_of_key(OURS))
        self.assertEqual(ours.shape, Shape("a100xl", 1, 12, 300, "base"))
        self.assertNotEqual(rows["od510vbf"].owner_tag, ours.owner_tag)
        self.assertGreater(ours.created_epoch, 1.7e9)

    def test_requests_carry_a_bearer_token_and_a_custom_user_agent(self):
        self.p.list_instances()
        h = self.srv.requests[-1]["headers"]
        self.assertEqual(h["Authorization"], f"Bearer {TOKEN}")
        self.assertNotIn("Python-urllib", h["User-Agent"])

    def test_pending_row_has_no_address(self):
        self.srv.routes[("GET", "/instances/list")] = (200, fixture("list_pending.json"))
        (inst,) = self.p.list_instances()
        self.assertEqual(inst.status, "STARTING")
        self.assertIsNone(inst.host)

    def test_list_never_returns_empty_on_failure(self):
        cases = {
            "http 500": (500, {"code": 500, "error": "boom", "message": "x"}),
            "http 401": (401, {"code": 401, "error": "unauthorized", "message": "bad token"}),
            "not json": (200, b"<html>gateway</html>"),
            "a list, not a map": (200, []),
            "row without uuid": (200, {"0": {"status": "RUNNING"}}),
            "row not an object": (200, {"0": "x"}),
            "bad createdAt": (200, {"0": {"uuid": "u1", "status": "RUNNING", "createdAt": 5}}),
        }
        for name, route in cases.items():
            self.srv.routes[("GET", "/instances/list")] = route
            with self.subTest(name), self.assertRaises(ProviderUnavailable):
                self.p.list_instances()

    def test_list_timeout_and_dropped_connection(self):
        for route in ("hang", "drop"):
            self.srv.routes[("GET", "/instances/list")] = route
            with self.subTest(route), self.assertRaises(ProviderUnavailable):
                self.p.list_instances()

    def test_connection_refused(self):
        p = ThunderProvider("LAB_TEST_THUNDER_TOKEN", "http://127.0.0.1:1", timeout=1.0)
        with self.assertRaises(ProviderUnavailable):
            p.list_instances()
        self.assertIsNone(p.exists("x"))

    def test_missing_token_is_unavailable_and_never_sent(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LAB_TEST_THUNDER_TOKEN")
            with self.assertRaises(ProviderUnavailable):
                self.p.list_instances()
            with self.assertRaises(CreateRejected):
                self.p.create(SHAPE, OURS)
        self.assertEqual(self.srv.requests, [])

    def test_error_text_never_contains_the_token(self):
        self.srv.routes[("GET", "/instances/list")] = (500, {"code": 500, "error": "x", "message": f"echo {TOKEN}"})
        try:
            self.p.list_instances()
        except ProviderUnavailable as exc:
            # the message is cut to 200 characters of the service's text; the token must be redacted if the service echoes it
            self.assertNotIn(TOKEN, str(exc))

    def test_redirects_are_not_followed(self):
        self.srv.routes[("GET", "/instances/list")] = lambda req: (302, {})
        with self.assertRaises(ProviderUnavailable):
            self.p.list_instances()

    # exists / find_created -------------------------------------------------------------------------
    def test_exists_is_three_valued(self):
        self.assertTrue(self.p.exists("ab12cd34"))
        self.assertFalse(self.p.exists("zzzz"))
        self.srv.routes[("GET", "/instances/list")] = (500, {})
        self.assertIsNone(self.p.exists("ab12cd34"))

    def test_find_created_matches_tag_shape_and_time_and_never_a_foreign_row(self):
        tag = owner_tag_of_key(OURS)
        created = self.p.list_instances()[1].created_epoch
        self.assertEqual(self.p.find_created(set(), SHAPE, created, tag), "ab12cd34")
        # a foreign instance of EXACTLY our shape is never adopted: wrong tag
        rows = fixture("list_two_instances.json")
        rows["0"].update(cpuCores="12", storage=300)
        self.srv.routes[("GET", "/instances/list")] = (200, rows)
        self.assertEqual(self.p.find_created(set(), SHAPE, created, tag), "ab12cd34")
        self.assertIsNone(self.p.find_created(set(), SHAPE, created, "0" * 16))
        self.assertIsNone(self.p.find_created({"ab12cd34"}, SHAPE, created, tag))  # in the before-set
        self.assertIsNone(self.p.find_created(set(), SHAPE, created + 10_000, tag))  # created before the attempt window
        self.assertIsNone(self.p.find_created(set(), Shape("a100", 1, 8, 100), created, tag))  # other shape

    def test_find_created_returns_none_for_two_matches(self):
        rows = fixture("list_two_instances.json")
        rows["0"]["sshPublicKeys"] = rows["1"]["sshPublicKeys"]
        rows["0"].update(cpuCores="12", storage=300, createdAt=rows["1"]["createdAt"])
        self.srv.routes[("GET", "/instances/list")] = (200, rows)
        created = self.p.list_instances()[1].created_epoch
        self.assertIsNone(self.p.find_created(set(), SHAPE, created, owner_tag_of_key(OURS)))

    # quote -----------------------------------------------------------------------------------------
    def test_quote_includes_extra_vcpus_and_disk(self):
        q = self.p.quote(SHAPE)  # a100 x1: 1.09 + (12-8)*0.04 + (300-100)*0.0003 = 1.31
        self.assertAlmostEqual(q.usd_per_hour, 1.31, places=6)
        self.assertTrue(q.available)
        self.assertEqual(q.fixed_usd, 0.0)

    def test_quote_unavailable_and_unknown(self):
        self.assertFalse(self.p.quote(Shape("h100", 1, 12, 300)).available)
        self.assertAlmostEqual(self.p.quote(Shape("h100", 1, 12, 300)).usd_per_hour, 3.2 + 8 * 0.04 + 200 * 0.0003, places=6)
        self.srv.routes[("GET", "/v2/status")] = (500, {})
        self.assertIsNone(self.p.quote(SHAPE).available)

    def test_quote_unknown_shape_or_failure_raises(self):
        with self.assertRaises(ProviderUnavailable):
            self.p.quote(Shape("tpu", 1, 4, 100))
        with self.assertRaises(ProviderUnavailable):
            self.p.quote(Shape("a100", 3, 12, 300))  # no a100xl_x3 price
        self.srv.routes[("GET", "/v2/pricing")] = (500, {})
        with self.assertRaises(ProviderUnavailable):
            self.p.quote(SHAPE)

    def test_quote_needs_no_token(self):
        os.environ.pop("LAB_TEST_THUNDER_TOKEN")
        self.p.quote(SHAPE)
        self.assertNotIn("Authorization", self.srv.requests[0]["headers"])

    # create ----------------------------------------------------------------------------------------
    def test_create_sends_our_key_and_parses_the_receipt(self):
        self.srv.routes[("POST", "/instances/create")] = (201, fixture("create_201.json"))
        receipt = self.p.create(SHAPE, OURS)
        self.assertEqual(receipt.handle, "ab12cd34")
        body = self.srv.of("POST")[0]["body"]
        self.assertEqual(body["public_key"], OURS)
        self.assertEqual((body["gpu_type"], body["num_gpus"], body["cpu_cores"], body["disk_size_gb"], body["template"]), ("a100xl", 1, 12, 300, "base"))

    def test_gpu_type_map_renames_the_request_value(self):
        p = ThunderProvider("LAB_TEST_THUNDER_TOKEN", self.srv.url, gpu_type_map={"a100": "A100XL"})
        self.srv.routes[("POST", "/instances/create")] = (201, fixture("create_201.json"))
        p.create(SHAPE, OURS)
        self.assertEqual(self.srv.of("POST")[0]["body"]["gpu_type"], "A100XL")

    def test_create_definite_refusal_is_rejected(self):
        for status in (400, 401, 403, 409, 422):
            self.srv.routes[("POST", "/instances/create")] = (status, fixture("error_400.json"))
            with self.subTest(status), self.assertRaises(CreateRejected):
                self.p.create(SHAPE, OURS)

    def test_create_unknown_outcome_is_ambiguous_and_never_retried(self):
        for route in ((500, {"code": 500}), (502, b"bad gateway"), (429, {}), "drop", "hang"):
            self.srv.requests.clear()
            self.srv.routes[("POST", "/instances/create")] = route
            with self.subTest(route), self.assertRaises(CreateAmbiguous):
                self.p.create(SHAPE, OURS)
            self.assertEqual(len(self.srv.of("POST")), 1)

    def test_create_success_without_a_handle_is_a_receipt_with_none(self):
        self.srv.routes[("POST", "/instances/create")] = (201, b"{}")
        self.assertIsNone(self.p.create(SHAPE, OURS).handle)

    def test_create_refuses_an_untaggable_key_before_sending(self):
        from nightshift.lab.contracts import ContractError

        with self.assertRaises(ContractError):
            self.p.create(SHAPE, "notakey")
        self.assertEqual(self.srv.requests, [])

    # delete ----------------------------------------------------------------------------------------
    def test_delete_checks_the_tag_then_addresses_the_current_index(self):
        self.srv.routes[("POST", "/instances/1/delete")] = (200, {"message": "Success"})
        self.p.delete("ab12cd34", owner_tag_of_key(OURS))
        posts = self.srv.of("POST")
        self.assertEqual([r["path"] for r in posts], ["/instances/1/delete"])  # the position of OUR row, not of the foreign one

    def test_delete_follows_a_changed_position(self):
        rows = fixture("list_two_instances.json")
        self.srv.routes[("GET", "/instances/list")] = (200, {"0": rows["1"]})  # the foreign row is gone: ours moved to position 0
        self.srv.routes[("POST", "/instances/0/delete")] = (200, {})
        self.p.delete("ab12cd34", owner_tag_of_key(OURS))
        self.assertEqual(self.srv.of("POST")[0]["path"], "/instances/0/delete")

    def test_concurrent_deleters_cannot_shift_the_position_between_check_and_request(self):
        rows = fixture("list_two_instances.json")
        state = {"gone": False}

        def listing(req):
            return (200, {"0": rows["0"]} if state["gone"] else rows)

        def first_delete(req):
            time.sleep(0.4)  # a slow delete; a second deleter that listed meanwhile would still see our row at position 1
            state["gone"] = True
            return (200, {})

        self.srv.routes[("GET", "/instances/list")] = listing
        self.srv.routes[("POST", "/instances/1/delete")] = first_delete
        self.srv.routes[("POST", "/instances/0/delete")] = (200, {})  # what a stale second request would hit: the foreign row, now at 0
        tag = owner_tag_of_key(OURS)
        other = ThunderProvider("LAB_TEST_THUNDER_TOKEN", self.srv.url, timeout=5.0)
        t1 = threading.Thread(target=self.p.delete, args=("ab12cd34", tag))
        t2 = threading.Thread(target=other.delete, args=("ab12cd34", tag))
        t1.start()
        time.sleep(0.1)
        t2.start()
        t1.join()
        t2.join()
        self.assertEqual([r["path"] for r in self.srv.of("POST")], ["/instances/1/delete"])

    def test_delete_by_uuid_when_configured(self):
        p = ThunderProvider("LAB_TEST_THUNDER_TOKEN", self.srv.url, delete_by="uuid")
        self.srv.routes[("POST", "/instances/ab12cd34/delete")] = (200, {})
        p.delete("ab12cd34", owner_tag_of_key(OURS))
        self.assertEqual(self.srv.of("POST")[0]["path"], "/instances/ab12cd34/delete")

    def test_delete_of_a_foreign_row_sends_nothing(self):
        for handle, tag in (("od510vbf", owner_tag_of_key(OURS)), ("ab12cd34", "f" * 16), ("ab12cd34", None)):
            with self.subTest(handle=handle), self.assertRaises(OwnershipError):
                self.p.delete(handle, tag)
        self.assertEqual(self.srv.of("POST"), [])

    def test_delete_of_an_absent_handle_is_a_no_op(self):
        self.p.delete("gone0000", owner_tag_of_key(OURS))
        self.assertEqual(self.srv.of("POST"), [])

    def test_delete_failures_are_unavailable_not_success(self):
        self.srv.routes[("POST", "/instances/1/delete")] = (500, {"code": 500})
        with self.assertRaises(ProviderUnavailable):
            self.p.delete("ab12cd34", owner_tag_of_key(OURS))
        self.srv.routes[("POST", "/instances/1/delete")] = "drop"
        with self.assertRaises(ProviderUnavailable):
            self.p.delete("ab12cd34", owner_tag_of_key(OURS))

    def test_delete_when_the_listing_fails_sends_nothing(self):
        self.srv.routes[("GET", "/instances/list")] = (500, {})
        with self.assertRaises(ProviderUnavailable):
            self.p.delete("ab12cd34", owner_tag_of_key(OURS))
        self.assertEqual(self.srv.of("POST"), [])

    def test_delete_404_is_treated_as_already_gone(self):
        self.srv.routes[("POST", "/instances/1/delete")] = (404, {"code": 404})
        self.p.delete("ab12cd34", owner_tag_of_key(OURS))


class FixtureSanity(unittest.TestCase):
    def test_openapi_fixture_describes_the_endpoints_the_provider_calls(self):
        doc = fixture("openapi_instances_trimmed.json")
        for path in ("/instances/create", "/instances/list", "/instances/{id}/delete", "/v2/pricing", "/v2/status"):
            self.assertIn(path, doc["paths"])
        schema = doc["components"]["schemas"]["thundertypes.InstanceCreateRequest"]
        for field in ("cpu_cores", "disk_size_gb", "gpu_type", "num_gpus", "template", "public_key"):
            self.assertIn(field, schema["properties"])


if __name__ == "__main__":
    unittest.main()
