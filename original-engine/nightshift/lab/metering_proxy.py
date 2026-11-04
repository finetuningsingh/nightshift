"""Metering proxy: the only route from the researchers' LLM calls to the Meta API, with the hard stop.

Runs on the control host, which alone holds the Meta key (docs/harness-design.md D4, "Budget enforcement"). The researcher host reaches
it through an SSH reverse tunnel; CORAL's LiteLLM gateway sends its upstream calls here with the per-trial token as the API key.

Reserve, then settle (the design's overshoot bound):

* Before anything is forwarded the proxy calls ``meter.reserve_llm(model, prompt_tokens_estimate, max_tokens, call_id)``. The estimate is
  the request body's UTF-8 bytes divided by 3, ``max_tokens`` is the request's value or 16000 (``limit.output``) times ``n`` (completions
  asked for), and the forwarded body is rewritten so that upstream is told exactly that bound (``max_tokens`` and, when present,
  ``max_completion_tokens`` are set to the reserved per-completion value; ``n`` must be a positive integer or the request is refused with
  400). So what is reserved is what upstream is allowed to generate. If the meter says the
  worst case would cross a cap (``BudgetExhausted``) the proxy answers HTTP 429 and records the exception in :attr:`MeteringProxy.exhausted`
  for the trial runner, which stops the runtime. Nothing is forwarded.
* On the response ``usage`` it calls ``meter.charge_llm(model, LlmUsage, call_id)``, which replaces the reservation with the actual usage.
* A stream that ends without a ``usage`` chunk, a client that disconnects, or an upstream failure after the request was sent leaves the
  worst-case reservation booked: a missing number can only over-count. Two cases are settled at zero because nothing was generated:
  the upstream answered with an HTTP error status, or the proxy could not connect to it at all.
* ``stream_options.include_usage`` is forced on every streaming request (it is invalid on a non-streaming one, so it is not added there).

This module also holds the small HTTP mounting layer (:class:`LabHttpServer`, :class:`Mount`) that puts the proxy (``/v1``) and the control
API (``/control``) on one loopback ``ThreadingHTTPServer``. The researcher's token is never forwarded; the Meta key never leaves this
process and is scrubbed from any upstream error body before it is relayed. Request bodies are never logged.

Standard library only.
"""

from __future__ import annotations

import hmac
import http.client
import json
import math
import re
import secrets
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional, Sequence

from nightshift.lab.contracts import BudgetExhausted, BudgetMeter, Clock, ContractError, LlmUsage, loads_strict

MAX_REQUEST_BYTES = 32 << 20  # CORAL's context limit is 1,048,576 tokens, about 3 MB of text; this is generous
DEFAULT_MAX_TOKENS = 16000  # opencode.json limit.output
PROMPT_BYTES_PER_TOKEN = 3
UPSTREAM_TIMEOUT_S = 600.0
_MAX_SSE_LINE = 1 << 22
STOP_DRAIN_TIMEOUT_S = 30.0  # how long LabHttpServer.stop waits for requests still being served
_REDACTED = "[REDACTED]"


# --------------------------------------------------------------------------------------------------
# HTTP mounting layer shared by the proxy and the control API
# --------------------------------------------------------------------------------------------------
class RequestTooLarge(Exception):
    pass


class BadRequest(Exception):
    pass


class Mount:
    """A component served under ``prefix`` of a :class:`LabHttpServer`. ``handle`` answers the request completely."""

    prefix: str = ""

    def handle(self, h: BaseHTTPRequestHandler) -> None:  # pragma: no cover - interface
        raise NotImplementedError


_CHUNK_SIZE = re.compile(rb"[0-9A-Fa-f]{1,16}")


def read_request_body(h: BaseHTTPRequestHandler, limit: int) -> bytes:
    """The request body, from Content-Length or chunked encoding; ``RequestTooLarge`` above ``limit`` bytes."""
    te = (h.headers.get("Transfer-Encoding") or "").lower()
    if "chunked" in te:
        out = bytearray()
        while True:
            line = h.rfile.readline(1024)
            digits = line.split(b";", 1)[0].strip() or b"0"
            if not _CHUNK_SIZE.fullmatch(digits):  # int() would take "-1" (read(-1) reads to EOF past the limit), "+5", "0x5", "1_0"
                raise BadRequest("malformed chunked body")
            size = int(digits, 16)
            if size == 0:
                while True:  # trailers
                    t = h.rfile.readline(1024)
                    if t in (b"\r\n", b"\n", b""):
                        break
                break
            if len(out) + size > limit:
                raise RequestTooLarge()
            chunk = h.rfile.read(size)
            if len(chunk) != size:
                raise BadRequest("truncated chunked body")
            out += chunk
            h.rfile.readline(8)  # the CRLF after the chunk
        return bytes(out)
    raw = h.headers.get("Content-Length")
    if raw is None:
        return b""
    try:
        n = int(raw)
    except ValueError:
        raise BadRequest("bad Content-Length") from None
    if n < 0:
        raise BadRequest("bad Content-Length")
    if n > limit:
        raise RequestTooLarge()
    data = h.rfile.read(n)
    if len(data) != n:
        raise BadRequest("truncated body")
    return data


def send_json(h: BaseHTTPRequestHandler, status: int, obj: Any, headers: Optional[dict] = None) -> None:
    body = json.dumps(obj, sort_keys=True).encode("utf-8")
    send_bytes(h, status, body, "application/json", headers)


def send_bytes(h: BaseHTTPRequestHandler, status: int, body: bytes, content_type: str, headers: Optional[dict] = None) -> None:
    h.send_response(status)
    h.send_header("Content-Type", content_type)
    h.send_header("Content-Length", str(len(body)))
    h.send_header("Connection", "close")
    for k, v in (headers or {}).items():
        h.send_header(k, v)
    h.end_headers()
    if h.command != "HEAD":
        h.wfile.write(body)


def bearer_ok(h: BaseHTTPRequestHandler, token: str) -> bool:
    """Constant-time check of ``Authorization: Bearer <token>``."""
    value = h.headers.get("Authorization") or ""
    scheme, _, presented = value.partition(" ")
    if scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(presented.strip().encode("utf-8"), token.encode("utf-8"))


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # close-delimited bodies: streaming needs no chunk framing
    mounts: tuple = ()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - never log request lines (URLs can carry ids)
        return

    def _dispatch(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        for mount in self.mounts:
            if path == mount.prefix or path.startswith(mount.prefix.rstrip("/") + "/"):
                try:
                    mount.handle(self)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except RequestTooLarge:
                    self._safe_json(413, {"error": "request too large"})
                except BadRequest as exc:
                    self._safe_json(400, {"error": str(exc)})
                except Exception:  # noqa: BLE001 - never leak internals to a researcher
                    self._safe_json(500, {"error": "internal error"})
                return
        self._safe_json(404, {"error": "not found"})

    def _safe_json(self, status: int, obj: Any) -> None:
        try:
            send_json(self, status, obj)
        except Exception:  # noqa: BLE001
            pass

    do_GET = _dispatch  # noqa: N815
    do_POST = _dispatch  # noqa: N815


class LabHttpServer(ThreadingHTTPServer):
    """One loopback server for several mounts. ``port=0`` lets the OS choose; read it back from :attr:`port`."""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, mounts: Sequence[Mount], host: str = "127.0.0.1", port: int = 0):
        handler = type("LabHandler", (_Handler,), {"mounts": tuple(mounts)})
        super().__init__((host, port), handler)
        self._thread: Optional[threading.Thread] = None
        self._active = 0
        self._drained = False
        self._idle = threading.Condition()

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    @property
    def active_requests(self) -> int:
        """Requests whose handler is still running (for example an upstream call that has not answered yet)."""
        with self._idle:
            return self._active

    def process_request(self, request: Any, client_address: Any) -> None:
        with self._idle:  # counted before the handler thread exists, so stop() cannot miss an accepted request
            self._active += 1
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._request_done()
            raise

    def _request_done(self) -> None:
        with self._idle:
            self._active -= 1
            self._idle.notify_all()

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_done()

    def start(self) -> "LabHttpServer":
        self._thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.1}, name="lab-http", daemon=True)
        self._thread.start()
        return self

    def stop(self, drain_timeout: float = STOP_DRAIN_TIMEOUT_S) -> None:
        """Close the listener, then wait up to ``drain_timeout`` seconds for requests already being served to settle, so the caller does
        not seal its ledger under a pending upstream call. :attr:`active_requests` tells whether the wait ran out."""
        try:
            if self._thread is not None:
                self.shutdown()
        finally:
            self.server_close()
            if self._thread is not None:
                self._thread.join(timeout=5)
        if self._drained:
            return
        self._drained = True
        deadline = time.monotonic() + max(0.0, float(drain_timeout))
        with self._idle:
            while self._active > 0:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._idle.wait(left)


# --------------------------------------------------------------------------------------------------
# the proxy
# --------------------------------------------------------------------------------------------------
def estimate_prompt_tokens(request_bytes: int) -> int:
    """Conservative prompt-token estimate: UTF-8 bytes of the request divided by 3 (UNVERIFIED for the Muse tokenizer)."""
    return max(1, math.ceil(request_bytes / PROMPT_BYTES_PER_TOKEN))


def _positive_int(v: Any) -> Optional[int]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, float) and (not math.isfinite(v) or v != int(v)):
        return None
    return int(v) if v >= 1 else None


def request_max_tokens(body: dict) -> int:
    for key in ("max_tokens", "max_completion_tokens"):
        n = _positive_int(body.get(key))
        if n is not None:
            return n
    return DEFAULT_MAX_TOKENS


def request_n(body: dict) -> Optional[int]:
    """Completions requested: 1 when ``n`` is absent, the value when it is a positive integer, None when it is present but not one."""
    if "n" not in body:
        return 1
    return _positive_int(body["n"])


def usage_from(doc: Any) -> Optional[LlmUsage]:
    """``LlmUsage`` from a response (or final SSE chunk) holding a ``usage`` object; None when it is absent or malformed."""
    usage = doc.get("usage") if isinstance(doc, dict) else None
    if not isinstance(usage, dict):
        return None
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if _is_count(prompt) and _is_count(completion):
        cached = 0
        details = usage.get("prompt_tokens_details")
        for cand in (details.get("cached_tokens") if isinstance(details, dict) else None, usage.get("cached_tokens")):
            if _is_count(cand):
                cached = int(cand)
                break
        return LlmUsage(int(prompt), int(completion), min(cached, int(prompt)))
    return None


def _is_count(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 and v == int(v)


class MeteringProxy(Mount):
    """``POST /v1/chat/completions`` with ``Authorization: Bearer <token>``, forwarded to ``<upstream_base_url>/chat/completions``."""

    prefix = "/v1"

    def __init__(
        self,
        meter: BudgetMeter,
        upstream_base_url: str,
        api_key: str,
        token: str,
        *,
        clock: Clock,
        timeout: float = UPSTREAM_TIMEOUT_S,
        max_request_bytes: int = MAX_REQUEST_BYTES,
    ):
        parts = urllib.parse.urlsplit(upstream_base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("upstream_base_url must be an http(s) URL")
        if not api_key or not token:
            raise ValueError("api_key and token are required")
        self._meter = meter
        self._scheme, self._host, self._port = parts.scheme, parts.hostname, parts.port
        self._path = parts.path.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._token = token
        self._clock = clock
        self._timeout = float(timeout)
        self._max_request_bytes = int(max_request_bytes)
        self._lock = threading.Lock()
        self._settle_lock = threading.Lock()  # held while a request books against the meter; close() takes it, so none is half done after it
        self._closed = False
        self._exhausted: Optional[BudgetExhausted] = None
        self.exhausted_event = threading.Event()
        self._stats = {"requests": 0, "settled": 0, "settled_zero": 0, "unsettled": 0, "rejected_budget": 0, "rejected_auth": 0, "upstream_errors": 0, "dropped_after_close": 0}

    def close(self) -> None:
        """Stop booking against the meter. The trial runner calls it once the servers' drain is over and before it reads the usage and seals the
        ledger: a request still waiting on upstream then keeps its reservation (what the ledger holds) and cannot change the meter afterwards."""
        with self._settle_lock:
            self._closed = True

    # ---- state the trial runner reads -------------------------------------------------------------
    @property
    def exhausted(self) -> Optional[BudgetExhausted]:
        """The first ``BudgetExhausted`` the meter raised for a call (reserve or settle); the runner stops the runtime when it is set."""
        return self._exhausted

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def _bump(self, key: str) -> None:
        with self._lock:
            self._stats[key] += 1

    def _note_exhausted(self, exc: BudgetExhausted) -> None:
        with self._lock:
            if self._exhausted is None:
                self._exhausted = exc
        self.exhausted_event.set()

    # ---- request handling -------------------------------------------------------------------------
    def handle(self, h: BaseHTTPRequestHandler) -> None:
        path = urllib.parse.urlsplit(h.path).path
        if path != "/v1/chat/completions":
            send_json(h, 404, {"error": {"message": "not found", "type": "invalid_request_error"}})
            return
        if h.command != "POST":
            send_json(h, 405, {"error": {"message": "method not allowed", "type": "invalid_request_error"}})
            return
        if not bearer_ok(h, self._token):
            self._bump("rejected_auth")
            send_json(h, 401, {"error": {"message": "invalid token", "type": "invalid_request_error", "code": "invalid_api_key"}})
            return
        raw = read_request_body(h, self._max_request_bytes)
        try:
            body = loads_strict(raw)
        except ContractError:
            send_json(h, 400, {"error": {"message": "request body is not strict JSON", "type": "invalid_request_error"}})
            return
        if not isinstance(body, dict) or not isinstance(body.get("model"), str) or not body["model"] or not isinstance(body.get("messages"), list):
            send_json(h, 400, {"error": {"message": "model and messages are required", "type": "invalid_request_error"}})
            return
        model = body["model"]
        stream = body.get("stream") is True
        if stream:
            options = body.get("stream_options")
            options = dict(options) if isinstance(options, dict) else {}
            options["include_usage"] = True
            body["stream_options"] = options
        n_choices = request_n(body)
        if n_choices is None:
            send_json(h, 400, {"error": {"message": "n must be a positive integer", "type": "invalid_request_error"}})
            return
        # Bound what upstream may generate by what is reserved: tell it the per-completion limit explicitly (an omitted or invalid limit
        # would otherwise be unbounded upstream) and reserve it once per requested completion.
        per_completion = request_max_tokens(body)
        body["max_tokens"] = per_completion
        if "max_completion_tokens" in body:
            body["max_completion_tokens"] = per_completion
        forwarded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        estimate = estimate_prompt_tokens(len(raw))
        call_id = "llm-" + secrets.token_hex(8)
        self._bump("requests")
        try:
            with self._settle_lock:
                if self._closed:
                    self._bump("dropped_after_close")
                    send_json(h, 503, {"error": {"message": "trial is over", "type": "upstream_error"}})
                    return
                self._meter.reserve_llm(model, estimate, per_completion * n_choices, call_id)
        except BudgetExhausted as exc:
            self._bump("rejected_budget")
            self._note_exhausted(exc)
            send_json(
                h,
                429,
                {"error": {"message": f"budget exhausted on axis {exc.axis}", "type": "budget_exhausted", "code": "budget_exhausted", "axis": exc.axis}},
            )
            return
        self._forward(h, forwarded, stream, model, call_id)

    # ---- forwarding -------------------------------------------------------------------------------
    def _connect(self) -> http.client.HTTPConnection:
        if self._scheme == "https":
            return http.client.HTTPSConnection(self._host, self._port, timeout=self._timeout)
        return http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)

    def _settle(self, model: str, usage: LlmUsage, call_id: str, *, zero: bool = False) -> None:
        with self._settle_lock:
            if self._closed:  # the trial's accounting is read and sealed: the reservation stands, as in the ledger
                self._bump("dropped_after_close")
                return
            try:
                self._meter.charge_llm(model, usage, call_id)
            except BudgetExhausted as exc:  # the call is already paid for; the next one is refused
                self._note_exhausted(exc)
        self._bump("settled_zero" if zero else "settled")

    def _scrub(self, data: bytes) -> bytes:
        key = self._api_key.encode("utf-8")
        return data.replace(key, _REDACTED.encode("utf-8")) if key and key in data else data

    def _forward(self, h: BaseHTTPRequestHandler, forwarded: bytes, stream: bool, model: str, call_id: str) -> None:
        conn = self._connect()
        try:
            try:
                conn.connect()
            except (OSError, http.client.HTTPException):
                # nothing was sent, so nothing was generated: settle the reservation at zero
                self._bump("upstream_errors")
                self._settle(model, LlmUsage(0, 0, 0), call_id, zero=True)
                send_json(h, 502, {"error": {"message": "upstream unreachable", "type": "upstream_error"}})
                return
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self._api_key,
                "Accept": "text/event-stream" if stream else "application/json",
                "User-Agent": "nightshift-lab-metering-proxy",
            }
            try:
                conn.request("POST", self._path, body=forwarded, headers=headers)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException):
                # the request may have been processed: the worst-case reservation stays booked
                self._bump("upstream_errors")
                self._bump("unsettled")
                send_json(h, 502, {"error": {"message": "upstream request failed", "type": "upstream_error"}})
                return
            if resp.status >= 300:
                data = self._scrub(resp.read(1 << 20))
                self._bump("upstream_errors")
                self._settle(model, LlmUsage(0, 0, 0), call_id, zero=True)
                status = resp.status if resp.status >= 400 else 502
                extra = {}
                if resp.getheader("Retry-After"):
                    extra["Retry-After"] = resp.getheader("Retry-After")
                send_bytes(h, status, data, resp.getheader("Content-Type") or "application/json", extra)
                return
            if stream:
                self._relay_stream(h, resp, model, call_id)
            else:
                self._relay_json(h, resp, model, call_id)
        finally:
            conn.close()

    def _relay_json(self, h: BaseHTTPRequestHandler, resp: http.client.HTTPResponse, model: str, call_id: str) -> None:
        try:
            data = resp.read()
        except (OSError, http.client.HTTPException):
            self._bump("unsettled")
            send_json(h, 502, {"error": {"message": "upstream response failed", "type": "upstream_error"}})
            return
        usage = None
        try:
            usage = usage_from(json.loads(data.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        if usage is not None:
            self._settle(model, usage, call_id)
        else:
            self._bump("unsettled")  # no usage: the reservation stands
        send_bytes(h, resp.status, self._scrub(data), resp.getheader("Content-Type") or "application/json")

    def _relay_stream(self, h: BaseHTTPRequestHandler, resp: http.client.HTTPResponse, model: str, call_id: str) -> None:
        h.send_response(resp.status)
        h.send_header("Content-Type", resp.getheader("Content-Type") or "text/event-stream")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("Connection", "close")
        h.end_headers()
        usage: Optional[LlmUsage] = None
        pending = b""
        while True:
            try:
                line = resp.readline(_MAX_SSE_LINE)
            except (OSError, http.client.HTTPException):
                break  # upstream died mid-stream: settle on a usage chunk if one already arrived, else keep the reservation
            if not line:
                break
            try:
                h.wfile.write(self._scrub(line))
                h.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                self._bump("unsettled")  # client gone: the reservation stands
                return
            pending += line
            if not pending.endswith(b"\n"):
                continue
            text, pending = pending.strip(), b""
            if text.startswith(b"data:") and b"[DONE]" not in text[:16]:
                try:
                    u = usage_from(json.loads(text[5:].decode("utf-8")))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    u = None
                if u is not None:
                    usage = u
        if usage is not None:
            self._settle(model, usage, call_id)
        else:
            self._bump("unsettled")
