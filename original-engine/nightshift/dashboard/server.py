"""Read-only dashboard server (docs/ARCHITECTURE.md sections 6.9 and 10).

Guarantees
----------
* Routes are an exact-match table: ``/ /app.js /app.css /api/state /api/events /api/health``. Nothing else is served, no path
  is ever joined to a file name, so there is no traversal surface. Static files come from a fixed directory.
* Every response, errors included, carries the CSP, ``nosniff``, ``no-referrer`` and ``no-store`` headers. No CORS headers.
* ``Host`` must be ``127.0.0.1:<port>``, ``localhost:<port>`` or ``[::1]:<port>`` and a present ``Origin`` must match it (403).
  Only GET and HEAD are allowed (405). Loopback only unless ``allow_remote`` (a warning is printed and the Host check then
  only requires a well-formed host on the bound port).
* The ledger is opened read-only and followed with ``refresh``; this server never writes to the run directory.
* Replay serves ever longer prefixes of an existing ledger. The chain is verified first and a broken one is refused.
  The replay clock starts at the first request that shows the run (``/api/state`` or ``/api/events``), not when the server
  object is built, so a 10x replay is not mostly over by the time a browser opens. ``/api/health`` never starts it.
* ``/api/health`` tells the truth about the run: ``mode`` is the state's ``run.mode`` vocabulary (``scripted``, ``simulated``,
  ``live``, ``replay``, ``unknown``), ``origin`` is what produced the words ignoring replay, and ``serving`` is only
  ``live`` or ``replay`` (how this server is serving the ledger). A scripted or simulated run is never reported as live.
"""
from __future__ import annotations

import bisect
import ipaddress
import json
import math
import re
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from urllib.parse import parse_qs, urlsplit

from .. import secrets
from ..ledger import Ledger, verify_events
from ..types import Event, Invariant, LedgerCorrupt
from ..view import build_view
from .state_api import build_state, describe_event, run_origin

CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
SECURITY_HEADERS = {"Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
                    "Cache-Control": "no-store"}
STATIC_FILES = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/app.css": ("app.css", "text/css; charset=utf-8")}
API_ROUTES = ("/api/state", "/api/events", "/api/health")
ROUTES = tuple(STATIC_FILES) + API_ROUTES
EVENTS_CAP = 500
MAX_STATIC_BYTES = 4 * 1024 * 1024
REQUEST_TIMEOUT_SEC = 10
REPLAY_MIN_SPEED, REPLAY_MAX_SPEED = 0.1, 100.0
_INT = re.compile(r"[0-9]{1,15}")
_HOST_RE = re.compile(r"^(?:[A-Za-z0-9.-]{1,253}|\[[0-9A-Fa-f:.]{2,45}\])(?::[0-9]{1,5})?$")
PACKAGE_DIR = Path(__file__).resolve().parent


class ReplayRefused(RuntimeError):
    """Replay was requested for a ledger whose chain does not verify (or that cannot be read)."""


def parse_since(raw: Optional[str]) -> int:
    """A non-negative int from a query value; anything that is not 1 to 15 plain digits is 0."""
    if raw is None or not _INT.fullmatch(raw):
        return 0
    return int(raw)


def is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _validate_replay(speed: Optional[float], duration: Optional[float]) -> None:
    if speed is not None and duration is not None:
        raise ValueError("replay_speed and replay_duration are mutually exclusive")
    if speed is not None:
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) \
                or not REPLAY_MIN_SPEED <= speed <= REPLAY_MAX_SPEED:
            raise ValueError(f"replay_speed must be a number between {REPLAY_MIN_SPEED} and {REPLAY_MAX_SPEED}")
    if duration is not None:
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("replay_duration must be a positive number of seconds")


class DashboardServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` that owns the ledger follower and the replay clock."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple, handler: type, *, run_dir: Path, task: Any, limits: Any, allow_remote: bool,
                 assets_dir: Path, replay_speed: Optional[float], replay_duration: Optional[float],
                 clock: Callable[[], float], registry: Optional[Iterable[Invariant]]):
        self.run_dir, self.task, self.limits, self.registry = run_dir, task, limits, registry
        self.allow_remote, self.assets_dir, self.clock = allow_remote, assets_dir, clock
        self._lock = threading.Lock()
        self.replay = replay_speed is not None or replay_duration is not None
        self.replay_speed: Optional[float] = None
        self._offsets: list[float] = []
        self._replay_events: list[Event] = []
        self._replay_start: Optional[float] = None
        try:
            self.ledger = Ledger.read_only(run_dir)
        except LedgerCorrupt as exc:
            if self.replay:
                raise ReplayRefused(f"the ledger is corrupt ({exc}); refusing to replay it") from exc
            raise
        if self.replay:
            self._init_replay(replay_speed, replay_duration)
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, handler)
        port = self.server_address[1]
        self.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

    # ---- replay ----------------------------------------------------------------------------------
    def _init_replay(self, speed: Optional[float], duration: Optional[float]) -> None:
        events = self.ledger.events()
        if not events:
            raise ReplayRefused("the ledger is empty; nothing to replay")
        ok, bad = verify_events(events)
        if not ok:
            raise ReplayRefused(f"the ledger chain is broken at seq {bad}; refusing to replay it")
        first, running = events[0].ts, 0.0
        for ev in events:
            running = max(running, ev.ts - first)   # running max keeps the prefix monotone even if clocks stepped back
            self._offsets.append(running)
        span = running
        self.replay_speed = float(speed) if speed is not None else (span / duration if span > 0 else REPLAY_MAX_SPEED)
        self._replay_events = events

    @property
    def replay_started(self) -> bool:
        """True once the replay clock is running (a replay server that nobody has asked yet is still at event 1)."""
        return self._replay_start is not None

    def _replay_prefix(self, start: bool = True) -> list[Event]:
        """The events replayed so far. ``start=True`` starts the clock if this is the first look at the run; ``start=False``
        (health probes) never does, and then shows only the first event."""
        with self._lock:
            if self._replay_start is None:
                if not start:
                    return self._replay_events[:1]
                self._replay_start = self.clock()
            began = self._replay_start
        virtual = max(0.0, (self.clock() - began) * (self.replay_speed or 1.0))
        n = bisect.bisect_right(self._offsets, virtual + 1e-9)
        return self._replay_events[: max(1, n)]

    # ---- snapshots -------------------------------------------------------------------------------
    def snapshot(self, *, start_replay: bool = True) -> tuple[list[Event], dict]:
        """(events, replay info or {}) for one consistent response."""
        if self.replay:
            prefix = self._replay_prefix(start_replay)
            return prefix, {"speed": self.replay_speed, "position_seq": prefix[-1].seq,
                            "total_seq": self._replay_events[-1].seq, "source_run_id": self._replay_events[0].run_id}
        with self._lock:
            self.ledger.refresh()
            return self.ledger.events(), {}

    def state(self) -> dict:
        events, replay = self.snapshot()
        view = build_view(events)
        now = events[-1].ts if self.replay and events else None
        return build_state(view, self.task, limits=self.limits, now=now, events=events, replay=replay or None,
                           registry=self.registry)

    @property
    def base_url(self) -> str:
        host = self.server_address[0]
        host = f"[{host}]" if ":" in host else host
        return f"http://{host}:{self.server_address[1]}"


class Handler(BaseHTTPRequestHandler):
    server: DashboardServer
    timeout = REQUEST_TIMEOUT_SEC
    protocol_version = "HTTP/1.0"
    server_version = "nightshift"
    sys_version = ""

    # ---- plumbing --------------------------------------------------------------------------------
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - signature fixed by the base class
        return

    def version_string(self) -> str:
        return self.server_version

    def _send(self, status: int, body: bytes, ctype: str, extra: Optional[dict] = None) -> None:
        if self.request_version == "HTTP/0.9":
            # An unparseable request line leaves the version at HTTP/0.9, for which Python 3.11/3.12 write no status line
            # and no headers at all. Every response must carry the security headers, so answer as HTTP/1.0.
            self.request_version = "HTTP/1.0"
        try:
            self.send_response(status)
            for k, v in SECURITY_HEADERS.items():
                self.send_header(k, v)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            self.close_connection = True

    def _json(self, status: int, obj: Any, extra: Optional[dict] = None) -> None:
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8", extra)

    def send_error(self, code: int, message: Optional[str] = None, explain: Optional[str] = None) -> None:
        phrase = self.responses.get(code, ("error",))[0]
        self._json(code, {"error": phrase})

    def __getattr__(self, name: str) -> Any:
        # every method other than GET/HEAD (including unknown ones) gets a 405, not the base class's 501
        if name.startswith("do_"):
            return self._method_not_allowed
        raise AttributeError(name)

    def _method_not_allowed(self) -> None:
        self._json(405, {"error": "method not allowed"}, {"Allow": "GET, HEAD"})

    # ---- request checks --------------------------------------------------------------------------
    def _host_ok(self) -> bool:
        hosts = self.headers.get_all("Host") or []
        if len(hosts) != 1:
            return False
        host = hosts[0].strip().lower()
        if self.server.allow_remote:
            return bool(_HOST_RE.match(host)) and (host.endswith(f":{self.server.server_address[1]}") or ":" not in host)
        return host in self.server.allowed_hosts

    def _origin_ok(self, host: str) -> bool:
        origins = self.headers.get_all("Origin") or []
        if not origins:
            return True
        return len(origins) == 1 and origins[0].strip().lower() == f"http://{host.strip().lower()}"

    def _guard(self) -> bool:
        if not self._host_ok() or not self._origin_ok(self.headers.get("Host", "")):
            self._json(403, {"error": "forbidden"})
            return False
        return True

    # ---- GET/HEAD --------------------------------------------------------------------------------
    def do_GET(self) -> None:
        self._serve()

    def do_HEAD(self) -> None:
        self._serve()

    def _serve(self) -> None:
        if not self._guard():
            return
        if not self.path.startswith("/") or self.path.startswith("//"):
            self._json(404, {"error": "not found"})
            return
        try:
            parts = urlsplit(self.path)
        except ValueError:
            self._json(400, {"error": "bad request"})
            return
        route = parts.path
        if route not in ROUTES:
            self._json(404, {"error": "not found"})
            return
        try:
            if route in STATIC_FILES:
                self._static(route)
            elif route == "/api/state":
                self._json(200, self.server.state())
            elif route == "/api/events":
                self._events(parts.query)
            else:
                self._health()
        except Exception:  # noqa: BLE001 - never leak a traceback to the client
            self._json(500, {"error": "internal error"})

    def _static(self, route: str) -> None:
        name, ctype = STATIC_FILES[route]
        path = self.server.assets_dir / name
        try:
            if not path.is_file() or path.stat().st_size > MAX_STATIC_BYTES:
                raise FileNotFoundError(name)
            body = path.read_bytes()
        except OSError:
            self._json(404, {"error": f"{name} is not available"})
            return
        self._send(200, body, ctype)

    def _events(self, query: str) -> None:
        try:
            q = parse_qs(query, keep_blank_values=True, max_num_fields=10)
        except ValueError:
            q = {}
        since = parse_since((q.get("since") or [None])[0])
        limit = parse_since((q.get("limit") or [None])[0]) or EVENTS_CAP
        events, _ = self.server.snapshot()
        total_last = events[-1].seq if events else 0
        rows = [ev for ev in events if ev.seq > since][: min(limit, EVENTS_CAP)]
        simulated = run_origin(build_view(events[:1]).run if events else {}) == "simulated"     # the same labels as the state timeline
        out = [{"seq": ev.seq, "ts": ev.ts, "kind": ev.kind, "actor": ev.actor, "text": describe_event(ev, simulated)} for ev in rows]
        self._json(200, secrets.scrub({"since": since, "last_seq": total_last, "count": len(out),
                                       "has_more": bool(rows) and rows[-1].seq < total_last, "events": out}))

    def _health(self) -> None:
        events, _ = self.server.snapshot(start_replay=False)
        run = build_view(events[:1]).run if events else {}      # run.started is the first event; nothing else is needed
        origin = run_origin(run)
        self._json(200, {"ok": True, "mode": "replay" if self.server.replay else origin, "origin": origin,
                         "serving": "replay" if self.server.replay else "live",
                         "run_id": events[0].run_id if events else "", "events": len(events),
                         "last_seq": events[-1].seq if events else 0})


def make_server(run_dir: Any, host: str = "127.0.0.1", port: int = 8765, *, replay_speed: Optional[float] = None,
                replay_duration: Optional[float] = None, allow_remote: bool = False, task: Any = None,
                limits: Any = None, assets_dir: Any = None, clock: Callable[[], float] = time.monotonic,
                registry: Optional[Iterable[Invariant]] = None) -> DashboardServer:
    """Create (not start) the dashboard server for ``run_dir``.

    Raises ``ValueError`` for a non-loopback ``host`` without ``allow_remote`` or a bad replay setting, and
    ``ReplayRefused`` when replaying a ledger whose chain is broken. ``assets_dir`` defaults to this package directory
    (``index.html``, ``app.js``, ``app.css``); tests pass a temp dir. ``port=0`` picks a free port.
    """
    _validate_replay(replay_speed, replay_duration)
    if not is_loopback_host(host):
        if not allow_remote:
            raise ValueError(f"refusing to bind {host!r}: the dashboard is loopback-only unless allow_remote is set")
        print(f"WARNING: dashboard bound to {host}:{port}, reachable by other machines. It has no authentication and "
              "shows the whole run, sealed-split numbers included.", file=sys.stderr)
    return DashboardServer((host.strip("[]"), int(port)), Handler, run_dir=Path(run_dir), task=task, limits=limits,
                           allow_remote=allow_remote, assets_dir=Path(assets_dir) if assets_dir else PACKAGE_DIR,
                           replay_speed=replay_speed, replay_duration=replay_duration, clock=clock, registry=registry)
