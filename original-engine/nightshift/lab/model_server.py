"""``ModelServer`` on a rented GPU, and the lease machinery the trainer shares.

A *lease* is one rented GPU in use: the rental, its ssh target, the per-minute budget ticker and the open port forwards. Everything that
rents a GPU goes through :func:`acquire_lease`:

* the rental's cap is the smaller of the caller's ``max_gpu_minutes``, the meter's remaining GPU minutes and the minutes the remaining
  dollars buy at the quoted rate, so the armed deadline (and the detached watchdog) can never outlast the trial's budget;
* a :class:`GpuTicker` books one GPU minute per tick (``gpu:<rental_id>:<n>`` for minutes, ``...:usd`` for dollars) from the create
  attempt on; the first :class:`~nightshift.lab.contracts.BudgetExhausted` it sees deletes the rental at once (a budget crossing is a hard
  stop, not a request);
* if anything fails while the machine is being prepared, the rental is deleted before the error leaves.

:func:`serve_on_lease` starts vLLM (``hostkit/serve.sh``) on the lease and returns an endpoint whose ``base_url`` is a control-host
loopback port forwarded over ssh: the control host never talks to a public port of a rental. The base URL has NO ``/v1`` suffix (the
generator appends ``/v1/chat/completions``). Adapters are identified by their sha256 listing hash; ``serve.sh`` refuses to start if an
adapter directory on the rental does not hash to the value the control host verified.

NOTHING here has run against a real GPU; the stack (vLLM 0.32.0 with ``--enable-lora`` on Glimmer) is UNVERIFIED end to end.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from nightshift.lab import gpu_remote as remote
from nightshift.lab.contracts import (
    AdapterRef,
    BaseModelRef,
    BudgetExhausted,
    BudgetMeter,
    Clock,
    GpuError,
    GpuUsage,
    LabLayout,
    ModelServer,
    Resource,
    ServeEndpoint,
    ServeSession,
    ServeSpec,
    Shape,
    TrainerError,
    verify_adapter_dir,
)
from nightshift.lab.gpu_rentals import Rental, RentalManager, RentalRefused

BASE_NAME = "muse-glimmer"
DEFAULT_VLLM_PORT = 8000
STAGE_SH_TIMEOUT = 600.0


DOWNLOAD_TOKEN_ENV = frozenset({"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"})


@dataclass(frozen=True)
class RemoteLayout:
    """Directories on the rental (tests point them into a temp directory because the "rental" is then the local machine) and the extra
    environment passed to every hostkit script."""

    opt: str = "/opt/lab"
    work: str = "/work"
    models: str = "/models"
    env: tuple = ()  # ((NAME, value), ...) extra environment for hostkit scripts

    @property
    def hostkit(self) -> str:
        return f"{self.opt}/hostkit"

    @property
    def stage(self) -> str:
        return f"{self.opt}/stage"

    def script_env(self, *, download_token: bool = False) -> dict:
        """The environment for a hostkit script. A Hugging Face download token in ``env`` is for ``bootstrap.sh`` alone (the weights download):
        every other script, and so the researcher's recipe that ``run_job.sh`` starts, gets the environment without it."""
        base = {"LAB_OPT": self.opt, "LAB_WORK": self.work, "LAB_MODELS": self.models, "HOSTKIT": self.hostkit}
        base.update({k: v for k, v in dict(self.env).items() if download_token or k not in DOWNLOAD_TOKEN_ENV})
        return base


def adapter_model_name(adapter: AdapterRef) -> str:
    """The sha-derived name an adapter is served under (valid vLLM LoRA name; stable for one listing hash)."""
    return f"lora-{adapter.sha256[:16]}"


class GpuTicker:
    """Books one GPU minute per tick against the trial's meter, from the create attempt on (conservative: billing starts at the attempt).

    ``poll_once`` books every tick that is due (``ceil(elapsed / 60)``, at least one); a daemon thread calls it every ``interval`` real
    seconds. The first BudgetExhausted (or any failure to record a charge: fail closed) is remembered in ``exhausted`` and ``on_exhausted``
    runs once. ``finish`` books the ticks up to the moment the rental was deleted.
    """

    def __init__(
        self,
        meter: Optional[BudgetMeter],
        rental_id: str,
        usd_per_hour: float,
        clock: Clock,
        start_epoch: float,
        on_exhausted: Callable[[], None],
        interval: float = 5.0,
    ) -> None:
        self.meter, self.rental_id, self.usd_per_hour = meter, rental_id, usd_per_hour
        self.clock, self.start_epoch, self.on_exhausted, self.interval = clock, start_epoch, on_exhausted, interval
        self.booked = 0
        self.exhausted: Optional[BaseException] = None
        self._fired = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _book(self, n: int) -> None:
        detail = {"rental": self.rental_id, "tick": n}
        problem: Optional[BaseException] = None
        for resource, amount, suffix in ((Resource.GPU_MINUTES, 1.0, ""), (Resource.GPU_USD, self.usd_per_hour / 60.0, ":usd")):
            try:
                assert self.meter is not None
                self.meter.charge(resource, amount, f"gpu:{self.rental_id}:{n}{suffix}", detail)
            except BudgetExhausted as exc:  # book the other half too, then stop
                problem = problem or exc
            except Exception as exc:  # noqa: BLE001 - an unrecorded charge is treated as exhausted (fail closed)
                problem = problem or exc
        if problem is not None and self.exhausted is None:
            self.exhausted = problem

    def poll_once(self, upto_epoch: Optional[float] = None) -> None:
        if self.meter is None:
            return
        fire = False
        with self._lock:
            now = self.clock() if upto_epoch is None else upto_epoch
            due = max(1, math.ceil((now - self.start_epoch) / 60.0 - 1e-9))
            while self.booked < due and self.exhausted is None:
                self.booked += 1
                self._book(self.booked)
            if self.exhausted is not None and not self._fired:
                self._fired = fire = True
        if fire:
            self.on_exhausted()

    def rearm(self) -> None:
        """Let ``on_exhausted`` run again at the next poll: the last attempt to stop the rental failed, so it still bills."""
        with self._lock:
            self._fired = False

    def start(self) -> None:
        self.poll_once()
        if self.meter is None:
            return

        def loop() -> None:
            while not self._stop.wait(self.interval):
                try:
                    self.poll_once()
                except Exception:  # noqa: BLE001 - the loop must not die silently while a rental bills
                    self.exhausted = self.exhausted or TrainerError("ticker failed")
                    return

        self._thread = threading.Thread(target=loop, name=f"gpu-ticker-{self.rental_id}", daemon=True)
        self._thread.start()

    def finish(self, end_epoch: float) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=10)
        if self.meter is not None and self.exhausted is None:
            with self._lock:
                due = max(1, math.ceil((end_epoch - self.start_epoch) / 60.0 - 1e-9))
                while self.booked < due:
                    self.booked += 1
                    self._book(self.booked)


@dataclass
class Lease:
    """One rented GPU in use. ``close`` stops the forwards, books the last ticks, deletes the rental and returns the billed usage."""

    manager: RentalManager
    rental: Rental
    target: remote.SshTarget
    root: Path
    layout_remote: RemoteLayout
    ticker: GpuTicker
    clock: Clock
    sleep: Callable[[float], None] = time.sleep
    poll_seconds: float = 5.0
    vllm_port: int = DEFAULT_VLLM_PORT
    forwards: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    _closed: Optional[GpuUsage] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    base: Optional[BaseModelRef] = None  # the checkpoint this rental was bootstrapped with

    @property
    def exhausted(self) -> Optional[BaseException]:
        return self.ticker.exhausted

    @property
    def closed(self) -> bool:
        return self._closed is not None

    def check_alive(self) -> None:
        """Raise BudgetExhausted if the ticker found the budget used up (the rental is being or has been deleted)."""
        exc = self.ticker.exhausted
        if exc is not None:
            raise exc if isinstance(exc, BudgetExhausted) else BudgetExhausted("gpu_minutes", 0.0, 0.0)

    def close(self, reason: str) -> GpuUsage:
        with self._lock:
            if self._closed is not None:
                return self._closed
            for f in self.forwards:
                try:
                    f.close()
                except Exception:  # noqa: BLE001
                    pass
            self.forwards.clear()
            last: Optional[Exception] = None
            for attempt in range(3):
                try:
                    self.manager.delete(self.rental, reason)
                    last = None
                    break
                except GpuError as exc:  # includes ProviderUnavailable; OwnershipError is final
                    last = exc
                    from nightshift.lab.contracts import OwnershipError

                    if isinstance(exc, OwnershipError):
                        break
                    self.sleep(self.poll_seconds)
            if last is None:
                row_end = self.clock()
                try:
                    self.ticker.finish(row_end)
                except Exception:  # noqa: BLE001
                    pass
            else:
                # The rental still bills: keep the ticker running (it books the minutes and, once the budget is crossed, retries this close).
                self.ticker.rearm()
                raise TrainerError(f"could not delete rental {self.rental.session}: {last}. It may still bill; the watchdog will retry at its cap.") from last
            self._closed = self.manager.usage_for_session(self.rental.session)
            return self._closed


def run_wrapped(lease: Lease, script_argv: Sequence[str], name: str, *, timeout_s: float, done_file: Optional[str] = None, download_token: bool = False) -> int:
    """Run ``script_argv`` detached on the rental and wait for it. A wrapper records the exit status in ``<work>/<name>.rc`` so a script that
    dies without writing its own marker is still noticed. Returns the status. Raises BudgetExhausted if the ticker stops the lease,
    GpuError on timeout."""
    rl = lease.layout_remote
    rc_file = f"{rl.work}/{name}.rc"
    remote.run(lease.target, f"mkdir -p {rl.work} && rm -f {rc_file} {rc_file}.tmp", timeout=60)
    wrapper = '"$@"; echo $? > "$RC.tmp"; mv -f "$RC.tmp" "$RC"'
    env = dict(rl.script_env(download_token=download_token), RC=rc_file)
    remote.run_detached(lease.target, ["bash", "-c", wrapper, "_", *script_argv], f"{rl.work}/{name}.runner.log", env=env)
    start = lease.clock()
    while True:
        lease.check_alive()
        if remote.remote_exists(lease.target, rc_file):
            res = remote.run(lease.target, f"cat {rc_file}", timeout=60)
            try:
                return int(res.text.strip())
            except ValueError:
                raise GpuError(f"{name}: unreadable exit status {res.text[:40]!r}") from None
        if lease.clock() - start > timeout_s:
            raise GpuError(f"{name} did not finish within {timeout_s:g} s")
        lease.sleep(lease.poll_seconds)


def create_when_settled(manager: RentalManager, session: str, shape: Shape, cap_hours: float, trial_id: str, item: str, *, wait_s: float, clock: Clock, sleep: Callable[[float], None], poll_seconds: float) -> Rental:
    """``manager.create``, waiting while ANOTHER create is in flight (concurrent trials share one rental ledger and one create is allowed at a
    time). A stuck ``uuid_unknown`` rental never settles by itself: after ``wait_s`` the refusal is raised."""
    start = clock()
    while True:
        try:
            return manager.create(session, shape, cap_hours, trial_id, item)
        except RentalRefused as exc:
            if exc.code != "unsettled" or clock() - start > wait_s:
                raise
            sleep(poll_seconds)


def _gpu_cap_minutes(manager: RentalManager, shape: Shape, max_minutes: float, meter: Optional[BudgetMeter]) -> float:
    if meter is None:
        return float(max_minutes)
    meter.check(Resource.GPU_MINUTES)
    meter.check(Resource.GPU_USD)
    spec, used = meter.spec, meter.usage()
    by_minutes = spec.gpu_minutes - used.gpu_minutes
    rate = manager.quote(shape).usd_per_hour
    by_dollars = (spec.dollars - used.dollars) / rate * 60.0 if rate > 0 else float("inf")
    # The rental manager bills and arms the lease in whole minutes, rounding the cap UP; round the allowance DOWN here so the armed lease
    # never reaches a minute the trial's remaining allowance does not cover.
    cap = float(math.floor(min(float(max_minutes), by_minutes, by_dollars) + 1e-9))
    if cap < 1.0:
        axis = "dollars" if by_dollars < 1.0 else "gpu_minutes"
        raise BudgetExhausted(axis, spec.dollars if axis == "dollars" else spec.gpu_minutes, used.dollars if axis == "dollars" else used.gpu_minutes)
    return cap


SSH_READY_TIMEOUT_S = 300.0


def _wait_for_ssh(lease: "Lease", clock: Clock, sleep: Callable[[float], None], timeout_s: float = SSH_READY_TIMEOUT_S) -> None:
    """A rental the provider reports RUNNING may not accept ssh yet: LIVE 2026-11-04, fresh scoring rentals answered "Connection refused",
    "kex_exchange_identification: Connection reset by peer" and a port forward that exited 255 before it listened. Wait (bounded) until a
    trivial command succeeds; the last error is raised if it never does."""
    deadline = clock() + timeout_s
    while True:
        lease.check_alive()
        try:
            remote.run(lease.target, "true", timeout=30.0)
            return
        except GpuError:
            if clock() >= deadline:
                raise
            sleep(10.0)


def acquire_lease(
    manager: RentalManager,
    shape: Shape,
    *,
    session: str,
    trial_id: str,
    item: str,
    max_minutes: float,
    meter: Optional[BudgetMeter],
    hostkit_dir: Path,
    root: Path,
    layout_remote: RemoteLayout = RemoteLayout(),
    stage_bundle: Optional[Path] = None,
    clock: Clock,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = 5.0,
    tick_interval: float = 5.0,
    ready_timeout_s: float = 900.0,
    bootstrap_timeout_s: float = 7200.0,
    vllm_port: int = DEFAULT_VLLM_PORT,
    create_wait_s: float = 300.0,
    base: Optional[BaseModelRef] = None,
) -> Lease:
    """Rent a GPU and prepare it (hostkit uploaded, staged files installed, bootstrap run). Raises BudgetExhausted before renting when the
    meter has no GPU room; RentalRefused / CreateRejected / CreateAmbiguous / ProviderUnavailable from the manager; the rental is deleted
    before any later error leaves this function. ``base`` is the checkpoint to download and serve: it reaches every hostkit script as
    ``GLIMMER_REPO`` / ``GLIMMER_REVISION`` (the weights pins still apply), and ``serve_on_lease`` refuses a ``ServeSpec`` for another one."""
    if base is not None:
        layout_remote = replace(layout_remote, env=tuple(layout_remote.env) + (("GLIMMER_REPO", base.repo_id), ("GLIMMER_REVISION", base.revision)))
    cap = _gpu_cap_minutes(manager, shape, max_minutes, meter)
    rental = create_when_settled(manager, session, shape, cap / 60.0, trial_id, item, wait_s=create_wait_s, clock=clock, sleep=sleep, poll_seconds=poll_seconds)
    rate = rental.usd_per_hour
    holder: dict = {}

    def kill() -> None:
        lease = holder.get("lease")
        if lease is not None:
            try:
                lease.close("budget exhausted: the trial's GPU allowance is used up")
            except Exception:  # noqa: BLE001 - the watchdog is the backstop
                pass

    ticker = GpuTicker(meter, rental.session, rate, clock, rental.create_attempt_epoch, kill, tick_interval)
    lease = Lease(manager, rental, None, Path(root), layout_remote, ticker, clock, sleep, poll_seconds, vllm_port, base=base)  # type: ignore[arg-type]
    holder["lease"] = lease
    try:
        ticker.start()
        lease.check_alive()
        manager.wait_ready(rental, ready_timeout_s)
        lease.check_alive()
        lease.target = manager.endpoint(rental)
        _wait_for_ssh(lease, clock, sleep)
        t0 = clock()
        remote.upload_tree(lease.target, hostkit_dir, layout_remote.hostkit)
        if stage_bundle is not None:
            remote.upload_tree(lease.target, stage_bundle, layout_remote.stage)
            rc = run_wrapped(lease, ["bash", f"{layout_remote.hostkit}/stage.sh", layout_remote.stage], "stage", timeout_s=STAGE_SH_TIMEOUT)
            if rc != 0:
                raise TrainerError(f"staging the public files on the rental failed (exit {rc}): {remote.read_remote_tail(lease.target, layout_remote.work + '/stage.runner.log', max_bytes=1500)}")
        rc = run_wrapped(lease, ["bash", f"{layout_remote.hostkit}/bootstrap.sh"], "bootstrap", timeout_s=bootstrap_timeout_s, download_token=True)
        if rc != 0:
            raise TrainerError(f"bootstrap of the rental failed (exit {rc}): {remote.read_remote_tail(lease.target, layout_remote.work + '/bootstrap.runner.log', max_bytes=1500)}")
        lease.timings["setup_s"] = clock() - t0
        try:
            done = remote.run(lease.target, f"cat {layout_remote.opt}/bootstrap.done", timeout=60).text
            info = json.loads(done)
            if isinstance(info.get("download_s"), (int, float)):
                lease.timings["download_s"] = float(info["download_s"])
        except (GpuError, ValueError):
            pass
        return lease
    except BaseException:
        try:
            lease.close("setup failed")
        except Exception:  # noqa: BLE001 - the original error is the one to report
            pass
        raise


def _http_get(url: str, timeout: float = 5.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read(1 << 20)
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except (urllib.error.URLError, OSError):
        return 0, b""


def serve_on_lease(
    lease: Lease,
    spec: ServeSpec,
    *,
    remote_adapter_dirs: Optional[Mapping[str, str]] = None,
    serve_timeout_s: float = 1800.0,
) -> ServeEndpoint:  # fmt: skip
    """Start vLLM for ``spec.base`` and ``spec.adapters`` on the lease and return the endpoint (a forwarded control-host loopback port).

    An adapter already on the rental (the one a training job just produced) is named in ``remote_adapter_dirs`` by its listing sha256; any
    other adapter is verified on the control host (``verify_adapter_dir``) and uploaded. Either way ``serve.sh`` re-hashes the directory on
    the rental and refuses a mismatch."""
    if lease.base != spec.base:
        have = f"{lease.base.repo_id}@{lease.base.revision}" if lease.base is not None else "an unrecorded checkpoint"
        raise TrainerError(f"this rental was prepared with {have}, not the requested {spec.base.repo_id}@{spec.base.revision}")
    rl = lease.layout_remote
    names: dict = {}
    args: list = []
    for ad in spec.adapters:
        verify_adapter_dir(ad, root=lease.root)  # the control-host copy is what we trust
        name = adapter_model_name(ad)
        rdir = (remote_adapter_dirs or {}).get(ad.sha256)
        if rdir is None:
            rdir = f"{rl.models}/adapters/{ad.adapter_id}"
            for f in ad.files:
                src = ad.resolve(lease.root) / f.name
                # LIVE 2026-11-04: a 767 MB adapter did not reach a fresh scoring rental within the 600 s default; allow 600 s plus 2 s per MB
                remote.upload(lease.target, src, f"{rdir}/{f.name}", timeout=600.0 + 2.0 * src.stat().st_size / 1e6)
        names[ad.sha256] = name
        args += ["--adapter", f"{name}={rdir}={ad.sha256}"]
    lease.check_alive()
    t0 = lease.clock()
    remote.run_detached(
        lease.target,
        ["bash", f"{rl.hostkit}/serve.sh", "--port", str(lease.vllm_port), "--served-name", BASE_NAME, "--max-model-len", str(spec.max_model_len), *args],
        f"{rl.work}/vllm.log",
        env=rl.script_env(),
    )  # fmt: skip
    local_port = remote.free_loopback_port()
    fwd = remote.forward_port(lease.target, local_port, lease.vllm_port)
    lease.forwards.append(fwd)
    fwd.wait_ready(30)
    base_url = f"http://127.0.0.1:{local_port}"
    expected = {BASE_NAME, *names.values()}
    while True:
        lease.check_alive()
        status, body = _http_get(base_url + "/v1/models")
        if status == 200:
            try:
                listed = {m.get("id") for m in json.loads(body).get("data", [])}
            except (ValueError, AttributeError):
                listed = set()
            if expected <= listed:
                break
        if not fwd.alive():
            raise TrainerError("the port forward to the model server died")
        if lease.clock() - t0 > serve_timeout_s:
            tail = remote.read_remote_tail(lease.target, f"{rl.work}/vllm.log", max_bytes=1500)
            raise TrainerError(f"the model server did not list {sorted(expected)} within {serve_timeout_s:g} s. Log tail: {tail}")
        lease.sleep(lease.poll_seconds)
    lease.timings["load_s"] = lease.clock() - t0
    return ServeEndpoint(base_url, BASE_NAME, names)


class _Session:
    def __init__(self, lease: Lease, endpoint: ServeEndpoint) -> None:
        self._lease, self._endpoint = lease, endpoint

    @property
    def endpoint(self) -> ServeEndpoint:
        return self._endpoint

    @property
    def lease(self) -> Lease:
        return self._lease

    def close(self) -> GpuUsage:
        return self._lease.close("serving session closed")

    def __enter__(self) -> "ServeSession":
        return self  # type: ignore[return-value]

    def __exit__(self, *exc: Any) -> None:
        self.close()


class RentalModelServer:
    """``ModelServer`` that rents a FRESH GPU per session (never one a researcher recipe ran on), serves the base and the adapters on it,
    and always deletes it. ``layout`` gives the lab root adapters are read from."""

    def __init__(
        self,
        manager: RentalManager,
        shape: Shape,
        hostkit_dir: Path,
        layout: Optional[LabLayout] = None,
        *,
        clock: Optional[Clock] = None,
        remote_layout: RemoteLayout = RemoteLayout(),
        stage_bundle: Optional[Path] = None,
        sleep: Callable[[float], None] = time.sleep,
        poll_seconds: float = 5.0,
        tick_interval: float = 5.0,
        vllm_port: int = DEFAULT_VLLM_PORT,
        ready_timeout_s: float = 900.0,
        bootstrap_timeout_s: float = 7200.0,
        serve_timeout_s: float = 1800.0,
    ) -> None:
        self.manager, self.shape, self.hostkit_dir = manager, shape, Path(hostkit_dir)
        self.layout = layout
        self.clock = clock or manager.clock
        self.remote_layout, self.stage_bundle = remote_layout, stage_bundle
        self.sleep, self.poll_seconds, self.tick_interval, self.vllm_port = sleep, poll_seconds, tick_interval, vllm_port
        self.ready_timeout_s, self.bootstrap_timeout_s, self.serve_timeout_s = ready_timeout_s, bootstrap_timeout_s, serve_timeout_s

    def serve(self, spec: ServeSpec, meter: Optional[BudgetMeter] = None) -> ServeSession:
        if self.layout is None:
            raise TrainerError("RentalModelServer needs the LabLayout to read adapters from")
        trial = spec.trial_id or "scorer"
        session = f"{re.sub(r'[^A-Za-z0-9_.-]', '-', trial)[:30]}-srv-{uuid.uuid4().hex[:8]}"
        lease = acquire_lease(
            self.manager, self.shape, session=session, trial_id=trial, item="serve", max_minutes=spec.max_gpu_minutes, meter=meter,
            hostkit_dir=self.hostkit_dir, root=self.layout.root, layout_remote=self.remote_layout, stage_bundle=self.stage_bundle,
            clock=self.clock, sleep=self.sleep, poll_seconds=self.poll_seconds, tick_interval=self.tick_interval,
            ready_timeout_s=self.ready_timeout_s, bootstrap_timeout_s=self.bootstrap_timeout_s, vllm_port=self.vllm_port, base=spec.base,
        )  # fmt: skip
        try:
            endpoint = serve_on_lease(lease, spec, serve_timeout_s=self.serve_timeout_s)
        except BaseException:
            try:
                lease.close("serve failed")
            except Exception:  # noqa: BLE001
                pass
            raise
        return _Session(lease, endpoint)  # type: ignore[return-value]
