"""Per-trial hard-stop budget meter (``TrialBudgetMeter``) and the lab-wide dollar cap (``LabCap``).

Why this exists and what it guarantees (docs/harness-design.md, "Budget enforcement"):

* Four axes per trial (``BudgetSpec``): researcher tokens, GPU minutes, dollars (researcher inference PLUS GPU, one cap) and wall seconds.
  ``AXIS_OF`` (contracts) maps a ``Resource`` to its axis. Wall time is derived from the injected clock since trial start and is only
  ever ``check``ed; nobody charges it.
* LLM calls are reserve-then-settle. ``reserve_llm`` books the worst case (estimated prompt tokens plus ``max_tokens``, priced with the
  reused ``usd_for``) BEFORE the request is forwarded and raises ``BudgetExhausted`` if booking it would cross a cap. ``charge_llm``
  replaces the reservation with the real usage as TWO events (``<call_id>:tokens`` and ``<call_id>:usd``); both are booked before
  ``BudgetExhausted`` is raised, so a cap crossed by the first half never leaves the second unbooked. A reservation that is never
  settled stays booked at its worst case, and ``usage()`` includes it, so a missing usage chunk can only over-count.
* Charges are idempotent per ``(call_id, resource)`` (a GPU tick may book minutes and dollars under one call id). The state is rebuilt from
  the trial ledger on reopen. Everything is under one lock.
* Fail closed: if a charge cannot be written to the ledger, the axis is treated as exhausted and ``BudgetExhausted`` is raised; the
  amount stays booked in memory.
* ``LabCap`` is one shared, thread-safe object for the whole coordinator: the Meta credit that caps researcher inference plus Wonder
  across every trial. ``reserve_llm`` reserves the worst-case dollars there too and raises ``BudgetExhausted("lab_dollars", ...)``.

Overshoot bound (stated, not zero): an LLM call whose prompt estimate was right cannot cross a cap; the residual error is the
prompt-estimate error times the number of concurrent callers. A GPU charge can cross a cap by one tick.

Import this module by its absolute name (``nightshift.lab.budget``); it shares a basename with ``nightshift/budget.py``.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from nightshift.budget import usd_for
from nightshift.lab.contracts import (
    AXIS_OF,
    BudgetCharge,
    BudgetExhausted,
    BudgetExhaustedEvent,
    BudgetSpec,
    BudgetUsage,
    Clock,
    ContractError,
    LabKind,
    LabLayout,
    LabState,
    LedgerWriter,
    LlmUsage,
    Resource,
    TrialSpec,
)
from nightshift.types import Usage

__all__ = ["LabCap", "TrialBudgetMeter", "usage_from_events", "lab_spent_usd", "outer_usd_spent", "LAB_AXIS"]

LAB_AXIS = "lab_dollars"  # BudgetExhaustedEvent.axis for the lab-wide cap
_ACTOR_PREFIX = "trial:"


def _finite_nonneg(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise ContractError(f"{what}: expected a finite number >= 0, got {v!r}")
    return float(v)


class LabCap:
    """The lab-wide dollar cap on researcher inference plus Wonder (``LabConfig.lab_budget_usd``). One thread-safe object shared by every
    trial meter; seed ``spent_usd`` from the ledgers at start (``lab_spent_usd``).

    ``reserve`` books a worst case under a key and refuses when ``spent + every outstanding reservation + usd`` would exceed the cap;
    ``settle`` replaces that reservation with the actual spend; ``add_spent`` books spend that was never reserved (a direct charge, Wonder).
    ``clock`` is accepted for symmetry with the other components and kept for callers; the cap itself has no time dependence.
    """

    def __init__(self, cap_usd: float, spent_usd: float = 0.0, *, clock: Clock = time.time) -> None:
        self._cap = _finite_nonneg(cap_usd, "LabCap.cap_usd")
        self._spent = _finite_nonneg(spent_usd, "LabCap.spent_usd")
        self.clock = clock
        self._reserved: dict[str, float] = {}
        self._lock = threading.RLock()

    @property
    def cap_usd(self) -> float:
        return self._cap

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent

    @property
    def reserved_usd(self) -> float:
        with self._lock:
            return math.fsum(self._reserved.values())

    def remaining(self) -> float:
        """Cap minus spend minus every outstanding reservation (may be negative after an overshoot)."""
        with self._lock:
            return self._cap - self._spent - math.fsum(self._reserved.values())

    def reserve(self, key: str, usd: float) -> None:
        """Atomically book ``usd`` under ``key`` or raise ``BudgetExhausted("lab_dollars", cap, projected)``. Idempotent per key."""
        usd = _finite_nonneg(usd, "LabCap.reserve usd")
        with self._lock:
            if key in self._reserved:
                return
            held = math.fsum(self._reserved.values())
            projected = self._spent + held + usd
            if projected > self._cap:
                raise BudgetExhausted(LAB_AXIS, self._cap, projected)
            self._reserved[key] = usd

    def settle(self, key: str, actual_usd: float) -> None:
        """Drop the reservation ``key`` (if any) and book the actual spend."""
        actual = _finite_nonneg(actual_usd, "LabCap.settle usd")
        with self._lock:
            self._reserved.pop(key, None)
            self._spent += actual

    def adopt(self, key: str, usd: float) -> None:
        """Turn ``usd`` of the seeded spend into the reservation ``key``. A recovered reservation (a trial that died with a call in flight) was
        seeded into ``spent_usd`` at its worst case by ``lab_spent_usd``; adopting it lets ``settle`` replace it instead of adding the real
        spend on top. Idempotent per key; the spend moved never exceeds what was seeded."""
        usd = _finite_nonneg(usd, "LabCap.adopt usd")
        with self._lock:
            if key in self._reserved:
                return
            self._spent -= min(usd, self._spent)
            self._reserved[key] = usd

    def add_spent(self, usd: float) -> None:
        usd = _finite_nonneg(usd, "LabCap.add_spent usd")
        with self._lock:
            self._spent += usd

    def exhausted(self) -> bool:
        return self.remaining() <= 0.0

    def projected(self) -> float:
        """Spend plus outstanding reservations: the figure compared with the cap."""
        with self._lock:
            return self._spent + math.fsum(self._reserved.values())


class TrialBudgetMeter:
    """Implements ``BudgetMeter`` for one trial, writing ``lab.budget_charged`` / ``lab.budget_exhausted`` to that trial's ledger.

    ``ledger`` is the trial's ``LedgerWriter``. If it also has ``events()`` (a ``LabLedger`` does), the meter replays the
    ``lab.budget_charged`` events of this trial on construction, so a meter built on a reopened ledger starts from what was booked.
    Meter factory used by the coordinator: ``lambda spec, ledger: TrialBudgetMeter(spec, ledger, spec.trial_id, clock=clock, lab_cap=cap)``.
    """

    def __init__(
        self,
        spec: BudgetSpec,
        ledger: LedgerWriter,
        trial_id: str,
        *,
        clock: Clock = time.time,
        lab_cap: Optional[LabCap] = None,
    ) -> None:
        if isinstance(spec, TrialSpec):  # the design's meter factory passes the TrialSpec; its budget is the allowance
            spec = spec.budget
        if not isinstance(spec, BudgetSpec):
            raise ContractError("TrialBudgetMeter: spec must be a BudgetSpec (or a TrialSpec, whose budget is used)")
        self._spec = spec
        self._ledger = ledger
        self._trial_id = trial_id
        self._actor = _ACTOR_PREFIX + trial_id
        self._clock = clock
        self._lab_cap = lab_cap
        self._lock = threading.RLock()
        self._tokens = 0.0
        self._gpu_minutes = 0.0
        self._researcher_usd = 0.0
        self._gpu_usd = 0.0
        self._booked: set[tuple[str, str]] = set()  # (call_id, resource value)
        self._reservations: dict[str, tuple[float, float]] = {}  # llm call_id -> (worst-case tokens, worst-case usd)
        self._logged: set[str] = set()  # axes whose lab.budget_exhausted is already in the ledger
        self._forced: set[str] = set()  # axes closed because a charge could not be recorded
        self._start = float(clock())
        self._replay()

    # ---- state ----------------------------------------------------------------------------------
    @property
    def spec(self) -> BudgetSpec:
        return self._spec

    @property
    def trial_id(self) -> str:
        return self._trial_id

    def _replay(self) -> None:
        events_fn = getattr(self._ledger, "events", None)
        if not callable(events_fn):
            return
        first_ts: Optional[float] = None
        pending: dict[str, tuple[float, float]] = {}
        for ev in events_fn():
            if first_ts is None:
                first_ts = float(ev.ts)
            if ev.kind == LabKind.BUDGET_RESERVED:
                held = reservation_from_event(ev.data, self._trial_id)
                if held is not None:
                    pending[held[0]] = held[1:]
            elif ev.kind == LabKind.BUDGET_CHARGED:
                try:
                    ch = BudgetCharge.from_dict(ev.data)
                except ContractError:
                    continue
                if ch.trial_id != self._trial_id:
                    continue
                self._apply(ch.resource, ch.amount, ch.call_id)
            elif ev.kind == LabKind.BUDGET_EXHAUSTED:
                axis = (ev.data or {}).get("axis")
                if isinstance(axis, str) and (ev.data or {}).get("trial_id") == self._trial_id:
                    self._logged.add(axis)
        # A reservation with no settlement charge was in flight when the process died: the request may have been forwarded and billed,
        # so it stays booked at its worst case (the same rule as a stream that ends without a usage chunk).
        # Settlement books tokens, then dollars: a crash between the two leaves only the unbooked part of the reservation outstanding.
        for call_id, (worst_tokens, worst_usd) in pending.items():
            tokens_held = 0.0 if (f"{call_id}:tokens", Resource.RESEARCHER_TOKENS.value) in self._booked else worst_tokens
            usd_held = 0.0 if (f"{call_id}:usd", Resource.RESEARCHER_USD.value) in self._booked else worst_usd
            if tokens_held or usd_held:
                self._reservations[call_id] = (tokens_held, usd_held)
                if usd_held and self._lab_cap is not None:
                    self._lab_cap.adopt(self._lab_key(call_id), usd_held)  # lab_spent_usd seeded these dollars as spent
        if first_ts is not None:
            self._start = first_ts

    def _apply(self, resource: Resource, amount: float, call_id: str) -> bool:
        """Add one charge to the in-memory totals. False if ``(call_id, resource)`` was already booked."""
        key = (call_id, resource.value)
        if key in self._booked:
            return False
        self._booked.add(key)
        if resource is Resource.RESEARCHER_TOKENS:
            self._tokens += amount
        elif resource is Resource.GPU_MINUTES:
            self._gpu_minutes += amount
        elif resource is Resource.RESEARCHER_USD:
            self._researcher_usd += amount
        elif resource is Resource.GPU_USD:
            self._gpu_usd += amount
        return True

    def _reserved_totals(self) -> tuple[float, float]:
        tokens = math.fsum(t for t, _ in self._reservations.values())
        usd = math.fsum(u for _, u in self._reservations.values())
        return tokens, usd

    def _used(self) -> dict[str, float]:
        """Used per axis, reservations included (the conservative figure every cap is compared with)."""
        rtok, rusd = self._reserved_totals()
        return {
            "researcher_tokens": self._tokens + rtok,
            "gpu_minutes": self._gpu_minutes,
            "dollars": self._researcher_usd + self._gpu_usd + rusd,
            "wall_seconds": max(0.0, float(self._clock()) - self._start),
        }

    def _limit(self, axis: str) -> float:
        return float(getattr(self._spec, axis))

    def usage(self) -> BudgetUsage:
        """Booked usage plus the worst case of every unsettled reservation (tokens and researcher dollars), wall time since start."""
        with self._lock:
            rtok, rusd = self._reserved_totals()
            return BudgetUsage(
                researcher_tokens=int(math.ceil(self._tokens + rtok)),
                gpu_minutes=self._gpu_minutes,
                researcher_usd=self._researcher_usd + rusd,
                gpu_usd=self._gpu_usd,
                wall_seconds=max(0.0, float(self._clock()) - self._start),
            )

    # ---- exhaustion -----------------------------------------------------------------------------
    def _log_exhausted(self, axis: str, limit: float, used: float, action: str) -> None:
        """Append lab.budget_exhausted once per axis. Never raises: the refusal does not depend on the ledger."""
        if axis in self._logged:
            return
        self._logged.add(axis)
        try:
            self._ledger.append(
                LabKind.BUDGET_EXHAUSTED,
                self._actor,
                BudgetExhaustedEvent(self._trial_id, axis, limit, used, action),
            )
        except Exception:  # noqa: BLE001 - fail closed: the caller raises BudgetExhausted regardless
            pass

    def _refuse(self, axis: str, limit: float, used: float, action: str) -> BudgetExhausted:
        self._log_exhausted(axis, limit, used, action)
        return BudgetExhausted(axis, limit, used)

    def check(self, resource: Resource) -> None:
        """Raise ``BudgetExhausted`` BEFORE a spend if the axis of ``resource`` is already used up (usage >= cap; a cap of 0 is used up at
        once), if a charge on that axis could not be recorded, or, for researcher tokens and dollars, if the lab-wide cap is used up."""
        resource = Resource(resource)
        axis = AXIS_OF[resource]
        with self._lock:
            used = self._used()[axis]
            limit = self._limit(axis)
            if axis in self._forced:
                raise self._refuse(axis, limit, used, "axis closed: a charge could not be recorded")
            if used >= limit:
                raise self._refuse(axis, limit, used, "check refused: axis used up")
            if self._lab_cap is not None and resource in (Resource.RESEARCHER_USD, Resource.RESEARCHER_TOKENS) and self._lab_cap.exhausted():
                raise self._refuse(LAB_AXIS, self._lab_cap.cap_usd, self._lab_cap.projected(), "check refused: lab-wide dollar cap used up")

    # ---- charging -------------------------------------------------------------------------------
    def _book(self, resource: Resource, amount: float, call_id: str, detail: Optional[Mapping[str, Any]], *, lab_spend: bool) -> Optional[BudgetExhausted]:
        """Validate, record and book one charge. Returns the ``BudgetExhausted`` to raise (never raises it itself) so a caller that books
        several charges can finish them all first. Idempotent per ``(call_id, resource)``."""
        resource = Resource(resource)
        if resource is Resource.WALL_SECONDS:
            raise ContractError("wall time is derived from the clock; it is checked, never charged")
        charge = BudgetCharge(self._trial_id, resource, amount, call_id, dict(detail or {}))  # validates amount, ids, detail
        axis = AXIS_OF[resource]
        if (call_id, resource.value) in self._booked:
            if axis in self._forced:  # a retry of a charge that could not be recorded must not look like success
                return self._refuse(axis, self._limit(axis), self._used()[axis], "axis closed: a charge could not be recorded")
            return None
        recorded = True
        try:
            self._ledger.append(LabKind.BUDGET_CHARGED, self._actor, charge)
        except Exception:  # noqa: BLE001 - fail closed: book in memory and close the axis
            recorded = False
        self._apply(resource, charge.amount, call_id)
        if lab_spend and self._lab_cap is not None:
            self._lab_cap.add_spent(charge.amount)
        used, limit = self._used()[axis], self._limit(axis)
        if not recorded:
            self._forced.add(axis)
            return self._refuse(axis, limit, used, "axis closed: a charge could not be recorded")
        if used >= limit:
            return self._refuse(axis, limit, used, f"charge crossed the cap: call {call_id}")
        return None

    def charge(self, resource: Resource, amount: float, call_id: str, detail: Optional[Mapping[str, Any]] = None) -> BudgetUsage:
        """Book ``amount`` (idempotent per ``(call_id, resource)``), append ``lab.budget_charged`` and return the usage. Raises
        ``BudgetExhausted`` after the charge that crossed a cap (the charge is booked). ``WALL_SECONDS`` is refused."""
        with self._lock:
            exc = self._book(resource, amount, call_id, detail, lab_spend=Resource(resource) is Resource.RESEARCHER_USD)
            if exc is not None:
                raise exc
            return self.usage()

    def reserve_llm(self, model: str, prompt_tokens_estimate: int, max_tokens: int, call_id: str) -> None:
        """Book the worst case of one LLM call before it is forwarded: ``prompt_tokens_estimate + max_tokens`` tokens and the same tokens
        priced with ``usd_for`` (no cache credit). Raises ``BudgetExhausted`` if that would cross the token cap, the dollar cap or the
        lab-wide cap; nothing is booked then. Idempotent per ``call_id``."""
        prompt = _count(prompt_tokens_estimate, "prompt_tokens_estimate")
        completion = _count(max_tokens, "max_tokens")
        with self._lock:
            if call_id in self._reservations or (f"{call_id}:tokens", Resource.RESEARCHER_TOKENS.value) in self._booked:
                return
            worst_tokens = float(prompt + completion)
            worst_usd = float(usd_for(model, Usage(prompt_tokens=prompt, completion_tokens=completion, cached_tokens=0, total_tokens=prompt + completion)))
            used = self._used()
            for axis, add in (("researcher_tokens", worst_tokens), ("dollars", worst_usd)):
                if axis in self._forced:
                    raise self._refuse(axis, self._limit(axis), used[axis], "axis closed: a charge could not be recorded")
                if used[axis] + add > self._limit(axis):
                    raise self._refuse(axis, self._limit(axis), used[axis] + add, f"reserve refused: call {call_id} would cross the cap")
            if self._lab_cap is not None:
                try:
                    self._lab_cap.reserve(self._lab_key(call_id), worst_usd)
                except BudgetExhausted as exc:
                    self._log_exhausted(LAB_AXIS, exc.limit, exc.used, f"reserve refused: call {call_id} would cross the lab-wide cap")
                    raise
            # Persist before the caller forwards anything, so a crash cannot erase spend the provider may have billed. Fail closed.
            try:
                self._ledger.append(
                    LabKind.BUDGET_RESERVED,
                    self._actor,
                    BudgetCharge(self._trial_id, Resource.RESEARCHER_TOKENS, worst_tokens, call_id, {"usd": worst_usd, "model": model}),
                )
            except Exception:  # noqa: BLE001
                if self._lab_cap is not None:
                    self._lab_cap.settle(self._lab_key(call_id), 0.0)
                self._forced.update(("researcher_tokens", "dollars"))
                raise self._refuse("researcher_tokens", self._limit("researcher_tokens"), used["researcher_tokens"], "axis closed: a reservation could not be recorded")
            self._reservations[call_id] = (worst_tokens, worst_usd)

    def charge_llm(self, model: str, usage: LlmUsage, call_id: str, detail: Optional[Mapping[str, Any]] = None) -> BudgetUsage:
        """Replace the reservation for ``call_id`` (if any) with the actual usage: tokens ``prompt + completion`` under
        ``<call_id>:tokens`` and dollars (``usd_for``, cached tokens at the cached rate) under ``<call_id>:usd``. Both are booked before
        ``BudgetExhausted`` is raised. ``detail`` (scalars) is added to both events, for example ``{"usage_estimated": True}``."""
        if not isinstance(usage, LlmUsage):
            raise ContractError("charge_llm: usage must be an LlmUsage")
        usd = float(
            usd_for(
                model,
                Usage(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens, cached_tokens=usage.cached_tokens, total_tokens=usage.total_tokens),
            )
        )
        if not math.isfinite(usd) or usd < 0:
            usd = 0.0
        info = {"model": model, "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "cached_tokens": usage.cached_tokens}
        info.update(dict(detail or {}))
        # Validate both suffixed charges before the reservation is released: a rejected id (the suffix can push a valid call_id past
        # the length limit) must leave the booked worst case in place, not erase it.
        for resource, suffix in ((Resource.RESEARCHER_TOKENS, ":tokens"), (Resource.RESEARCHER_USD, ":usd")):
            BudgetCharge(self._trial_id, resource, 0.0, call_id + suffix, dict(info))
        with self._lock:
            reservation = self._reservations.pop(call_id, None)
            new_usd = (f"{call_id}:usd", Resource.RESEARCHER_USD.value) not in self._booked
            first = self._book(Resource.RESEARCHER_TOKENS, float(usage.total_tokens), f"{call_id}:tokens", info, lab_spend=False)
            second = self._book(Resource.RESEARCHER_USD, usd, f"{call_id}:usd", info, lab_spend=False)
            if self._lab_cap is not None:
                # Settle exactly once: the reservation is replaced by the actual spend; an unreserved call adds its spend; a repeated
                # charge_llm (already booked) adds nothing.
                if reservation is not None:
                    self._lab_cap.settle(self._lab_key(call_id), usd if new_usd else 0.0)
                elif new_usd:
                    self._lab_cap.add_spent(usd)
            exc = first or second
            if exc is not None:
                raise exc
            return self.usage()

    def _lab_key(self, call_id: str) -> str:
        return f"{self._trial_id}:{call_id}"


def _count(v: Any, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ContractError(f"{what}: expected an integer >= 0, got {v!r}")
    return v


# ---- helpers for the coordinator and abort-trial --------------------------------------------------
def reservation_from_event(data: Any, trial_id: Optional[str]) -> Optional[tuple[str, float, float]]:
    """``(call_id, worst_tokens, worst_usd)`` of a ``lab.budget_reserved`` payload, or None if it is malformed or another trial's."""
    try:
        ch = BudgetCharge.from_dict(data)
    except ContractError:
        return None
    if (trial_id is not None and ch.trial_id != trial_id) or ch.resource is not Resource.RESEARCHER_TOKENS:
        return None
    usd = ch.detail.get("usd")
    if isinstance(usd, bool) or not isinstance(usd, (int, float)) or not math.isfinite(usd) or usd < 0:
        return None
    return ch.call_id, float(ch.amount), float(usd)


def usage_from_events(events: Iterable[Any], trial_id: Optional[str] = None) -> BudgetUsage:
    """Rebuild a trial's ``BudgetUsage`` by replaying its ledger events (``lab.budget_charged``; idempotent per ``(call_id, resource)``).
    Wall seconds are the span from the first to the last event of the ledger. A ``lab.budget_reserved`` event with no settlement
    (``<call_id>:tokens``) counts at its worst case, as the meter does after a reopen.
    Used by ``cli abort-trial`` and by ``lab_spent_usd``."""
    tokens = gpu_minutes = researcher_usd = gpu_usd = 0.0
    seen: set[tuple[str, str]] = set()
    pending: dict[str, tuple[float, float]] = {}
    first_ts: Optional[float] = None
    last_ts = 0.0
    for ev in events:
        if first_ts is None:
            first_ts = float(ev.ts)
        last_ts = float(ev.ts)
        if ev.kind == LabKind.BUDGET_RESERVED:
            held = reservation_from_event(ev.data, trial_id)
            if held is not None:
                pending[held[0]] = held[1:]
            continue
        if ev.kind != LabKind.BUDGET_CHARGED:
            continue
        try:
            ch = BudgetCharge.from_dict(ev.data)
        except ContractError:
            continue
        if trial_id is not None and ch.trial_id != trial_id:
            continue
        key = (ch.call_id, ch.resource.value)
        if key in seen:
            continue
        seen.add(key)
        if ch.resource is Resource.RESEARCHER_TOKENS:
            tokens += ch.amount
        elif ch.resource is Resource.GPU_MINUTES:
            gpu_minutes += ch.amount
        elif ch.resource is Resource.RESEARCHER_USD:
            researcher_usd += ch.amount
        elif ch.resource is Resource.GPU_USD:
            gpu_usd += ch.amount
    for call_id, (worst_tokens, worst_usd) in pending.items():
        if (f"{call_id}:tokens", Resource.RESEARCHER_TOKENS.value) not in seen:
            tokens += worst_tokens
        if (f"{call_id}:usd", Resource.RESEARCHER_USD.value) not in seen:
            researcher_usd += worst_usd
    wall = 0.0 if first_ts is None else max(0.0, last_ts - first_ts)
    return BudgetUsage(int(math.ceil(tokens)), gpu_minutes, researcher_usd, gpu_usd, wall)


def outer_usd_spent(outer_dir: Path | str) -> float:
    """Dollars the reused ``BudgetGate`` booked for Wonder calls in ``outer/ledger.jsonl`` (``budget.charged`` events with unit usd).
    0.0 when there is no such ledger."""
    path = Path(outer_dir) / "ledger.jsonl"
    if not path.exists():
        return 0.0
    from nightshift.ledger import Ledger  # local: only the cap seeding needs the reused class

    led = Ledger.read_only(outer_dir)
    try:
        total = 0.0
        for ev in led.events():
            if ev.kind == "budget.charged" and (ev.data or {}).get("unit") == "usd":
                amount = ev.data.get("amount")
                if isinstance(amount, (int, float)) and not isinstance(amount, bool) and math.isfinite(amount) and amount > 0:
                    total += float(amount)
        return total
    finally:
        led.close()


def lab_spent_usd(state: LabState, layout: Optional[LabLayout] = None) -> float:
    """The figure that seeds ``LabCap.spent_usd``: researcher-inference dollars of every finished trial, plus (with ``layout``) the
    dollars of trials that started but have no ``lab.trial_finished`` yet (their trial ledger is replayed read-only), plus the Wonder
    dollars in ``outer/ledger.jsonl``. Without ``layout`` the Wonder figure is the sum of the candidates' ``propose_usd``."""
    spent = math.fsum(rec.budget_used.researcher_usd for rec in state.trials.values())
    if layout is None:
        return spent + math.fsum(c.propose_usd for c in state.candidates.values())
    from nightshift.lab.ledger import LabLedger  # local import: ledger.py does not depend on this module

    for trial_id in state.trials_started:
        if trial_id in state.trials:
            continue
        if not (layout.trial_ledger_dir(trial_id) / "ledger.jsonl").exists():
            continue
        led = LabLedger.read_only(layout.trial_ledger_dir(trial_id))
        try:
            spent += usage_from_events(led.events(), trial_id).researcher_usd
        finally:
            led.close()
    return spent + outer_usd_spent(layout.outer_dir)
