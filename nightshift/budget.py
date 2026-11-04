"""Deterministic resource enforcement (spec 3.3).

The controller's arithmetic -- never agent confidence -- authorises spending:
aggregate dollar ceiling shared via one budget file, reservation before
rental, per-trial GPU-seconds, per-rental lifetime, concurrency limit,
exact-owned cleanup through a lease store, and off-host stop.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .errors import BudgetError, ReservationRejected, ResourceError
from .ledger import EventLedger
from .providers import JobResult, JobSpec, Provider, Rental, RentalRequest

EPSILON = 1e-9


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


@contextmanager
def _file_lock(path: Path, timeout: float = 10.0) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() >= deadline:
                raise BudgetError(f"timed out waiting for budget lock {lock_path}")
            time.sleep(0.01)
    try:
        yield
    finally:
        os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


@dataclass
class Reservation:
    reservation_id: str
    run_id: str
    trial_id: str
    amount_usd: float
    total_seconds: float
    status: str = "reserved"

    def to_record(self) -> dict[str, Any]:
        return {
            "reservation_id": self.reservation_id,
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "amount_usd": float(self.amount_usd),
            "total_seconds": float(self.total_seconds),
            "status": self.status,
        }


class BudgetStore:
    """JSON budget file shared across run roots that point at the same path."""

    def __init__(self, path: str | Path, ceiling_usd: float | None = None):
        self.path = Path(path)
        with _file_lock(self.path):
            data = self._load_unlocked()
            if data is None:
                if ceiling_usd is None:
                    raise BudgetError("a new budget file requires an explicit ceiling_usd")
                data = {
                    "version": 1,
                    "ceiling_usd": float(ceiling_usd),
                    "reservations": {},
                    "trial_charges_seconds": {},
                }
                _atomic_write_json(self.path, data)
            elif ceiling_usd is not None and abs(float(data["ceiling_usd"]) - float(ceiling_usd)) > EPSILON:
                raise BudgetError(
                    f"budget file ceiling {data['ceiling_usd']} does not match "
                    f"configured ceiling {ceiling_usd}"
                )
            self.ceiling_usd = float(data["ceiling_usd"])

    def _load_unlocked(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BudgetError(f"corrupt budget file {self.path}") from exc
        data.setdefault("reservations", {})
        data.setdefault("trial_charges_seconds", {})
        return data

    def _update(self, mutator: Callable[[dict[str, Any]], Any]) -> Any:
        with _file_lock(self.path):
            data = self._load_unlocked()
            if data is None:
                raise BudgetError(f"budget file vanished: {self.path}")
            result = mutator(data)
            _atomic_write_json(self.path, data)
            return result

    def snapshot(self) -> dict[str, Any]:
        with _file_lock(self.path):
            data = self._load_unlocked()
            if data is None:
                raise BudgetError(f"budget file vanished: {self.path}")
            return json.loads(json.dumps(data))

    @staticmethod
    def _active_amount(data: dict[str, Any]) -> float:
        return sum(
            float(r["amount_usd"])
            for r in data["reservations"].values()
            if r.get("status") in ("reserved", "committed")
        )

    def committed_and_reserved_usd(self) -> float:
        return self._active_amount(self.snapshot())

    def remaining_usd(self) -> float:
        return self.ceiling_usd - self.committed_and_reserved_usd()

    def trial_charged_seconds(self, trial_id: str) -> float:
        return float(self.snapshot()["trial_charges_seconds"].get(trial_id, 0.0))

    def trial_reserved_seconds(self, trial_id: str) -> float:
        return sum(
            float(r["total_seconds"])
            for r in self.snapshot()["reservations"].values()
            if r.get("status") == "reserved" and r.get("trial_id") == trial_id
        )

    def active_reservations(self) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.snapshot()["reservations"].values()
            if r.get("status") == "reserved"
        ]

    def reserve(
        self, run_id: str, trial_id: str, amount_usd: float, total_seconds: float
    ) -> Reservation:
        reservation = Reservation(
            reservation_id=f"res-{uuid.uuid4().hex[:12]}",
            run_id=run_id,
            trial_id=trial_id,
            amount_usd=float(amount_usd),
            total_seconds=float(total_seconds),
        )

        def mutator(data: dict[str, Any]) -> Reservation:
            active = self._active_amount(data)
            ceiling = float(data["ceiling_usd"])
            if active + reservation.amount_usd > ceiling + EPSILON:
                raise ReservationRejected(
                    f"reservation of ${reservation.amount_usd:.6f} would exceed the "
                    f"aggregate ceiling ${ceiling:.6f} (committed+reserved ${active:.6f})"
                )
            data["reservations"][reservation.reservation_id] = reservation.to_record()
            return reservation

        return self._update(mutator)

    def release(self, reservation_id: str) -> None:
        def mutator(data: dict[str, Any]) -> None:
            record = data["reservations"].get(reservation_id)
            if record is not None and record.get("status") == "reserved":
                record["status"] = "released"

        self._update(mutator)

    def settle(
        self, reservation_id: str | None, actual_amount_usd: float, trial_id: str, charged_seconds: float
    ) -> None:
        def mutator(data: dict[str, Any]) -> None:
            if reservation_id is not None:
                record = data["reservations"].get(reservation_id)
                if record is not None and record.get("status") == "reserved":
                    record["status"] = "committed"
                    record["amount_usd"] = float(actual_amount_usd)
            charges = data["trial_charges_seconds"]
            charges[trial_id] = float(charges.get(trial_id, 0.0)) + float(charged_seconds)

        self._update(mutator)

    def charge_trial(self, trial_id: str, seconds: float) -> float:
        def mutator(data: dict[str, Any]) -> float:
            charges = data["trial_charges_seconds"]
            charges[trial_id] = float(charges.get(trial_id, 0.0)) + float(seconds)
            return float(charges[trial_id])

        return float(self._update(mutator))


class LeaseStore:
    """Exact set of rentals this run owns. Cleanup never uses name patterns."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            _atomic_write_json(self.path, {"version": 1, "leases": []})

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ResourceError(f"unreadable lease file {self.path}") from exc

    def _save(self, data: dict[str, Any]) -> None:
        _atomic_write_json(self.path, data)

    def leases(self) -> list[dict[str, Any]]:
        return [dict(lease) for lease in self._load().get("leases", [])]

    def live(self) -> list[dict[str, Any]]:
        return [lease for lease in self.leases() if lease.get("status") == "live"]

    def pending_or_live(self) -> list[dict[str, Any]]:
        return [
            lease for lease in self.leases() if lease.get("status") in ("pending", "live")
        ]

    def find(self, rental_id: str) -> dict[str, Any] | None:
        for lease in self.leases():
            if lease.get("rental_id") == rental_id:
                return lease
        return None

    def add_pending(
        self, reservation_id: str | None, run_id: str, trial_id: str, fake: bool
    ) -> str:
        """Write a lease entry before the provider create call."""
        lease_id = f"lease-{uuid.uuid4().hex[:12]}"
        data = self._load()
        data["leases"].append(
            {
                "lease_id": lease_id,
                "rental_id": None,
                "reservation_id": reservation_id,
                "run_id": run_id,
                "trial_id": trial_id,
                "status": "pending",
                "fake": bool(fake),
            }
        )
        self._save(data)
        return lease_id

    def activate(self, lease_id: str, rental: Rental) -> None:
        data = self._load()
        for lease in data["leases"]:
            if lease.get("lease_id") == lease_id:
                lease.update(rental.to_record())
                lease["status"] = "live"
        self._save(data)

    def add(self, rental: Rental) -> str:
        lease_id = f"lease-{uuid.uuid4().hex[:12]}"
        data = self._load()
        record = rental.to_record()
        record["lease_id"] = lease_id
        record["status"] = "live"
        data["leases"].append(record)
        self._save(data)
        return lease_id

    def mark(self, rental_id: str | None, status: str, **extra: Any) -> None:
        data = self._load()
        for lease in data["leases"]:
            if rental_id is not None and lease.get("rental_id") == rental_id:
                lease["status"] = status
                lease.update(extra)
        self._save(data)

    def mark_lease(self, lease_id: str, status: str, **extra: Any) -> None:
        data = self._load()
        for lease in data["leases"]:
            if lease.get("lease_id") == lease_id:
                lease["status"] = status
                lease.update(extra)
        self._save(data)


@dataclass
class RentalExecution:
    rental: Rental
    result: JobResult
    charged_seconds: float
    charged_usd: float


class ResourceManager:
    """Owns reservations, rentals, lifetimes, concurrency, and charging."""

    def __init__(
        self,
        budget_store: BudgetStore,
        lease_store: LeaseStore,
        provider: Provider,
        per_trial_gpu_seconds: float,
        max_rental_lifetime_seconds: float,
        max_concurrent_rentals: int,
        run_id: str = "run-0",
        ledger: EventLedger | None = None,
        clock: Any = None,
    ):
        self.budget = budget_store
        self.leases = lease_store
        self.provider = provider
        self.per_trial_gpu_seconds = float(per_trial_gpu_seconds)
        self.max_rental_lifetime_seconds = float(max_rental_lifetime_seconds)
        self.max_concurrent_rentals = int(max_concurrent_rentals)
        self.run_id = run_id
        self.ledger = ledger
        self._clock = clock
        self._settled_rentals: set[str] = set()

    @classmethod
    def from_config(
        cls,
        config: Any,
        provider: Provider,
        leases_path: str | Path,
        ledger: EventLedger | None = None,
        clock: Any = None,
    ) -> "ResourceManager":
        return cls(
            budget_store=BudgetStore(
                config.budgets.budget_file,
                ceiling_usd=config.budgets.aggregate_ceiling_usd,
            ),
            lease_store=LeaseStore(leases_path),
            provider=provider,
            per_trial_gpu_seconds=config.budgets.per_trial_gpu_seconds,
            max_rental_lifetime_seconds=config.budgets.max_rental_lifetime_seconds,
            max_concurrent_rentals=config.budgets.max_concurrent_rentals,
            run_id=config.run_name,
            ledger=ledger,
            clock=clock,
        )

    def now(self) -> float:
        clock = self._clock
        if clock is None:
            clock = getattr(self.provider, "clock", None)
        if clock is not None and hasattr(clock, "now"):
            return float(clock.now())
        if callable(clock):
            return float(clock())
        if hasattr(self.provider, "now"):
            return float(self.provider.now())
        return time.time()

    def _event(self, event_type: str, payload: dict[str, Any], actor: str = "controller") -> None:
        if self.ledger is not None:
            record = dict(payload)
            record.setdefault("fake", bool(getattr(self.provider, "fake", False)))
            self.ledger.append(event_type, actor, record)

    def _concurrency_used(self) -> int:
        lease_reservations = {
            lease.get("reservation_id")
            for lease in self.leases.pending_or_live()
            if lease.get("reservation_id")
        }
        unleased = [
            r
            for r in self.budget.active_reservations()
            if r.get("reservation_id") not in lease_reservations
        ]
        return len(self.leases.pending_or_live()) + len(unleased)

    def live_rental_count(self) -> int:
        return len(self.leases.live())

    def trial_used_seconds(self, trial_id: str) -> float:
        return self.budget.trial_charged_seconds(trial_id) + self.budget.trial_reserved_seconds(
            trial_id
        )

    def reserve(self, request: RentalRequest) -> Reservation:
        """Check every limit and reserve spend. No provider call happens here."""
        request.validate()
        total = request.total_seconds
        lifetime = (
            float(request.lifetime_seconds)
            if request.lifetime_seconds is not None
            else total
        )
        reasons: list[str] = []
        if total > self.max_rental_lifetime_seconds + EPSILON or lifetime > self.max_rental_lifetime_seconds + EPSILON:
            reasons.append(
                f"declared seconds {total} exceed max per-rental lifetime "
                f"{self.max_rental_lifetime_seconds}"
            )
        remaining_trial = self.per_trial_gpu_seconds - self.trial_used_seconds(request.trial_id)
        if total > remaining_trial + EPSILON:
            reasons.append(
                f"declared seconds {total} exceed remaining per-trial GPU-seconds "
                f"{remaining_trial} for trial {request.trial_id}"
            )
        if self._concurrency_used() >= self.max_concurrent_rentals:
            reasons.append(
                f"concurrency limit {self.max_concurrent_rentals} already reached"
            )
        if reasons:
            message = "; ".join(reasons)
            self._event(
                "reservation_rejected",
                {"request": request.to_record(), "reason": message},
            )
            raise ReservationRejected(message)
        try:
            reservation = self.budget.reserve(
                run_id=request.run_id or self.run_id,
                trial_id=request.trial_id,
                amount_usd=request.estimated_cost_usd,
                total_seconds=total,
            )
        except ReservationRejected as exc:
            self._event(
                "reservation_rejected",
                {"request": request.to_record(), "reason": str(exc)},
            )
            raise
        self._event(
            "spend_reserved",
            {"reservation": reservation.to_record(), "request": request.to_record()},
        )
        return reservation

    def release_reservation(self, reservation_id: str) -> None:
        self.budget.release(reservation_id)
        self._event("spend_released", {"reservation_id": reservation_id})

    def create_rental(self, request: RentalRequest) -> Rental:
        reservation = self.reserve(request)
        lease_id = self.leases.add_pending(
            reservation.reservation_id,
            request.run_id or self.run_id,
            request.trial_id,
            fake=bool(getattr(self.provider, "fake", False)),
        )
        try:
            rental = self.provider.create(request)
        except Exception:
            self.leases.mark_lease(lease_id, "failed")
            self.budget.release(reservation.reservation_id)
            raise
        rental.reservation_id = reservation.reservation_id
        rental.trial_id = rental.trial_id or request.trial_id
        # Controller-owned hard expiry, regardless of provider defaults.
        approved_lifetime = min(
            self.max_rental_lifetime_seconds,
            float(request.lifetime_seconds)
            if request.lifetime_seconds is not None
            else max(request.total_seconds, 1.0),
        )
        rental.expires_at = min(rental.expires_at, rental.created_at + approved_lifetime)
        self.leases.activate(lease_id, rental)
        self._event("rental_created", {"rental": rental.to_record()})
        return rental

    def _settle(self, rental: Rental, destroyed_at: float) -> tuple[float, float]:
        if rental.rental_id in self._settled_rentals:
            return 0.0, 0.0
        self._settled_rentals.add(rental.rental_id)
        charged_seconds = max(0.0, float(destroyed_at) - float(rental.created_at))
        charged_usd = float(rental.hourly_price_usd) * charged_seconds / 3600.0
        self.budget.settle(
            rental.reservation_id, charged_usd, rental.trial_id or "unknown", charged_seconds
        )
        return charged_seconds, charged_usd

    def destroy_rental(self, rental_id: str, reason: str = "destroyed") -> tuple[float, float]:
        lease = self.leases.find(rental_id)
        if lease is None:
            raise ResourceError(
                f"refusing to destroy unowned rental {rental_id!r}: not in this lease store"
            )
        if lease.get("status") == "destroyed":
            return 0.0, 0.0
        destroyed = self.provider.destroy(rental_id)
        now = self.now()
        self.leases.mark(
            rental_id,
            "destroyed",
            destroyed_at=now,
            destroy_reason=reason,
            provider_confirmed=bool(destroyed),
        )
        rental = Rental(
            rental_id=rental_id,
            provider=str(lease.get("provider", self.provider.name)),
            run_id=str(lease.get("run_id", self.run_id)),
            trial_id=str(lease.get("trial_id", "")),
            gpu_type=str(lease.get("gpu_type", "")),
            hourly_price_usd=float(lease.get("hourly_price_usd", 0.0)),
            created_at=float(lease.get("created_at", now)),
            expires_at=float(lease.get("expires_at", now)),
            fake=bool(lease.get("fake", False)),
            reservation_id=lease.get("reservation_id"),
        )
        charged_seconds, charged_usd = self._settle(rental, now)
        self._event(
            "rental_destroyed",
            {
                "rental_id": rental_id,
                "reason": reason,
                "charged_seconds": charged_seconds,
                "charged_usd": charged_usd,
                "provider_confirmed": bool(destroyed),
            },
        )
        return charged_seconds, charged_usd

    def run_job(self, rental: Rental, job: JobSpec) -> JobResult:
        if self.now() >= rental.expires_at:
            self.reap_expired()
            result = JobResult(
                job.job_id, "failed", "lifetime_exceeded", 0.0,
                fake=bool(getattr(self.provider, "fake", False)),
            )
            self._event("job_failed", {"job": result.to_record(), "rental_id": rental.rental_id})
            return result
        self._event(
            "job_started",
            {"job_id": job.job_id, "trial_id": job.trial_id, "rental_id": rental.rental_id},
        )
        try:
            result = self.provider.run(rental, job)
        except Exception as exc:  # a provider error is a failed job, not a crash of accounting
            result = JobResult(
                job.job_id, "failed", f"provider_error: {exc}", 0.0,
                fake=bool(getattr(self.provider, "fake", False)),
            )
        if self.now() > rental.expires_at and result.status == "succeeded":
            result = JobResult(
                result.job_id, "failed", "lifetime_exceeded", result.duration_seconds,
                fake=result.fake, output=result.output,
            )
        event = "job_finished" if result.succeeded else "job_failed"
        self._event(event, {"job": result.to_record(), "rental_id": rental.rental_id})
        return result

    def run_and_destroy(self, request: RentalRequest, job: JobSpec) -> RentalExecution:
        rental = self.create_rental(request)
        try:
            result = self.run_job(rental, job)
        finally:
            charged_seconds, charged_usd = self.destroy_rental(rental.rental_id)
        return RentalExecution(rental, result, charged_seconds, charged_usd)

    def reap_expired(self, now: float | None = None) -> list[str]:
        """Destroy rentals past hard expiry; record jobs as lifetime_exceeded."""
        moment = self.now() if now is None else float(now)
        reaped: list[str] = []
        for lease in self.leases.live():
            if moment >= float(lease.get("expires_at", moment + 1)):
                rental_id = str(lease["rental_id"])
                self._event(
                    "job_failed",
                    {
                        "job": {
                            "job_id": lease.get("job_id", "unknown"),
                            "status": "failed",
                            "reason": "lifetime_exceeded",
                            "fake": bool(lease.get("fake", False)),
                        },
                        "rental_id": rental_id,
                        "reason": "lifetime_exceeded",
                    },
                )
                self.destroy_rental(rental_id, reason="lifetime_exceeded")
                reaped.append(rental_id)
        return reaped


def stop_from_leases(
    leases_path: str | Path,
    provider: Provider,
    ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Off-host stop (spec 3.3): destroy exactly the live owned rentals listed.

    Needs only the lease file and provider credentials (the provider object).
    Rentals the provider reports but the lease file does not list as live and
    owned are never touched.
    """
    store = LeaseStore(leases_path)
    destroyed: list[str] = []
    already_gone: list[str] = []
    for lease in store.live():
        rental_id = lease.get("rental_id")
        if not rental_id:
            continue
        confirmed = provider.destroy(str(rental_id))
        store.mark(
            str(rental_id),
            "destroyed",
            destroy_reason="stop",
            provider_confirmed=bool(confirmed),
            stopped_at=time.time(),
        )
        (destroyed if confirmed else already_gone).append(str(rental_id))
    summary: dict[str, Any] = {
        "leases_file": str(leases_path),
        "destroyed": destroyed,
        "already_gone": already_gone,
        "fake": bool(getattr(provider, "fake", False)),
    }
    stop_record_path = Path(str(leases_path) + ".stop.json")
    prior: list[Any] = []
    if stop_record_path.exists():
        try:
            prior = json.loads(stop_record_path.read_text(encoding="utf-8")).get("stops", [])
        except (json.JSONDecodeError, AttributeError):
            prior = []
    prior.append(summary)
    _atomic_write_json(stop_record_path, {"stops": prior})
    if ledger_path is not None:
        EventLedger(ledger_path).append("rentals_stopped", "controller", summary)
    return summary
