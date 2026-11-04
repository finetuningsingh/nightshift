"""Compute providers (spec 3.4).

Interface: estimate(request), create(request) -> rental, run(rental, job),
destroy(rental_id), list_owned(run_id).

The fake provider is CPU-only, deterministic, and labelled fake. The SSH
adapter is NOT YET VALIDATED LIVE: it refuses to operate rather than silently
succeeding, and fails clearly when its external tool is missing. No real
provider SDK is imported at module import time.
"""

from __future__ import annotations

import itertools
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .errors import ProviderError


class SimulatedClock:
    """Deterministic controllable clock, in epoch seconds."""

    def __init__(self, start: float = 1_700_000_000.0):
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("simulated clock cannot go backwards")
        self._now += float(seconds)
        return self._now


@dataclass(frozen=True)
class RentalRequest:
    gpu_type: str = "fake-gpu"
    hourly_price_usd: float = 1.0
    compute_seconds: float = 0.0
    setup_seconds: float = 0.0
    transfer_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    trial_id: str = "trial-0"
    run_id: str = "run-0"
    job_id: str = "job-0"
    lifetime_seconds: float | None = None

    @property
    def total_seconds(self) -> float:
        return (
            float(self.compute_seconds)
            + float(self.setup_seconds)
            + float(self.transfer_seconds)
            + float(self.evaluation_seconds)
        )

    @property
    def estimated_cost_usd(self) -> float:
        return float(self.hourly_price_usd) * self.total_seconds / 3600.0

    def validate(self) -> None:
        if self.hourly_price_usd < 0:
            raise ProviderError("hourly_price_usd cannot be negative")
        for name in ("compute_seconds", "setup_seconds", "transfer_seconds", "evaluation_seconds"):
            if float(getattr(self, name)) < 0:
                raise ProviderError(f"{name} cannot be negative")
        if self.lifetime_seconds is not None and self.lifetime_seconds <= 0:
            raise ProviderError("lifetime_seconds must be positive")

    def to_record(self) -> dict[str, Any]:
        return {
            "gpu_type": self.gpu_type,
            "hourly_price_usd": float(self.hourly_price_usd),
            "compute_seconds": float(self.compute_seconds),
            "setup_seconds": float(self.setup_seconds),
            "transfer_seconds": float(self.transfer_seconds),
            "evaluation_seconds": float(self.evaluation_seconds),
            "total_seconds": self.total_seconds,
            "estimated_cost_usd": self.estimated_cost_usd,
            "trial_id": self.trial_id,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "lifetime_seconds": self.lifetime_seconds,
        }


@dataclass
class Rental:
    rental_id: str
    provider: str
    run_id: str
    gpu_type: str
    hourly_price_usd: float
    created_at: float
    expires_at: float
    fake: bool = False
    status: str = "live"
    trial_id: str = ""
    reservation_id: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "rental_id": self.rental_id,
            "provider": self.provider,
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "gpu_type": self.gpu_type,
            "hourly_price_usd": float(self.hourly_price_usd),
            "created_at": float(self.created_at),
            "expires_at": float(self.expires_at),
            "status": self.status,
            "reservation_id": self.reservation_id,
            "fake": bool(self.fake),
        }


@dataclass(frozen=True)
class JobSpec:
    job_id: str = "job-0"
    trial_id: str = "trial-0"
    duration_seconds: float = 0.0
    should_fail: bool = False
    kind: str = "fake"
    command: tuple[str, ...] = ()


@dataclass
class JobResult:
    job_id: str
    status: str  # "succeeded" or "failed"; never a score
    reason: str = ""
    duration_seconds: float = 0.0
    fake: bool = False
    output: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    def to_record(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "reason": self.reason,
            "duration_seconds": float(self.duration_seconds),
            "fake": bool(self.fake),
            "output": dict(self.output),
        }


class Provider(ABC):
    name: str = "provider"
    fake: bool = False

    @abstractmethod
    def estimate(self, request: RentalRequest) -> dict[str, Any]:
        ...

    @abstractmethod
    def create(self, request: RentalRequest) -> Rental:
        ...

    @abstractmethod
    def run(self, rental: Rental, job: JobSpec) -> JobResult:
        ...

    @abstractmethod
    def destroy(self, rental_id: str) -> bool:
        ...

    @abstractmethod
    def list_owned(self, run_id: str) -> list[Rental]:
        ...


class FakeProvider(Provider):
    """CPU-only deterministic provider. Every record it makes is fake."""

    name = "fake"
    fake = True

    def __init__(self, clock: SimulatedClock | None = None):
        self.clock = clock or SimulatedClock()
        self._counter = itertools.count(1)
        self.rentals: dict[str, Rental] = {}
        self.create_calls = 0
        self.destroy_calls: list[str] = []

    def now(self) -> float:
        return self.clock.now()

    def estimate(self, request: RentalRequest) -> dict[str, Any]:
        request.validate()
        return {
            "provider": self.name,
            "estimated_cost_usd": request.estimated_cost_usd,
            "total_seconds": request.total_seconds,
            "fake": True,
        }

    def create(self, request: RentalRequest) -> Rental:
        request.validate()
        self.create_calls += 1
        created = self.clock.now()
        lifetime = (
            float(request.lifetime_seconds)
            if request.lifetime_seconds is not None
            else max(request.total_seconds, 1.0)
        )
        rental = Rental(
            rental_id=f"fake-rental-{next(self._counter)}",
            provider=self.name,
            run_id=request.run_id,
            trial_id=request.trial_id,
            gpu_type=request.gpu_type,
            hourly_price_usd=float(request.hourly_price_usd),
            created_at=created,
            expires_at=created + lifetime,
            fake=True,
        )
        self.rentals[rental.rental_id] = rental
        # Setup time is real rental time and is charged to the trial.
        if request.setup_seconds:
            self.clock.advance(float(request.setup_seconds))
        return rental

    def run(self, rental: Rental, job: JobSpec) -> JobResult:
        live = self.rentals.get(rental.rental_id)
        if live is None or live.status != "live":
            return JobResult(job.job_id, "failed", "rental_not_live", 0.0, fake=True)
        duration = float(job.duration_seconds)
        if duration:
            self.clock.advance(duration)
        if job.should_fail:
            return JobResult(job.job_id, "failed", "job_failed", duration, fake=True)
        return JobResult(job.job_id, "succeeded", "", duration, fake=True)

    def destroy(self, rental_id: str) -> bool:
        self.destroy_calls.append(rental_id)
        rental = self.rentals.get(rental_id)
        if rental is None:
            return False
        rental.status = "destroyed"
        return True

    def list_owned(self, run_id: str) -> list[Rental]:
        return [
            r for r in self.rentals.values() if r.run_id == run_id and r.status == "live"
        ]

    def list_all(self) -> list[Rental]:
        return list(self.rentals.values())

    def seed_foreign_rental(self, run_id: str = "someone-else") -> Rental:
        """Test helper: a live rental this run does not own."""
        now = self.clock.now()
        rental = Rental(
            rental_id=f"fake-rental-{next(self._counter)}",
            provider=self.name,
            run_id=run_id,
            gpu_type="fake-gpu",
            hourly_price_usd=1.0,
            created_at=now,
            expires_at=now + 3600,
            fake=True,
        )
        self.rentals[rental.rental_id] = rental
        return rental


class SSHProvider(Provider):
    """Operator-owned SSH GPU host adapter. NOT YET VALIDATED LIVE.

    This adapter has never been exercised against a real host in this
    repository. It therefore refuses to create rentals, and fails clearly
    when the `ssh` executable is missing, instead of silently succeeding.
    """

    name = "ssh"
    fake = False
    validated_live = False

    def __init__(self, host: str = "", user: str = ""):
        self.host = host
        self.user = user

    def _require_ssh(self) -> str:
        ssh = shutil.which("ssh")
        if ssh is None:
            raise ProviderError(
                "ssh provider unavailable: the 'ssh' executable was not found on PATH"
            )
        if not self.host:
            raise ProviderError("ssh provider unavailable: no host configured")
        return ssh

    def estimate(self, request: RentalRequest) -> dict[str, Any]:
        request.validate()
        return {
            "provider": self.name,
            "estimated_cost_usd": request.estimated_cost_usd,
            "total_seconds": request.total_seconds,
            "fake": False,
            "validated_live": False,
        }

    def create(self, request: RentalRequest) -> Rental:
        self._require_ssh()
        raise ProviderError(
            "ssh provider is not yet validated live; refusing to create a rental"
        )

    def run(self, rental: Rental, job: JobSpec) -> JobResult:
        self._require_ssh()
        raise ProviderError(
            "ssh provider is not yet validated live; refusing to run a job"
        )

    def destroy(self, rental_id: str) -> bool:
        self._require_ssh()
        raise ProviderError(
            "ssh provider is not yet validated live; refusing to destroy a rental"
        )

    def list_owned(self, run_id: str) -> list[Rental]:
        self._require_ssh()
        raise ProviderError("ssh provider is not yet validated live; cannot list rentals")


def make_provider(name: str, options: dict[str, Any] | None = None) -> Provider:
    options = options or {}
    if name == "fake":
        return FakeProvider()
    if name == "ssh":
        return SSHProvider(host=str(options.get("host", "")), user=str(options.get("user", "")))
    raise ProviderError(f"unknown provider: {name!r}")
