"""Acceptance A1-A7: deterministic resource enforcement (spec 3.3, 3.4)."""

import pytest

from nightshift.budget import BudgetStore, LeaseStore, ResourceManager, stop_from_leases
from nightshift.errors import ReservationRejected, ResourceError
from nightshift.ledger import EventLedger, read_events
from nightshift.providers import FakeProvider, JobSpec, RentalRequest, SimulatedClock


def make_manager(tmp_path, ceiling=10.0, per_trial=1000.0, lifetime=500.0, concurrency=2,
                 budget_path=None, run_id="run-1", with_ledger=False, leases_name="leases.json"):
    clock = SimulatedClock()
    provider = FakeProvider(clock=clock)
    ledger = EventLedger(tmp_path / "ledger.jsonl") if with_ledger else None
    manager = ResourceManager(
        budget_store=BudgetStore(budget_path or tmp_path / "budget.json", ceiling_usd=ceiling),
        lease_store=LeaseStore(tmp_path / leases_name),
        provider=provider,
        per_trial_gpu_seconds=per_trial,
        max_rental_lifetime_seconds=lifetime,
        max_concurrent_rentals=concurrency,
        run_id=run_id,
        ledger=ledger,
    )
    return manager, provider, clock


def request(**overrides):
    values = dict(
        hourly_price_usd=3600.0,  # $1 per second, easy arithmetic
        compute_seconds=5,
        setup_seconds=0,
        transfer_seconds=0,
        evaluation_seconds=0,
        trial_id="trial-1",
        run_id="run-1",
    )
    values.update(overrides)
    return RentalRequest(**values)


def test_a1_reservation_over_ceiling_rejected_before_provider_create(tmp_path):
    manager, provider, _ = make_manager(tmp_path, ceiling=10.0)
    with pytest.raises(ReservationRejected):
        manager.create_rental(request(compute_seconds=11))  # $11 > $10 ceiling
    assert provider.create_calls == 0


def test_a1_second_reservation_counts_committed_plus_reserved(tmp_path):
    manager, provider, _ = make_manager(tmp_path, ceiling=10.0)
    manager.reserve(request(compute_seconds=7))
    with pytest.raises(ReservationRejected):
        manager.reserve(request(compute_seconds=4, trial_id="trial-2"))
    assert provider.create_calls == 0


def test_a2_seconds_over_rental_lifetime_rejected_before_provider_call(tmp_path):
    manager, provider, _ = make_manager(tmp_path, ceiling=10_000.0, lifetime=100.0)
    with pytest.raises(ReservationRejected, match="lifetime"):
        manager.create_rental(request(compute_seconds=60, setup_seconds=60))
    assert provider.create_calls == 0


def test_a2_seconds_over_remaining_trial_budget_rejected_before_provider_call(tmp_path):
    manager, provider, _ = make_manager(tmp_path, ceiling=10_000.0, per_trial=100.0, lifetime=1000.0)
    manager.reserve(request(compute_seconds=80))
    with pytest.raises(ReservationRejected, match="per-trial"):
        manager.create_rental(request(compute_seconds=30))
    assert provider.create_calls == 0


def test_a3_ceiling_shared_across_two_run_roots_one_budget_file(tmp_path):
    shared = tmp_path / "shared-budget.json"
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    root_a.mkdir(), root_b.mkdir()
    manager_a, _, _ = make_manager(root_a, ceiling=10.0, budget_path=shared, run_id="run-a")
    manager_b, provider_b, _ = make_manager(root_b, ceiling=10.0, budget_path=shared, run_id="run-b")
    manager_a.reserve(request(compute_seconds=8, run_id="run-a"))
    with pytest.raises(ReservationRejected):
        manager_b.create_rental(request(compute_seconds=3, run_id="run-b", trial_id="trial-b"))
    assert provider_b.create_calls == 0
    assert manager_b.budget.committed_and_reserved_usd() == pytest.approx(8.0)


def test_a4_failed_job_rental_time_including_setup_is_charged(tmp_path):
    manager, _, _ = make_manager(tmp_path, ceiling=10_000.0)
    execution = manager.run_and_destroy(
        request(compute_seconds=20, setup_seconds=10),
        JobSpec(job_id="job-fail", trial_id="trial-1", duration_seconds=20, should_fail=True),
    )
    assert execution.result.status == "failed"
    assert execution.charged_seconds == pytest.approx(30.0)  # setup + job
    assert manager.budget.trial_charged_seconds("trial-1") == pytest.approx(30.0)


def test_a5_expired_rental_destroyed_and_job_failed_lifetime_exceeded(tmp_path):
    manager, provider, clock = make_manager(tmp_path, ceiling=10_000.0, with_ledger=True)
    rental = manager.create_rental(request(compute_seconds=10, lifetime_seconds=10))
    clock.advance(15)
    reaped = manager.reap_expired()
    assert reaped == [rental.rental_id]
    assert provider.rentals[rental.rental_id].status == "destroyed"
    events = read_events(tmp_path / "ledger.jsonl")
    failures = [e for e in events if e["type"] == "job_failed"]
    assert failures and failures[0]["payload"]["reason"] == "lifetime_exceeded"
    assert failures[0]["payload"]["job"]["reason"] == "lifetime_exceeded"
    assert manager.budget.trial_charged_seconds("trial-1") == pytest.approx(15.0)


def test_a6_concurrency_limit_blocks_extra_live_rental(tmp_path):
    manager, provider, _ = make_manager(tmp_path, ceiling=10_000.0, concurrency=1)
    manager.create_rental(request(compute_seconds=10))
    with pytest.raises(ReservationRejected, match="concurrency"):
        manager.create_rental(request(compute_seconds=10, trial_id="trial-2"))
    assert provider.create_calls == 1


def test_a7_stop_destroys_exactly_the_owned_live_rentals_listed(tmp_path):
    manager, provider, _ = make_manager(tmp_path, ceiling=10_000.0)
    owned = manager.create_rental(request(compute_seconds=10))
    foreign = provider.seed_foreign_rental()  # provider reports it; we do not own it
    summary = stop_from_leases(tmp_path / "leases.json", provider)
    assert summary["destroyed"] == [owned.rental_id]
    assert provider.rentals[owned.rental_id].status == "destroyed"
    assert provider.rentals[foreign.rental_id].status == "live"
    assert foreign.rental_id not in provider.destroy_calls
    assert summary["fake"] is True
    assert (tmp_path / "leases.json.stop.json").exists()


class _StubProvider:
    """Provider stub handed to stop/destroy as data; never a shell."""

    fake = True

    def __init__(self, name, confirmed):
        self.name = name
        self.confirmed = confirmed
        self.destroy_calls = []

    def destroy(self, rental_id):
        self.destroy_calls.append(rental_id)
        return self.confirmed


def test_b4_stop_missing_lease_file_is_refused_and_not_created(tmp_path):
    missing = tmp_path / "missing-leases.json"
    with pytest.raises(ResourceError, match="does not exist"):
        stop_from_leases(missing, _StubProvider("fake", True))
    assert not missing.exists()


def test_b4_stop_provider_mismatch_is_refused(tmp_path):
    manager, provider, _ = make_manager(tmp_path, ceiling=10_000.0)
    owned = manager.create_rental(request(compute_seconds=10))
    lease = manager.leases.find(owned.rental_id)
    assert lease["provider"] == "fake"
    other = _StubProvider("some-other-provider", True)
    with pytest.raises(ResourceError, match="do not match"):
        stop_from_leases(tmp_path / "leases.json", other)
    assert other.destroy_calls == []
    assert manager.leases.find(owned.rental_id)["status"] == "live"
    assert provider.rentals[owned.rental_id].status == "live"


def test_b4_stop_unconfirmed_destroy_stays_live_for_retry(tmp_path):
    manager, _provider, _ = make_manager(tmp_path, ceiling=10_000.0)
    owned = manager.create_rental(request(compute_seconds=10))
    failing = _StubProvider("fake", False)
    summary = stop_from_leases(tmp_path / "leases.json", failing)
    assert summary["destroyed"] == []
    assert summary["stop_failed"] == [owned.rental_id]
    assert summary["ok"] is False
    lease = manager.leases.find(owned.rental_id)
    assert lease["status"] == "stop_failed"
    assert lease["provider_confirmed"] is False
    # Still counted live, so a later confirmed stop can destroy it.
    assert [x["rental_id"] for x in manager.leases.live()] == [owned.rental_id]
    retry = stop_from_leases(tmp_path / "leases.json", _StubProvider("fake", True))
    assert retry["destroyed"] == [owned.rental_id] and retry["ok"] is True
    assert manager.leases.find(owned.rental_id)["status"] == "destroyed"


def test_b4_destroy_rental_unconfirmed_does_not_settle(tmp_path):
    manager, provider, _ = make_manager(tmp_path, ceiling=10_000.0)
    rental = manager.create_rental(request(compute_seconds=10))
    provider.destroy = lambda rental_id: False  # unconfirmed, as data
    with pytest.raises(ResourceError, match="not confirmed"):
        manager.destroy_rental(rental.rental_id)
    assert manager.budget.trial_charged_seconds("trial-1") == pytest.approx(0.0)
    assert manager.leases.find(rental.rental_id)["status"] == "stop_failed"


def test_b4_cli_stop_requires_provider_and_fails_on_unconfirmed(tmp_path):
    from nightshift.cli import main
    from nightshift.providers import Rental

    store = LeaseStore(tmp_path / "leases.json")
    store.add(Rental(
        rental_id="fake-rental-cli", provider="fake", run_id="run-1",
        gpu_type="fake-gpu", hourly_price_usd=1.0, created_at=1.0,
        expires_at=9999999999.0, fake=True,
    ))
    with pytest.raises(SystemExit):
        main(["stop", "--leases", str(tmp_path / "leases.json")])
    # A fresh fake provider cannot confirm this rental, so stop exits 1
    # and the lease stays live instead of being falsely marked destroyed.
    assert main(["stop", "--leases", str(tmp_path / "leases.json"),
                 "--provider", "fake"]) == 1
    assert LeaseStore(tmp_path / "leases.json").leases()[0]["status"] == "stop_failed"
    assert main(["stop", "--leases", str(tmp_path / "missing.json"),
                 "--provider", "fake"]) == 1


def test_fake_provider_is_labelled_and_deterministic(tmp_path):
    manager, provider, _ = make_manager(tmp_path, ceiling=10_000.0)
    assert provider.fake is True
    estimate = provider.estimate(request(compute_seconds=2))
    assert estimate["fake"] is True and estimate["estimated_cost_usd"] == 2.0
    rental = manager.create_rental(request(compute_seconds=2))
    assert rental.to_record()["fake"] is True
    result = manager.run_job(rental, JobSpec(duration_seconds=2))
    assert result.succeeded and result.to_record()["fake"] is True
