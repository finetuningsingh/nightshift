"""``ModelServer`` over a vLLM that is already running on ONE rental the operator owns (the paired reference experiment).

The normal ``RentalModelServer`` rents a GPU for each scoring session. The reference experiment generates, trains and scores on a single
bounded rental (a second rental would exceed the owner's cap), so scoring there must reuse the endpoint that is already up. This server
changes nothing about scoring itself: the request still goes through ``score_adapter`` / ``ScoreService``, the guard and the ledger. It only
answers ``serve`` with the existing endpoint and never rents or deletes anything.

``close`` reports the scoring window as GPU time at the rental's rate, with the rental id, so the ledger's ``lab.score_cost`` is not zero;
the rental's own ledger (``RentalManager``) remains the authority on what was actually billed.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Callable, Mapping, Optional

from nightshift.lab.contracts import BudgetMeter, GpuUsage, ScorerError, ServeEndpoint, ServeSession, ServeSpec


class _AttachedSession:
    def __init__(self, endpoint: ServeEndpoint, started: float, clock: Callable[[], float], rate_usd_per_hour: float, rental_id: str):
        self._endpoint, self._started, self._clock, self._rate, self._rental_id = endpoint, started, clock, rate_usd_per_hour, rental_id
        self._usage: Optional[GpuUsage] = None

    @property
    def endpoint(self) -> ServeEndpoint:
        return self._endpoint

    def close(self) -> GpuUsage:
        if self._usage is None:
            minutes = max(0.0, (self._clock() - self._started) / 60.0)
            self._usage = GpuUsage(round(minutes, 3), round(minutes / 60.0 * self._rate, 6), (self._rental_id,))
        return self._usage

    def __enter__(self) -> "ServeSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AttachedModelServer:
    """``base_url`` has no ``/v1`` suffix (the generator appends it). ``adapter_names`` maps an adapter's listing sha256 to the LoRA name the
    server was started with. ``serve`` checks that the server lists the base model and every adapter the spec needs, so a wrong or dead
    endpoint refuses before any prompt is sent."""

    def __init__(self, base_url: str, base_name: str, adapter_names: Mapping[str, str], *, rental_id: str, rate_usd_per_hour: float,
                 clock: Callable[[], float] = time.time, list_models: Optional[Callable[[str], list]] = None):
        if rate_usd_per_hour < 0:
            raise ScorerError("rate_usd_per_hour must be non-negative")
        self._base_url, self._base_name, self._adapters = base_url.rstrip("/"), base_name, dict(adapter_names)
        self._rental_id, self._rate, self._clock = rental_id, float(rate_usd_per_hour), clock
        self._list_models = list_models or _http_models

    def serve(self, spec: ServeSpec, meter: Optional[BudgetMeter] = None) -> ServeSession:
        served = set(self._list_models(self._base_url))
        needed = {self._base_name}
        for adapter in spec.adapters:
            if adapter.sha256 not in self._adapters:
                raise ScorerError(f"the attached server does not serve adapter {adapter.sha256}")
            needed.add(self._adapters[adapter.sha256])
        missing = sorted(needed - served)
        if missing:
            raise ScorerError(f"the attached server at {self._base_url} does not list {missing}; it lists {sorted(served)}")
        endpoint = ServeEndpoint(self._base_url, self._base_name, {sha: self._adapters[sha] for sha in self._adapters})
        return _AttachedSession(endpoint, self._clock(), self._clock, self._rate, self._rental_id)


def _http_models(base_url: str, timeout: float = 15.0) -> list:
    try:
        with urllib.request.urlopen(base_url + "/v1/models", timeout=timeout) as resp:
            doc = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ScorerError(f"cannot list models at {base_url}: {exc}") from None
    return [m.get("id") for m in doc.get("data", []) if isinstance(m, dict) and isinstance(m.get("id"), str)]
