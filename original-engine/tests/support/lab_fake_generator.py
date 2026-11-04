"""Fakes for the scorer side of the lab harness (owner 1): a generator, a verifier backend, a model server and its session.

They implement the contract protocols (``Generator``, ``VerifierBackend``, ``ModelServer``, ``ServeSession``) with deterministic,
scriptable behaviour and record every call, so tests can assert what the production code asked for. Nothing here is imported by
production code.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Mapping, Optional, Sequence

from nightshift.lab.contracts import (
    GenerationParams,
    GenPrompt,
    GenResult,
    GpuUsage,
    ScorerError,
    ServeEndpoint,
    ServeSpec,
    Verifier,
)
from nightshift.lab.verifiers import PromptResult


class FakeGenerator:
    """``responses`` is a mapping prompt_id -> text, or a callable ``(model, prompt, params, seed) -> text``; the default answer is
    ``"OK"``. ``errors`` maps prompt_id -> error text (that prompt fails), ``truncated`` is a set of prompt ids that end with
    ``finish_reason == "length"``, ``raise_systemic`` makes ``generate`` raise ``ScorerError`` like a dead endpoint."""

    def __init__(self, responses: Any = None, *, errors: Optional[Mapping[str, str]] = None, truncated: Sequence[str] = (),
                 raise_systemic: Optional[str] = None):
        self._responses = responses
        self._errors = dict(errors or {})
        self._truncated = set(truncated)
        self._raise = raise_systemic
        self.calls: list = []  # (model, [prompt ids], params, seed)
        self._lock = threading.Lock()

    def generate(self, model: str, prompts: Sequence[GenPrompt], params: GenerationParams, seed: int) -> list:
        with self._lock:
            self.calls.append((model, [p.prompt_id for p in prompts], params, seed))
        if self._raise:
            raise ScorerError(self._raise)
        out = []
        for p in prompts:
            if p.prompt_id in self._errors:
                out.append(GenResult(p.prompt_id, "", "", 0, self._errors[p.prompt_id]))
                continue
            if callable(self._responses):
                text = self._responses(model, p, params, seed)
            elif self._responses is not None:
                text = self._responses.get(p.prompt_id, "OK")
            else:
                text = "OK"
            finish = "length" if p.prompt_id in self._truncated else "stop"
            out.append(GenResult(p.prompt_id, text, finish, len(text.split()), None))
        return out

    @property
    def models_seen(self) -> list:
        return [c[0] for c in self.calls]


class FakeVerifierBackend:
    """``VerifierBackend`` without third-party code. ``rule(row, response) -> list[bool]`` decides each constraint (default: every
    constraint passes iff the response is not blank). For the test verifier ``loose_rule`` defaults to ``rule``."""

    def __init__(self, kind: Verifier = Verifier.IFEVALG, rule: Optional[Callable[[Mapping[str, Any], str], Sequence[bool]]] = None,
                 loose_rule: Optional[Callable[[Mapping[str, Any], str], Sequence[bool]]] = None, pins: Optional[Mapping[str, str]] = None,
                 error_keys: Sequence[str] = ()):
        self.kind = Verifier(kind)
        self._rule = rule
        self._loose_rule = loose_rule
        self._pins = dict(pins) if pins is not None else {"fake_verifier": "1"}
        self._error_keys = set(error_keys)
        self.calls: list = []  # (keys, responses, mode)

    @staticmethod
    def _ids(row: Mapping[str, Any]) -> tuple:
        ids = row.get("instruction_id_list") if row.get("instruction_id_list") is not None else row.get("instruction_ids")
        if not ids:
            raise ScorerError(f"fake verifier: row {row.get('key')!r} has no constraint ids")
        return tuple(ids)

    def _flags(self, rule, row, text) -> tuple:
        ids = self._ids(row)
        if rule is None:
            return tuple([bool(text.strip())] * len(ids))
        flags = tuple(bool(f) for f in rule(row, text))
        if len(flags) != len(ids):
            raise ScorerError("fake verifier rule returned the wrong number of flags")
        return flags

    def score(self, rows: Sequence[Mapping[str, Any]], responses: Sequence[str], *, mode: str = "both") -> list:
        if len(rows) != len(responses):
            raise ScorerError("fake verifier: rows and responses differ in length")
        self.calls.append(([str(r.get("key")) for r in rows], list(responses), mode))
        out = []
        for row, text in zip(rows, responses):
            key = str(row.get("key"))
            ids = self._ids(row)
            if key in self._error_keys:
                dead = tuple([False] * len(ids))
                out.append(PromptResult(key, self.kind, ids, dead, dead if self.kind is Verifier.IFBENCH_UPSTREAM else None, True, 1))
                continue
            strict = self._flags(self._rule, row, text)
            loose = None
            if self.kind is Verifier.IFBENCH_UPSTREAM and mode == "both":
                loose = self._flags(self._loose_rule or self._rule, row, text)
            out.append(PromptResult(key, self.kind, ids, strict, loose, False, 0))
        return out

    def pins(self) -> dict:
        return dict(self._pins)


class FakeServeSession:
    """``ServeSession``: records ``close()`` and returns the usage it was built with."""

    def __init__(self, spec: ServeSpec, usage: GpuUsage, base_url: str = "http://127.0.0.1:18000/v1", base_name: str = "glimmer-base"):
        self.spec = spec
        self._usage = usage
        self.close_calls = 0
        adapters = {a.sha256: f"lora-{a.sha256[:8]}" for a in spec.adapters}
        self._endpoint = ServeEndpoint(base_url, base_name, adapters)

    @property
    def endpoint(self) -> ServeEndpoint:
        return self._endpoint

    def close(self) -> GpuUsage:
        self.close_calls += 1
        return self._usage

    def __enter__(self) -> "FakeServeSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class FakeModelServer:
    """``ModelServer``: ``serve`` returns a ``FakeServeSession``. ``fail`` is an exception instance raised by ``serve``."""

    def __init__(self, usage: Optional[GpuUsage] = None, fail: Optional[BaseException] = None):
        self.usage = usage or GpuUsage(12.0, 0.26, ("fake-rental-1",))
        self.fail = fail
        self.specs: list = []
        self.meters: list = []
        self.sessions: list = []

    def serve(self, spec: ServeSpec, meter: Any = None) -> FakeServeSession:
        self.specs.append(spec)
        self.meters.append(meter)
        if self.fail is not None:
            raise self.fail
        session = FakeServeSession(spec, self.usage)
        self.sessions.append(session)
        return session
