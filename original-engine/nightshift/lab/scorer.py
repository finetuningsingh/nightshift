"""``ScoreService``: the one scorer for all three arms (implements ``contracts.Scorer``).

Same code path for stock Glimmer (arm 1, no adapter), v0 and evolved adapters, and for researcher-requested dev scores:

    access policy (ACCESS_POLICY, row by row) -> ScoreGuard -> load rows (hash-checked) -> generate per seed -> verify
    -> write gen_seed<k>.jsonl (private) and per_prompt.jsonl (flags only) -> write request.json and score.json -> return ScoreRecord

The service never appends a ledger event: the caller (trial runner, ``cli final``) appends ``lab.score_recorded``. It never builds a
record from missing responses: a generation error aborts the request before anything is written.

The ``guard`` is injected by the CLI (built from the ledger: ``final_guard`` for the final phase, the selection guard in the search).
The phase in a request is never trusted by itself: every request the table marks GUARD is passed to the guard, which derives the phase
from the ledger. Requests the table marks DENY never reach the guard, the data or the generator.
"""

from __future__ import annotations

import contextlib
import enum
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Collection, Mapping, Optional, Protocol, Sequence, runtime_checkable

from nightshift.lab.contracts import (
    METRIC_INSTRUCTION_LOOSE,
    METRIC_INSTRUCTION_STRICT,
    METRIC_PROMPT_ALL,
    METRIC_PROMPT_FRACTION,
    METRIC_PROMPT_LOOSE,
    METRIC_PROMPT_STRICT,
    METRICS_BY_SPLIT,
    TEST_ASSIGNMENT_ID,
    AccessDenied,
    AssignmentManifest,
    AssignmentRole,
    ContractError,
    Generator,
    GenPrompt,
    LabLayout,
    Phase,
    ScoreGuard,
    ScoreRecord,
    ScoreRequest,
    ScorerError,
    SeedAggregate,
    Split,
    Verifier,
    loads_strict,
    read_regular_bytes,
    scorer_version_of,
    utc_iso,
    write_record,
)
from nightshift.lab.verifiers import PromptResult, VerifierBackend, aggregate

# Bump when the scorer's own behaviour changes (aggregation, truncation rule, file formats); it is part of ``scorer_version``.
SCORER_MODULE_VERSION = 1

MAX_DATA_BYTES = 256 * 1024 * 1024
ROLE_TEST = "test"
ALL_METRICS = (
    METRIC_PROMPT_ALL,
    METRIC_PROMPT_FRACTION,
    METRIC_PROMPT_STRICT,
    METRIC_PROMPT_LOOSE,
    METRIC_INSTRUCTION_STRICT,
    METRIC_INSTRUCTION_LOOSE,
)


class Access(str, enum.Enum):
    ALLOW = "allow"  # no further check
    GUARD = "guard"  # allowed only if the injected ScoreGuard (derived from the ledger) passes
    DENY = "deny"  # AccessDenied before anything is read


# (phase, assignment role or "test", split) -> decision. Any combination not listed is DENY.
ACCESS_POLICY: dict = {
    (Phase.SEARCH, AssignmentRole.EVOLUTION.value, Split.DEV): Access.ALLOW,
    (Phase.SEARCH, AssignmentRole.EVOLUTION.value, Split.SELECTION): Access.GUARD,
    (Phase.SEARCH, AssignmentRole.HELDOUT.value, Split.DEV): Access.DENY,  # held-out data is unreachable in the search, dev included
    (Phase.SEARCH, AssignmentRole.HELDOUT.value, Split.SELECTION): Access.DENY,
    (Phase.SEARCH, ROLE_TEST, Split.TEST): Access.DENY,
    (Phase.FINAL, AssignmentRole.HELDOUT.value, Split.DEV): Access.GUARD,
    (Phase.FINAL, AssignmentRole.HELDOUT.value, Split.SELECTION): Access.GUARD,
    (Phase.FINAL, AssignmentRole.EVOLUTION.value, Split.DEV): Access.DENY,
    (Phase.FINAL, AssignmentRole.EVOLUTION.value, Split.SELECTION): Access.DENY,
    (Phase.FINAL, ROLE_TEST, Split.TEST): Access.GUARD,
}


def _atomic_write(path: Path, text: str, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if private:
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n" for r in rows)


@runtime_checkable
class EvidenceSource(Protocol):
    """A generator that can say what produced its generations (the Inspect task, its log hash, the code commit). The service puts the answer
    into the score record's ``verifier_pins``; a generator without it adds nothing."""

    def drain_evidence(self) -> Mapping[str, str]: ...


class ScoreService:
    """See the module docstring. ``verifiers`` maps ``Verifier.IFBENCH_UPSTREAM`` (test) and ``Verifier.IFEVALG`` (dev, selection) to a
    backend. ``expected_test_sha256``, when given, is the set of accepted hashes of the test file."""

    def __init__(self, layout: LabLayout, manifest: AssignmentManifest, verifiers: Mapping[Verifier, VerifierBackend], *,
                 guard: ScoreGuard, clock, expected_test_sha256: Optional[Collection[str]] = None):
        if not callable(guard) or not callable(clock):
            raise ScorerError("guard and clock must be callables")
        self._layout = layout
        self._manifest = manifest
        self._verifiers = {Verifier(k): v for k, v in verifiers.items()}
        self._guard = guard
        self._clock = clock
        self._expected_test = frozenset(expected_test_sha256) if expected_test_sha256 is not None else None

    # ------------------------------------------------------------------ Scorer
    @property
    def scorer_version(self) -> str:
        pins = {f"{v.value}.{k}": str(val) for v, backend in sorted(self._verifiers.items(), key=lambda kv: kv[0].value)
                for k, val in backend.pins().items()}
        return scorer_version_of(pins, ALL_METRICS, SCORER_MODULE_VERSION)

    def score(self, request: ScoreRequest, generator: Generator, model: str) -> ScoreRecord:
        if not isinstance(request, ScoreRequest):
            raise ScorerError("score() needs a ScoreRequest")
        if not isinstance(model, str) or not model.strip():
            raise ScorerError("score() needs the served model name")
        self.precheck(request)
        # One score per request id, end to end. The precheck above is only a cheap early refusal: two callers can both pass it before either
        # has written score.json. The lock is held from the re-check to the last write, so the second caller is refused instead of
        # generating, verifying and overwriting the first one's files (private data would be scored twice and the returned record could
        # disagree with the one on disk). flock is released by the OS if the process dies, so a crash leaves no stale lock.
        # The same holds for the LOGICAL score (phase, split, assignment, adapter) of a guarded split: the guard reads a ledger the caller
        # has not yet appended to, so two requests with different ids for the same logical score would both pass it. The logical lock and
        # the marker written with score.json close that gap for concurrent and for batched callers.
        with self._request_lock(request.request_id), self._logical_lock(request):
            if os.path.lexists(self._layout.score_record_path(request.request_id)):
                raise ScorerError(f"score {request.request_id} already exists; a score is written once and never re-rolled")
            self._check_logical_unscored(request, holding_lock=True)
            return self._score_locked(request, generator, model)

    @staticmethod
    def logical_key(request: ScoreRequest) -> Optional[str]:
        """Identity of the score a request would produce, for the splits that are scored once per (adapter, assignment, phase); ``None``
        for dev scores, which a researcher may repeat."""
        if request.split is Split.DEV:
            return None
        parts = [request.phase.value, request.split.value, request.assignment_id, request.adapter.sha256 if request.adapter is not None else "stock"]
        return hashlib.sha256(json.dumps(parts).encode("utf-8")).hexdigest()

    def _logical_marker(self, key: str) -> Path:
        return self._layout.private_dir / "score_locks" / f"logical-{key}.scored"

    def _check_logical_unscored(self, request: ScoreRequest, *, holding_lock: bool = False) -> None:
        """Refuse a logical score that already exists. A marker whose request has no score.json is what a process killed between the two
        writes leaves behind: nothing was revealed, so it does not count. With the logical lock held (``holding_lock``) such a marker is
        removed; otherwise it is ignored only when no writer holds the lock (a live writer between its two writes is not an orphan)."""
        key = self.logical_key(request)
        if key is None:
            return
        marker = self._logical_marker(key)
        if not os.path.lexists(marker):
            return
        refusal = ScorerError(f"a {request.split.value} score of this adapter on {request.assignment_id!r} already exists; a score is written once and never re-rolled")
        try:
            owner = marker.read_text(encoding="utf-8").strip()
            recorded = os.path.lexists(self._layout.score_record_path(owner))
        except (OSError, ContractError, ValueError):
            raise refusal from None
        if recorded:
            raise refusal
        if holding_lock:
            with contextlib.suppress(OSError):
                os.unlink(marker)
            return
        fd = self._flock(f"logical-{key}.lock", f"a {request.split.value} score of this adapter on {request.assignment_id!r} is already being computed; a score is written once and never re-rolled")
        os.close(fd)  # nobody is writing: the marker is an orphan of a crash and the locked re-check in score() removes it

    def _flock(self, name: str, message: str):
        lock_dir = self._layout.private_dir / "score_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_dir / name, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise ScorerError(message) from None
        return fd

    @contextlib.contextmanager
    def _request_lock(self, request_id: str):
        fd = self._flock(f"{request_id}.lock", f"score {request_id} is already being computed; a score is written once and never re-rolled")
        try:
            yield
        finally:
            os.close(fd)  # closing releases the lock

    @contextlib.contextmanager
    def _logical_lock(self, request: ScoreRequest):
        key = self.logical_key(request)
        if key is None:
            yield
            return
        fd = self._flock(f"logical-{key}.lock", f"a {request.split.value} score of this adapter on {request.assignment_id!r} is already being computed; a score is written once and never re-rolled")
        try:
            yield
        finally:
            os.close(fd)

    def _score_locked(self, request: ScoreRequest, generator: Generator, model: str) -> ScoreRecord:
        score_path = self._layout.score_record_path(request.request_id)

        backend = self._backend_for(request.split)
        rows, data_sha = self._load_rows(request)
        prompts = [GenPrompt(str(r["key"]), r["prompt"]) for r in rows]

        per_seed_results: list = []  # (seed, generations, texts, results)
        evidence: dict = {}
        for seed in request.seeds:
            gens = generator.generate(model, prompts, request.params, seed)
            self._check_generations(gens, prompts, seed)
            texts = ["" if g.truncated else g.text for g in gens]  # a truncated generation fails in every arm
            results = backend.score(rows, texts, mode="both")
            if len(results) != len(rows) or any(r.key != p.prompt_id for r, p in zip(results, prompts)):
                raise ScorerError("the verifier returned results that do not match the prompts")
            per_seed_results.append((seed, gens, results))
            if isinstance(generator, EvidenceSource):
                evidence.update(generator.drain_evidence())

        # everything generated and verified: now (and only now) write files
        score_dir = self._layout.score_dir(request.request_id)
        score_dir.mkdir(parents=True, exist_ok=True)
        aggregates = []
        per_prompt_rows: list = []
        for seed, gens, results in per_seed_results:
            _atomic_write(self._layout.gen_responses_path(request.request_id, seed), _jsonl([
                {"prompt_key": g.prompt_id, "text": g.text, "finish_reason": g.finish_reason,
                 "completion_tokens": g.completion_tokens, "error": g.error} for g in gens]), private=True)
            metrics = aggregate(results)
            aggregates.append(SeedAggregate(seed, len(results), metrics, sum(1 for g in gens if g.truncated),
                                            sum(1 for r in results if r.error)))
            per_prompt_rows.extend(self._per_prompt_row(request.split, seed, g, r) for g, r in zip(gens, results))
        per_prompt_path = self._layout.per_prompt_path(request.request_id)
        _atomic_write(per_prompt_path, _jsonl(per_prompt_rows))

        pins = dict(backend.pins())
        pins["prompts_sha256"] = data_sha
        for key, value in evidence.items():  # how the generations were produced: generator kind, Inspect task/log hashes, code commit
            if key in pins:
                raise ScorerError(f"generation evidence {key!r} would overwrite a verifier pin")
            pins[key] = str(value)
        if request.split is Split.TEST:
            pins.setdefault("strip_thinking", "true")
        try:
            record = ScoreRecord(
                score_id=request.request_id,
                request_id=request.request_id,
                phase=request.phase,
                assignment_id=request.assignment_id,
                split=request.split,
                trial_id=request.trial_id,
                arm=request.arm,
                final_request_id=request.final_request_id,
                base=request.base,
                adapter_sha256=request.adapter.sha256 if request.adapter is not None else None,
                scorer_version=request.scorer_version,
                verifier=Verifier.IFBENCH_UPSTREAM if request.split is Split.TEST else Verifier.IFEVALG,
                verifier_pins=pins,
                params=request.params,
                per_seed=tuple(aggregates),
                n_prompts=len(rows),
                per_prompt=self._layout.file_ref(per_prompt_path),
                headline_metric=METRICS_BY_SPLIT[request.split][0],
                created_utc=utc_iso(self._clock()),
            )
        except ContractError as exc:
            raise ScorerError(f"cannot build the score record: {exc}") from None
        write_record(self._layout.score_request_path(request.request_id), request)
        key = self.logical_key(request)
        marker = self._logical_marker(key) if key is not None else None
        if marker is not None:
            _atomic_write(marker, request.request_id + "\n", private=True)
        try:
            write_record(score_path, record)
        except BaseException:
            if marker is not None:  # no score.json: nothing was revealed, the logical score is still unscored
                try:
                    os.unlink(marker)
                except OSError:
                    pass
            raise
        return record

    def precheck(self, request: ScoreRequest) -> None:
        """Everything that can refuse a request before any data is read or any GPU is rented: the access table, the injected guard, the
        scorer version and the no-re-roll rule. ``score_session`` calls it before it serves a model; ``score`` calls it again (the guard only
        reads the ledger, so a second call is harmless)."""
        if not isinstance(request, ScoreRequest):
            raise ScorerError("precheck() needs a ScoreRequest")
        if self._check_access(request) is Access.GUARD:
            self._guard(request)  # raises AccessDenied / FinalComparisonRefused
        if request.scorer_version != self.scorer_version:
            raise ScorerError(f"request scorer_version {request.scorer_version!r} is not this scorer's {self.scorer_version!r}")
        if os.path.lexists(self._layout.score_record_path(request.request_id)):
            raise ScorerError(f"score {request.request_id} already exists; a score is written once and never re-rolled")
        self._check_logical_unscored(request)

    # ------------------------------------------------------------------ access
    def _role_of(self, assignment_id: str) -> str:
        if assignment_id == TEST_ASSIGNMENT_ID:
            return ROLE_TEST
        try:
            return self._manifest.get(assignment_id).role.value
        except KeyError:
            raise AccessDenied(f"assignment {assignment_id!r} is not in the frozen manifest") from None

    def _check_access(self, request: ScoreRequest) -> Access:
        role = self._role_of(request.assignment_id)
        decision = ACCESS_POLICY.get((request.phase, role, request.split), Access.DENY)
        if decision is Access.DENY:
            raise AccessDenied(f"{request.phase.value} phase: {request.split.value} of a {role} assignment may not be scored")
        return decision

    def _backend_for(self, split: Split) -> VerifierBackend:
        want = Verifier.IFBENCH_UPSTREAM if split is Split.TEST else Verifier.IFEVALG
        backend = self._verifiers.get(want)
        if backend is None:
            raise ScorerError(f"no {want.value} verifier is configured for the {split.value} split")
        return backend

    # ------------------------------------------------------------------ data
    def _data_path_and_revision(self, request: ScoreRequest):
        if request.split is Split.TEST:
            return self._layout.test_data(), None
        a = self._manifest.get(request.assignment_id)
        if request.split is Split.DEV:
            return self._layout.dev_data(request.assignment_id), a.data_revisions["dev_sha256"]
        return self._layout.selection_data(request.assignment_id), a.data_revisions["selection_sha256"]

    def _load_rows(self, request: ScoreRequest):
        path, want = self._data_path_and_revision(request)
        rel = path.relative_to(self._layout.root).as_posix()
        try:
            raw = read_regular_bytes(self._layout.root, rel, max_bytes=MAX_DATA_BYTES)
        except (ContractError, OSError) as exc:
            raise ScorerError(f"cannot read the {request.split.value} data of {request.assignment_id}: {exc}") from None
        digest = hashlib.sha256(raw).hexdigest()
        if want is not None and digest != want:
            raise ScorerError(f"{rel} does not match the frozen sha256 of assignment {request.assignment_id} ({request.split.value})")
        if self._expected_test is not None and request.split is Split.TEST and digest not in self._expected_test:
            raise ScorerError("the test file does not match any accepted sha256")
        rows = []
        seen = set()
        required = ("key", "prompt", "instruction_id_list", "kwargs") if request.split is Split.TEST else ("key", "prompt", "ground_truth", "instruction_ids")
        for n, line in enumerate(raw.decode("utf-8").split("\n"), 1):  # NOT splitlines(): U+2028 may sit inside a JSON string
            if not line.strip():
                continue
            try:
                row = loads_strict(line)
            except ContractError as exc:
                raise ScorerError(f"{rel} line {n}: {exc}") from None
            if not isinstance(row, dict) or any(k not in row for k in required):
                raise ScorerError(f"{rel} line {n}: a {request.split.value} row needs {required}")
            row["key"] = str(row["key"])
            if not row["key"] or row["key"] in seen:
                raise ScorerError(f"{rel} line {n}: empty or duplicate key {row['key']!r}")
            if not isinstance(row["prompt"], str) or not row["prompt"].strip():
                raise ScorerError(f"{rel} line {n}: empty prompt")
            seen.add(row["key"])
            rows.append(row)
        if not rows:
            raise ScorerError(f"{rel} has no rows")
        return rows, digest

    @staticmethod
    def _check_generations(gens: Sequence[Any], prompts: Sequence[GenPrompt], seed: int) -> None:
        if len(gens) != len(prompts) or any(g.prompt_id != p.prompt_id for g, p in zip(gens, prompts)):
            raise ScorerError(f"seed {seed}: the generator returned results that do not match the prompts")
        failed = [g for g in gens if g.error is not None]
        if failed:
            raise ScorerError(f"seed {seed}: {len(failed)} of {len(gens)} generations failed (first: {failed[0].error!r}); nothing is scored from missing responses")

    # ------------------------------------------------------------------ rows
    @staticmethod
    def _per_prompt_row(split: Split, seed: int, gen: Any, res: PromptResult) -> dict:
        if split is Split.TEST:
            if res.loose is None:
                raise ScorerError(f"the test verifier returned no loose flags for {res.key!r}")
            return {
                "prompt_key": res.key, "seed": seed,
                "strict_prompt": res.passed_strict, "loose_prompt": res.passed_loose,
                "strict_instructions": list(res.strict), "loose_instructions": list(res.loose),
                "truncated": gen.truncated, "error": res.error,
            }
        return {
            "prompt_key": res.key, "seed": seed, "passed": res.passed_strict, "fraction": res.fraction,
            "constraints": [{"id": i, "passed": f} for i, f in zip(res.instruction_ids, res.strict)],
            "truncated": gen.truncated, "error": res.error,
        }
