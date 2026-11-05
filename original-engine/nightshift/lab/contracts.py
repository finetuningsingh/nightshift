"""Shared contracts for the Glimmer three-arm harness (``nightshift.lab``).

This module is the only thing every other lab module may depend on without depending on a sibling. It holds:

* persisted records (frozen dataclasses with explicit ``to_dict`` / ``from_dict``),
* the enums, the ledger event kinds and the exceptions,
* the protocols the implementers must satisfy,
* the on-disk layout (``LabLayout``), pure path arithmetic.

Rules every record follows (reasons are in docs/harness-design.md, "Contracts"):

* Python 3.11+, standard library only, nothing imported from the rest of ``nightshift`` (so it loads without the sandbox).
* A record that is persisted as its own file or ledger payload carries ``schema_version``. Small value objects embedded in a
  record (``PackRef``, ``FileRef``, ``SeedAggregate``, ...) are versioned by their container and have no field of their own.
* ``from_dict`` rejects unknown fields, missing fields (``Optional`` fields must be present as ``null``), a wrong
  ``schema_version``, wrong types, non-finite numbers and negative budgets. Construction validates the same way
  (``__post_init__``), so an invalid record cannot be built in memory either.
* Strings that go into a ledger event are limited to ``MAX_TEXT_BYTES``: ``nightshift.ledger.Ledger`` silently truncates any
  string over 8192 bytes (nightshift/ledger.py ``cap_strings``) and writes ``NaN`` as a bare token (not strict JSON). Big
  payloads (per-prompt results, diffs, traces) are files, referenced by ``FileRef`` (path + sha256 + size).
* Records that have a ``dict`` field are NOT hashable (a frozen dataclass hashes its fields): never put a TrialSpec, TrialRecord,
  ScoreRecord, ... in a ``set`` or use one as a dict key; key by id. Records are compared with ``==``.
* Exchange types that are never persisted (Quote, GpuInstance, CreateReceipt, ServeSpec, ServeEndpoint, GenPrompt, GenResult,
  TrialContext, SubmittedAttempt, RuntimeResult, ProposalContext, ProposalResult, LabWorld) have no ``to_dict``: ``canonical_json``
  and ``write_record`` raise TypeError on them. Trial-scoped secrets (``TrialContext.llm_token``) are excluded from ``repr``.
* Every enum is a ``_StrEnum``: ``str(x)`` and ``f"{x}"`` are the value on every Python from 3.11 (3.12 changed mixed-in formatting).
* Times are ISO 8601 strings made by ``utc_iso(epoch)`` (always ``+00:00``, seconds precision). Components take ``clock() -> epoch``.
* Import this module by its absolute dotted name (``nightshift.lab.contracts``) and run lab code with ``python -m``;
  ``nightshift/lab/ledger.py`` and ``budget.py`` share a basename with ``nightshift/ledger.py`` and ``budget.py``.
"""

from __future__ import annotations

import contextlib
import enum
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

SCHEMA_VERSION = 1
LAYOUT_VERSION = 1

# nightshift/ledger.py cap_strings truncates strings above this many UTF-8 bytes. Records stay well below it.
LEDGER_STRING_CAP_BYTES = 8192
MAX_TEXT_BYTES = 4000

# The files that define an adapter's behaviour (docs: Glimmer adapter = PEFT adapter_config.json + adapter_model.safetensors).
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")

# Values the Glimmer chat template accepts for ``Reasoning strength:`` (Meta model card / chat_template.jinja).
REASONING_STRENGTHS = ("low", "medium", "high", "xhigh")

# The IFBench test set belongs to no assignment. Every ScoreRequest / ScoreRecord on the test split uses this assignment id and no
# other split may use it; an AssignmentManifest may not contain it.
TEST_ASSIGNMENT_ID = "ifbench_test"

# Metric names used by the scorer, stats and report (never spell them as literals elsewhere).
METRIC_PROMPT_ALL = "prompt_all"  # every constraint passes (IFEvalG); headline on dev and selection
METRIC_PROMPT_FRACTION = "prompt_fraction"  # open-instruct mean reward (IFEvalG)
METRIC_PROMPT_STRICT = "prompt_strict"  # upstream IFBench strict, prompt level; headline on test
METRIC_PROMPT_LOOSE = "prompt_loose"
METRIC_INSTRUCTION_STRICT = "instruction_strict"
METRIC_INSTRUCTION_LOOSE = "instruction_loose"

# Canonical keys of TrainResult.timings (seconds). The report prints exactly these.
TIMING_KEYS = ("download_s", "load_s", "generation_s", "train_s", "export_s", "eval_s")

# Keys of Assignment.data_revisions and TrialSpec.data_revisions (sha256 of the data files the trial used).
REV_TRAIN_SHA256 = "train_sha256"
REV_DEV_SHA256 = "dev_sha256"
REV_SELECTION_SHA256 = "selection_sha256"
DATA_REVISION_KEYS = (REV_TRAIN_SHA256, REV_DEV_SHA256, REV_SELECTION_SHA256)

# Keys of AssignmentManifest.source_revisions. The split salt is a secret kept under private/; only its hash is recorded.
SRC_IFRLVR_PARQUET_SHA256 = "ifrlvr_parquet_sha256"
SRC_IFBENCH_COMMIT = "ifbench_commit"
SRC_OPEN_INSTRUCT_COMMIT = "open_instruct_commit"
SRC_SPLIT_SALT_SHA256 = "split_salt_sha256"
SOURCE_REVISION_KEYS = (SRC_IFRLVR_PARQUET_SHA256, SRC_IFBENCH_COMMIT, SRC_OPEN_INSTRUCT_COMMIT, SRC_SPLIT_SALT_SHA256)

# Keys the lab.json pins must hold, and the logical secret names whose environment variable names lab.json records.
CONFIG_PIN_KEYS = ("coral_commit", "opencode_version", "ifbench_commit", "open_instruct_commit")
CONFIG_SECRET_NAMES = ("meta_api_key", "thunder_api_key")

# Statistics guards. MIN_POOLED_DF is the value the critics' simulation used (docs/harness-design.md, "Statistics").
MIN_POOLED_DF = 8
MAX_SCREEN_BATCH = 5
MIN_FINAL_TRIALS_PER_PACK = 5
CONTAMINATION_NGRAM = 13

# Adapters live under the lab root at this relative directory; AdapterRef.path is always artifacts/adapters/<adapter_id>.
ADAPTER_REL_ROOT = "artifacts/adapters"

# Keys several modules agree on (owners 3, 5 and 6): the isolation level of a trial in ``TrialRecord.environment``, and the
# ``BudgetCharge.detail`` key that marks a charge the proxy estimated instead of reading from the provider's ``usage``.
ENV_ISOLATION = "isolation"
DETAIL_USAGE_ESTIMATED = "usage_estimated"


# --------------------------------------------------------------------------------------------------
# exceptions
# --------------------------------------------------------------------------------------------------
class LabError(Exception):
    """Base class of every lab exception."""


class ContractError(LabError, ValueError):
    """A record failed validation (construction or load)."""


class SchemaVersionError(ContractError):
    """A persisted record has a schema_version this code does not read."""


class BudgetExhausted(LabError):
    """A budget axis is used up. Raised BEFORE the spend (``check``) or right after the charge that crossed the cap (``charge``)."""

    def __init__(self, axis: str, limit: float, used: float):
        super().__init__(f"budget exhausted: {axis} used {used:g} of {limit:g}")
        self.axis, self.limit, self.used = axis, limit, used


class AccessDenied(LabError):
    """A split, assignment or file was requested by a caller that may not see it (scorer access policy, redaction)."""


class FinalComparisonRefused(LabError):
    """The final comparison was requested a second time for the same pack version without a recorded reason."""


class GateNotRecorded(LabError):
    """A prerequisite gate (the terms gate) has no outcome in the ledger; the outer loop must not propose."""


class LedgerError(LabError):
    """An event kind or payload is not a lab contract, or the ledger could not be opened for writing."""


class GpuError(LabError):
    """Base class of GPU provider and rental failures."""


class ProviderUnavailable(GpuError):
    """A provider read or delete had an UNKNOWN outcome. Never treat it as 'absent' or 'empty'."""


class CreateRejected(GpuError):
    """The provider definitely refused the create. Nothing exists."""


class CreateAmbiguous(GpuError):
    """The create may have been accepted (timeout after send). The caller must recover before creating again."""


class OwnershipError(GpuError):
    """A delete or adopt targeted an instance that is not recorded as ours. Nothing was sent."""


class ScorerError(LabError):
    """Scoring could not produce a valid ScoreRecord (missing responses, server failure, bad data)."""


class TrainerError(LabError):
    """The trainer or its GPU host failed in a way the job result cannot express."""


class RuntimeFailure(LabError):
    """The researcher runtime failed to start, or died in a way that is not a normal exit."""


class ProposalInvalid(LabError):
    """The proposer's reply did not parse, or its edits do not apply or are not allowed. No GPU is spent on it."""


# --------------------------------------------------------------------------------------------------
# validators (used by __post_init__ and by the reader)
# --------------------------------------------------------------------------------------------------
_IDENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_HEX = re.compile(r"^[0-9a-f]+$")
E = TypeVar("E", bound=enum.Enum)


def _num(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ContractError(f"{what}: expected a number, got {type(v).__name__}")
    try:
        f = float(v)
    except OverflowError:
        raise ContractError(f"{what}: number out of range") from None
    if not math.isfinite(f):
        raise ContractError(f"{what}: must be finite, got {v!r}")
    return f


def _nonneg(v: Any, what: str) -> float:
    f = _num(v, what)
    if f < 0:
        raise ContractError(f"{what}: must be >= 0, got {f:g}")
    return f


def _pos(v: Any, what: str) -> float:
    f = _num(v, what)
    if f <= 0:
        raise ContractError(f"{what}: must be > 0, got {f:g}")
    return f


def _frac(v: Any, what: str) -> float:
    f = _num(v, what)
    if not 0.0 <= f <= 1.0:
        raise ContractError(f"{what}: must be within [0, 1], got {f:g}")
    return f


def _int(v: Any, what: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ContractError(f"{what}: expected an integer, got {type(v).__name__}")
    if minimum is not None and v < minimum:
        raise ContractError(f"{what}: must be >= {minimum}, got {v}")
    return v


def _bool(v: Any, what: str) -> bool:
    if not isinstance(v, bool):
        raise ContractError(f"{what}: expected a boolean, got {type(v).__name__}")
    return v


def _text(v: Any, what: str, *, empty: bool = False, max_bytes: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(v, str):
        raise ContractError(f"{what}: expected a string, got {type(v).__name__}")
    if "\x00" in v:
        raise ContractError(f"{what}: contains a NUL character")
    if not empty and not v.strip():
        raise ContractError(f"{what}: must not be empty")
    try:
        n = len(v.encode("utf-8"))
    except UnicodeEncodeError:
        raise ContractError(f"{what}: not encodable as UTF-8") from None
    if n > max_bytes:
        raise ContractError(f"{what}: {n} bytes exceeds the {max_bytes}-byte limit")
    return v


def _ident(v: Any, what: str) -> str:
    s = _text(v, what, max_bytes=64)
    if not _IDENT.match(s):
        raise ContractError(f"{what}: {s!r} is not a safe identifier ([A-Za-z0-9][A-Za-z0-9_.-]{{0,63}})")
    return s


def _hex(v: Any, what: str, n: int) -> str:
    s = _text(v, what, max_bytes=n)
    if len(s) != n or not _HEX.match(s):
        raise ContractError(f"{what}: expected {n} lowercase hex characters")
    return s


def _sha256(v: Any, what: str) -> str:
    return _hex(v, what, 64)


def _commit(v: Any, what: str) -> str:
    return _hex(v, what, 40)


def _utc(v: Any, what: str) -> str:
    s = _text(v, what, max_bytes=64)
    try:
        t = datetime.fromisoformat(s)
    except ValueError:
        raise ContractError(f"{what}: {s!r} is not an ISO 8601 datetime") from None
    if t.tzinfo is None or t.utcoffset() is None:
        raise ContractError(f"{what}: {s!r} has no timezone")
    return s


def _relpath(v: Any, what: str) -> str:
    s = _text(v, what, max_bytes=1024)
    p = PurePosixPath(s)
    if "\\" in s or p.is_absolute() or str(p) != s or any(part in ("", ".", "..", ".git") for part in p.parts):
        raise ContractError(f"{what}: {s!r} must be a normalised relative POSIX path with no '..' and no '.git'")
    return s


def _enum(v: Any, cls: type[E], what: str) -> E:
    if isinstance(v, cls):
        return v
    if isinstance(v, str):
        try:
            return cls(v)
        except ValueError:
            pass
    raise ContractError(f"{what}: {v!r} is not one of {[m.value for m in cls]}")


def _obj(v: Any, cls: type, what: str) -> Any:
    if not isinstance(v, cls):
        raise ContractError(f"{what}: expected {cls.__name__}, got {type(v).__name__}")
    return v


def _opt(v: Any, check: Callable[[Any, str], Any], what: str) -> Any:
    return None if v is None else check(v, what)


def _seq(v: Any, what: str) -> tuple:
    if isinstance(v, (str, bytes, Mapping)) or not isinstance(v, (list, tuple)):
        raise ContractError(f"{what}: expected a list, got {type(v).__name__}")
    return tuple(v)


def _tuple_of(v: Any, what: str, item: Callable[[Any, str], Any], *, nonempty: bool = False, unique: bool = False) -> tuple:
    items = tuple(item(x, f"{what}[{i}]") for i, x in enumerate(_seq(v, what)))
    if nonempty and not items:
        raise ContractError(f"{what}: must not be empty")
    if unique and len(set(items)) != len(items):
        raise ContractError(f"{what}: entries must be unique")
    return items


def _str_map(v: Any, what: str, *, value: Callable[[Any, str], Any] = _text) -> dict:
    if not isinstance(v, Mapping):
        raise ContractError(f"{what}: expected an object, got {type(v).__name__}")
    out = {}
    for k, x in v.items():
        key = _text(k, f"{what} key", max_bytes=200)
        out[key] = value(x, f"{what}[{key!r}]")
    return out


def _scalar(v: Any, what: str) -> Any:
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return _num(v, what)
    return _text(v, what, empty=True)


def _set(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


class _Reader:
    """Strict reader of a JSON object: missing and unknown fields are errors, ``schema_version`` is checked first."""

    __slots__ = ("what", "_d", "_used")

    def __init__(self, data: Any, what: str, *, versioned: bool = True):
        if not isinstance(data, Mapping):
            raise ContractError(f"{what}: expected a JSON object, got {type(data).__name__}")
        self.what, self._d, self._used = what, dict(data), set()
        if versioned:
            v = self.raw("schema_version")
            if isinstance(v, bool) or not isinstance(v, int):
                raise ContractError(f"{what}: schema_version must be an integer")
            if v != SCHEMA_VERSION:
                raise SchemaVersionError(f"{what}: schema_version {v} is not supported (this code reads {SCHEMA_VERSION})")

    def raw(self, key: str) -> Any:
        if key not in self._d:
            raise ContractError(f"{self.what}: missing field {key!r}")
        self._used.add(key)
        return self._d[key]

    def obj(self, key: str, fn: Callable[[Any], Any]) -> Any:
        return fn(self.raw(key))

    def opt(self, key: str, fn: Callable[[Any], Any]) -> Any:
        v = self.raw(key)
        return None if v is None else fn(v)

    def seq(self, key: str, fn: Callable[[Any], Any]) -> tuple:
        return tuple(fn(x) for x in _seq(self.raw(key), f"{self.what}.{key}"))

    def done(self) -> None:
        extra = sorted(set(self._d) - self._used)
        if extra:
            raise ContractError(f"{self.what}: unknown field(s) {extra}")


def _jsonable(x: Any) -> Any:
    if isinstance(x, enum.Enum):
        return x.value
    if isinstance(x, (tuple, list)):
        return [_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if hasattr(x, "to_dict"):
        return x.to_dict()
    return x


# --------------------------------------------------------------------------------------------------
# strict JSON helpers shared by every module
# --------------------------------------------------------------------------------------------------
def canonical_json(obj: Any) -> str:
    """Sorted keys, no spaces, UTF-8 characters kept, NaN and Infinity refused. The input to every record hash."""
    return json.dumps(_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_sha256(record: Any) -> str:
    """sha256 of the canonical JSON of a record's ``to_dict()``. The rule hash recorded with every decision is this."""
    return sha256_text(canonical_json(record))


def _reject_constant(name: str) -> Any:
    raise ContractError(f"non-finite JSON constant {name}")


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise ContractError(f"duplicate JSON key {k!r}")
        out[k] = v
    return out


def loads_strict(text: str | bytes) -> Any:
    """``json.loads`` that refuses NaN, Infinity and duplicate keys."""
    try:
        return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_no_duplicates)
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON: {exc}") from None


def assert_ledger_safe(data: Any, *, _path: str = "data") -> None:
    """Raise ContractError if a ledger payload holds a non-finite number or a string the ledger would truncate."""
    if isinstance(data, str):
        if len(data.encode("utf-8", errors="replace")) >= LEDGER_STRING_CAP_BYTES:
            raise ContractError(f"{_path}: string of {len(data)} characters would be truncated by the ledger; store it as a file")
    elif isinstance(data, float):
        if not math.isfinite(data):
            raise ContractError(f"{_path}: non-finite number")
    elif isinstance(data, Mapping):
        for k, v in data.items():
            assert_ledger_safe(k, _path=f"{_path} key")
            assert_ledger_safe(v, _path=f"{_path}.{k}")
    elif isinstance(data, (list, tuple)):
        for i, v in enumerate(data):
            assert_ledger_safe(v, _path=f"{_path}[{i}]")


def dumps_record(record: Any) -> str:
    """Human-readable, deterministic JSON text of a record (sorted keys, 2-space indent, trailing newline)."""
    return json.dumps(_jsonable(record), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_record(path: str | os.PathLike[str], record: Any) -> str:
    """Atomically write a record (temp file + ``os.replace``). Returns the file's sha256 (the hash of the bytes written)."""
    text = dumps_record(record)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return sha256_text(text)


def read_record(path: str | os.PathLike[str], cls: type) -> Any:
    """Load a record file with ``cls.from_dict`` (strict)."""
    return cls.from_dict(loads_strict(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------------------------------
# enums
# --------------------------------------------------------------------------------------------------
class _StrEnum(str, enum.Enum):
    """A str-valued enum whose ``str()`` and ``format()`` are the value on every Python from 3.11."""

    def __str__(self) -> str:
        return str(self.value)

    def __format__(self, format_spec: str) -> str:
        return format(str(self.value), format_spec)


class Arm(_StrEnum):
    """The three arms of the final claim. Arm 1 has no trial and no pack: it is the scorer with no adapter."""

    STOCK = "stock"
    V0 = "v0"
    EVOLVED = "evolved"

    @property
    def number(self) -> int:
        return {"stock": 1, "v0": 2, "evolved": 3}[self.value]


class Split(_StrEnum):
    DEV = "dev"
    SELECTION = "selection"
    TEST = "test"


# Metric names a ScoreRecord on each split may carry. The headline is the first entry.
METRICS_BY_SPLIT: dict = {
    Split.DEV: (METRIC_PROMPT_ALL, METRIC_PROMPT_FRACTION),
    Split.SELECTION: (METRIC_PROMPT_ALL, METRIC_PROMPT_FRACTION),
    Split.TEST: (METRIC_PROMPT_STRICT, METRIC_PROMPT_LOOSE, METRIC_INSTRUCTION_STRICT, METRIC_INSTRUCTION_LOOSE),
}


class Phase(_StrEnum):
    """SEARCH: trials run while evolving (adaptive search trace). FINAL: budget-matched trials of the frozen packs."""

    SEARCH = "search"
    FINAL = "final"


class AssignmentRole(_StrEnum):
    EVOLUTION = "evolution"
    HELDOUT = "heldout"


class Verifier(_StrEnum):
    """Which implementation of the IFBench constraint checks produced a score. Test = upstream IFBench; dev/selection = IFEvalG."""

    IFBENCH_UPSTREAM = "ifbench_upstream"
    IFEVALG = "ifevalg"


class Resource(_StrEnum):
    """What a charge is booked against. Axes of BudgetSpec: tokens; gpu_minutes; dollars = researcher_usd + gpu_usd; wall.

    ``WALL_SECONDS`` is derived from the injected clock since trial start (``check`` only); nobody calls ``charge(WALL_SECONDS)``.
    """

    RESEARCHER_TOKENS = "researcher_tokens"
    GPU_MINUTES = "gpu_minutes"
    RESEARCHER_USD = "researcher_usd"
    GPU_USD = "gpu_usd"
    WALL_SECONDS = "wall_seconds"


# Resource -> BudgetSpec axis name (the only place this mapping is written down).
AXIS_OF: dict = {
    Resource.RESEARCHER_TOKENS: "researcher_tokens",
    Resource.GPU_MINUTES: "gpu_minutes",
    Resource.RESEARCHER_USD: "dollars",
    Resource.GPU_USD: "dollars",
    Resource.WALL_SECONDS: "wall_seconds",
}


class TrialStatus(_StrEnum):
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_SUBMISSION = "no_submission"
    FAILED = "failed"
    ABORTED = "aborted"
    CONTAMINATED = "contaminated"  # the training data matched selection, held-out or test data; counts as a failed trial


class ExitReason(_StrEnum):
    FINISHED = "finished"
    STOPPED_BUDGET = "stopped_budget"
    STOPPED_WALL = "stopped_wall"
    STOPPED_CONTAMINATED = "stopped_contaminated"  # the contamination scan hit; the runner stopped the runtime
    CRASHED = "crashed"


class JobStatus(_StrEnum):
    OK = "ok"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"


class Outcome(_StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INVALID = "invalid"


class DecisionStage(_StrEnum):
    """SCREEN: candidates of one round ranked by delta with no threshold. CONFIRM: the best one is tested against the margin."""

    SCREEN = "screen"
    CONFIRM = "confirm"


class FailedTrialPolicy(_StrEnum):
    """What a scheduled search trial that failed, had no submission, was aborted or contaminated counts as (BUDGET_EXHAUSTED with a
    scored adapter always counts normally). There is no 'exclude' policy: judging a flaky pack on its lucky survivors is a false accept."""

    INVALIDATES_CANDIDATE = "invalidates_candidate"
    SCORED_ZERO = "scored_zero"


class ComparisonMethod(_StrEnum):
    BLOCKED_T = "blocked_t"  # two or more held-out assignments: t interval over per-assignment differences
    WELCH = "welch"  # one held-out assignment: Welch interval over trials


class MixedRowPolicy(_StrEnum):
    """What the manifest does with rows whose constraints span evolution and held-out families. Only DROPPED exists: a held-out-only
    policy would put evolution families inside a held-out assignment and the manifest invariant could not hold."""

    DROPPED = "dropped"


class ItemMode(_StrEnum):
    """How selection and held-out items were made. FRESH_INSTANTIATION: new constraint instantiations from the IFEvalG generators over
    base prompts (the verbatim strings are not in the public parquet). SALTED_DRAW: rows drawn from the public dataset with the secret
    salt; the residual risk that a researcher who retrieves the dataset trains on them is then stated in the report."""

    FRESH_INSTANTIATION = "fresh_instantiation"
    SALTED_DRAW = "salted_draw"


class Lever(_StrEnum):
    SEARCH_POLICY = "search_policy"
    CONTEXT_NOTES = "context_notes"
    BUDGET_USE = "budget_use"
    COORDINATION = "coordination"
    VERIFICATION_DISCIPLINE = "verification_discipline"
    ANTI_OVERFIT = "anti_overfit"
    TOOLING = "tooling"
    PROMPT_TEXT = "prompt_text"
    OTHER = "other"


class EditOp(_StrEnum):
    REPLACE_FILE = "replace_file"
    SEARCH_REPLACE = "search_replace"


class Gate(_StrEnum):
    TERMS_10_1_IX = "terms_10_1_ix"


class GateOutcome(_StrEnum):
    CLARIFIED = "clarified"
    PROCEED_WITH_RISK = "proceed_with_risk"


def utc_iso(epoch: float) -> str:
    """The one way to turn a clock reading into a record time: UTC, seconds precision, always ``+00:00``."""
    return datetime.fromtimestamp(_nonneg(epoch, "epoch"), tz=timezone.utc).replace(microsecond=0).isoformat()


def owner_tag_of_key(public_key: str) -> str:
    """Ownership tag of a rental: 16 hex characters of sha256 over the key's type and blob (the comment is ignored)."""
    parts = _text(public_key, "public_key", max_bytes=2000).split()
    if len(parts) < 2:
        raise ContractError("public_key: expected '<type> <base64> [comment]'")
    return sha256_text(parts[0] + " " + parts[1])[:16]


def scorer_version_of(verifier_pins: Mapping[str, str], metric_names: Sequence[str], module_version: int) -> str:
    """``scorer_version`` is this value: a digest of the verifier pins, the metric names and the scorer module's own version."""
    version = _int(module_version, "module_version", minimum=1)
    pins = _str_map(verifier_pins, "verifier_pins")
    digest = sha256_text(canonical_json({"pins": pins, "metrics": sorted(metric_names), "module": version}))
    return f"sv{version}-{digest[:16]}"


# --------------------------------------------------------------------------------------------------
# small value objects (embedded; versioned by their container)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FileRef:
    """A file inside the lab root: relative POSIX path, sha256 of its bytes, size."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _set(self, "path", _relpath(self.path, "FileRef.path"))
        _sha256(self.sha256, "FileRef.sha256")
        _int(self.size_bytes, "FileRef.size_bytes", minimum=0)

    def to_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, d: Any) -> "FileRef":
        r = _Reader(d, "FileRef", versioned=False)
        out = cls(r.raw("path"), r.raw("sha256"), r.raw("size_bytes"))
        r.done()
        return out


def _reject_links(rootp: Path, p: Path) -> os.stat_result:
    """lstat every component below ``rootp``: refuse symbolic links, non-regular files and hard-linked files."""
    rel = p.relative_to(rootp)
    if not rel.parts:
        raise ContractError(f"{p} is the lab root, not a file")
    cur, st = rootp, None
    for part in rel.parts:
        cur = cur / part
        st = os.lstat(cur)
        if stat.S_ISLNK(st.st_mode):
            raise ContractError(f"{cur}: symbolic links are refused")
    if not stat.S_ISREG(st.st_mode):
        raise ContractError(f"{p}: not a regular file")
    if st.st_nlink > 1:
        raise ContractError(f"{p}: hard-linked files are refused")
    return st


def file_ref(root: str | os.PathLike[str], path: str | os.PathLike[str]) -> FileRef:
    """FileRef for an existing regular file under ``root`` (computes sha256 and size). Symbolic links and hard links are refused."""
    rootp, p = Path(os.path.abspath(root)), Path(os.path.abspath(path))
    try:
        rel = p.relative_to(rootp).as_posix()
    except ValueError:
        raise ContractError(f"file_ref: {p} is not under {rootp}") from None
    st = _reject_links(rootp, p)
    return FileRef(rel, sha256_file(p), st.st_size)


def read_regular_bytes(root: str | os.PathLike[str], relpath: str, *, max_bytes: int) -> bytes:
    """Read a lab-root-relative regular file without following links, refusing files above ``max_bytes``.

    Use this for every file a researcher or a rental could have written (harvested notes, traces, recipe tarballs, frozen files).
    """
    rel = _relpath(relpath, "relpath")
    cap = _int(max_bytes, "max_bytes", minimum=0)
    rootp = Path(os.path.abspath(root))
    p = rootp / rel
    st = _reject_links(rootp, p)
    if st.st_size > cap:
        raise ContractError(f"{rel}: {st.st_size} bytes exceeds the {cap}-byte limit")
    fd = os.open(p, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as fh:
        data = fh.read(cap + 1)
    if len(data) > cap:
        raise ContractError(f"{rel}: grew past the {cap}-byte limit while reading")
    return data


@dataclass(frozen=True)
class PackRef:
    """A researcher pack version: the pack git repository (lab-root-relative, e.g. ``packs/pack.git``) and one commit in it."""

    repo_path: str
    commit: str

    def __post_init__(self) -> None:
        _relpath(self.repo_path, "PackRef.repo_path")
        _commit(self.commit, "PackRef.commit")

    def resolve(self, root: str | os.PathLike[str]) -> Path:
        return Path(root) / self.repo_path

    def to_dict(self) -> dict:
        return {"repo_path": self.repo_path, "commit": self.commit}

    @classmethod
    def from_dict(cls, d: Any) -> "PackRef":
        r = _Reader(d, "PackRef", versioned=False)
        out = cls(r.raw("repo_path"), r.raw("commit"))
        r.done()
        return out


@dataclass(frozen=True)
class BaseModelRef:
    """The starting checkpoint. ``revision`` is the Hugging Face commit id, which addresses the content."""

    repo_id: str
    revision: str

    def __post_init__(self) -> None:
        _text(self.repo_id, "BaseModelRef.repo_id", max_bytes=200)
        _commit(self.revision, "BaseModelRef.revision")

    def to_dict(self) -> dict:
        return {"repo_id": self.repo_id, "revision": self.revision}

    @classmethod
    def from_dict(cls, d: Any) -> "BaseModelRef":
        r = _Reader(d, "BaseModelRef", versioned=False)
        out = cls(r.raw("repo_id"), r.raw("revision"))
        r.done()
        return out


@dataclass(frozen=True)
class LedgerAnchor:
    """``Ledger.head_anchor()``: sequence number and the first 16 hex characters of the head hash."""

    seq: int
    hash16: str

    def __post_init__(self) -> None:
        _int(self.seq, "LedgerAnchor.seq", minimum=0)
        _hex(self.hash16, "LedgerAnchor.hash16", 16)

    def to_dict(self) -> dict:
        return {"seq": self.seq, "hash16": self.hash16}

    @classmethod
    def from_dict(cls, d: Any) -> "LedgerAnchor":
        r = _Reader(d, "LedgerAnchor", versioned=False)
        out = cls(r.raw("seq"), r.raw("hash16"))
        r.done()
        return out


@dataclass(frozen=True)
class GenerationParams:
    """Decoding settings, recorded with every score. Glimmer defaults come from the model card / generation_config.json."""

    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    reasoning_strength: str
    system_prompt: str

    def __post_init__(self) -> None:
        _set(self, "temperature", _nonneg(self.temperature, "GenerationParams.temperature"))
        _set(self, "top_p", _pos(self.top_p, "GenerationParams.top_p"))
        if self.top_p > 1.0:
            raise ContractError("GenerationParams.top_p: must be <= 1")
        _int(self.top_k, "GenerationParams.top_k", minimum=0)
        _int(self.max_tokens, "GenerationParams.max_tokens", minimum=1)
        if self.reasoning_strength not in REASONING_STRENGTHS:
            raise ContractError(f"GenerationParams.reasoning_strength: {self.reasoning_strength!r} not in {REASONING_STRENGTHS}")
        _text(self.system_prompt, "GenerationParams.system_prompt")

    @classmethod
    def recommended(cls, max_tokens: int) -> "GenerationParams":
        """Meta's recommended sampling for Glimmer (temperature 1.0, top_p 0.95, top_k 64) with an explicit system message."""
        return cls(1.0, 0.95, 64, max_tokens, "high", "You are a helpful AI assistant.")

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "reasoning_strength": self.reasoning_strength,
            "system_prompt": self.system_prompt,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "GenerationParams":
        r = _Reader(d, "GenerationParams", versioned=False)
        out = cls(r.raw("temperature"), r.raw("top_p"), r.raw("top_k"), r.raw("max_tokens"), r.raw("reasoning_strength"), r.raw("system_prompt"))
        r.done()
        return out


# --------------------------------------------------------------------------------------------------
# budget
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BudgetSpec:
    """The fixed allowance of one trial (``budget.json``). ``dollars`` caps researcher inference plus GPU rental together."""

    researcher_tokens: int
    gpu_minutes: float
    dollars: float
    wall_seconds: float
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _int(self.researcher_tokens, "BudgetSpec.researcher_tokens", minimum=0)
        _set(self, "gpu_minutes", _nonneg(self.gpu_minutes, "BudgetSpec.gpu_minutes"))
        _set(self, "dollars", _nonneg(self.dollars, "BudgetSpec.dollars"))
        _set(self, "wall_seconds", _nonneg(self.wall_seconds, "BudgetSpec.wall_seconds"))

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "researcher_tokens": self.researcher_tokens,
            "gpu_minutes": self.gpu_minutes,
            "dollars": self.dollars,
            "wall_seconds": self.wall_seconds,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "BudgetSpec":
        r = _Reader(d, "BudgetSpec")
        out = cls(r.raw("researcher_tokens"), r.raw("gpu_minutes"), r.raw("dollars"), r.raw("wall_seconds"))
        r.done()
        return out


def _check_version(rec: Any) -> None:
    if isinstance(rec.schema_version, bool) or rec.schema_version != SCHEMA_VERSION:
        raise SchemaVersionError(f"{type(rec).__name__}: schema_version {rec.schema_version!r} is not {SCHEMA_VERSION}")


@dataclass(frozen=True)
class BudgetUsage:
    """What a trial has used. Reported separately for researcher inference and GPU (the report keeps the cost centres apart)."""

    researcher_tokens: int
    gpu_minutes: float
    researcher_usd: float
    gpu_usd: float
    wall_seconds: float

    def __post_init__(self) -> None:
        _int(self.researcher_tokens, "BudgetUsage.researcher_tokens", minimum=0)
        for name in ("gpu_minutes", "researcher_usd", "gpu_usd", "wall_seconds"):
            _set(self, name, _nonneg(getattr(self, name), f"BudgetUsage.{name}"))

    @property
    def dollars(self) -> float:
        return self.researcher_usd + self.gpu_usd

    def exhausted_axes(self, spec: BudgetSpec) -> tuple[str, ...]:
        """Axes of ``spec`` that are used up (usage >= cap; a cap of 0 is exhausted at once). Names are BudgetSpec field names."""
        pairs = (
            ("researcher_tokens", self.researcher_tokens, spec.researcher_tokens),
            ("gpu_minutes", self.gpu_minutes, spec.gpu_minutes),
            ("dollars", self.dollars, spec.dollars),
            ("wall_seconds", self.wall_seconds, spec.wall_seconds),
        )
        return tuple(name for name, used, cap in pairs if used >= cap)

    def to_dict(self) -> dict:
        return {
            "researcher_tokens": self.researcher_tokens,
            "gpu_minutes": self.gpu_minutes,
            "researcher_usd": self.researcher_usd,
            "gpu_usd": self.gpu_usd,
            "wall_seconds": self.wall_seconds,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "BudgetUsage":
        r = _Reader(d, "BudgetUsage", versioned=False)
        out = cls(r.raw("researcher_tokens"), r.raw("gpu_minutes"), r.raw("researcher_usd"), r.raw("gpu_usd"), r.raw("wall_seconds"))
        r.done()
        return out


@dataclass(frozen=True)
class LlmUsage:
    """Token usage of one LLM call, as the provider reports it (or as the proxy estimates it, then ``detail`` says so)."""

    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int

    def __post_init__(self) -> None:
        _int(self.prompt_tokens, "LlmUsage.prompt_tokens", minimum=0)
        _int(self.completion_tokens, "LlmUsage.completion_tokens", minimum=0)
        _int(self.cached_tokens, "LlmUsage.cached_tokens", minimum=0)
        if self.cached_tokens > self.prompt_tokens:
            raise ContractError("LlmUsage: cached_tokens cannot exceed prompt_tokens")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens, "cached_tokens": self.cached_tokens}

    @classmethod
    def from_dict(cls, d: Any) -> "LlmUsage":
        r = _Reader(d, "LlmUsage", versioned=False)
        out = cls(r.raw("prompt_tokens"), r.raw("completion_tokens"), r.raw("cached_tokens"))
        r.done()
        return out


@dataclass(frozen=True)
class CostBreakdown:
    """The four cost centres of the outer search, never mixed (outer proposals, candidate trials split in two, scoring)."""

    outer_propose_usd: float
    candidate_researcher_usd: float
    candidate_gpu_usd: float
    scorer_usd: float

    def __post_init__(self) -> None:
        for name in ("outer_propose_usd", "candidate_researcher_usd", "candidate_gpu_usd", "scorer_usd"):
            _set(self, name, _nonneg(getattr(self, name), f"CostBreakdown.{name}"))

    @property
    def total_usd(self) -> float:
        return self.outer_propose_usd + self.candidate_researcher_usd + self.candidate_gpu_usd + self.scorer_usd

    def to_dict(self) -> dict:
        return {
            "outer_propose_usd": self.outer_propose_usd,
            "candidate_researcher_usd": self.candidate_researcher_usd,
            "candidate_gpu_usd": self.candidate_gpu_usd,
            "scorer_usd": self.scorer_usd,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "CostBreakdown":
        r = _Reader(d, "CostBreakdown", versioned=False)
        out = cls(r.raw("outer_propose_usd"), r.raw("candidate_researcher_usd"), r.raw("candidate_gpu_usd"), r.raw("scorer_usd"))
        r.done()
        return out


@dataclass(frozen=True)
class BudgetCharge:
    """Payload of ``lab.budget_charged`` in a trial ledger. ``call_id`` makes a charge idempotent across restarts."""

    trial_id: str
    resource: Resource
    amount: float
    call_id: str
    detail: dict
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.trial_id, "BudgetCharge.trial_id")
        _set(self, "resource", _enum(self.resource, Resource, "BudgetCharge.resource"))
        _set(self, "amount", _nonneg(self.amount, "BudgetCharge.amount"))
        _text(self.call_id, "BudgetCharge.call_id", max_bytes=200)
        _set(self, "detail", _str_map(self.detail, "BudgetCharge.detail", value=_scalar))

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "trial_id": self.trial_id,
            "resource": self.resource.value,
            "amount": self.amount,
            "call_id": self.call_id,
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "BudgetCharge":
        r = _Reader(d, "BudgetCharge")
        out = cls(r.raw("trial_id"), r.raw("resource"), r.raw("amount"), r.raw("call_id"), r.raw("detail"))
        r.done()
        return out


@dataclass(frozen=True)
class BudgetExhaustedEvent:
    """Payload of ``lab.budget_exhausted``: the first time an axis was found used up, and what was done about it. ``lab_dollars`` is the
    lab-wide cap on researcher inference plus Wonder (``LabConfig.lab_budget_usd``), shared by every trial's meter."""

    trial_id: str
    axis: str
    limit: float
    used: float
    action: str
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.trial_id, "BudgetExhaustedEvent.trial_id")
        if self.axis not in ("researcher_tokens", "gpu_minutes", "dollars", "wall_seconds", "lab_dollars"):
            raise ContractError(f"BudgetExhaustedEvent.axis: {self.axis!r} is not a BudgetSpec axis or the lab-wide cap")
        _set(self, "limit", _nonneg(self.limit, "BudgetExhaustedEvent.limit"))
        _set(self, "used", _nonneg(self.used, "BudgetExhaustedEvent.used"))
        _text(self.action, "BudgetExhaustedEvent.action", max_bytes=200)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "trial_id": self.trial_id,
            "axis": self.axis,
            "limit": self.limit,
            "used": self.used,
            "action": self.action,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "BudgetExhaustedEvent":
        r = _Reader(d, "BudgetExhaustedEvent")
        out = cls(r.raw("trial_id"), r.raw("axis"), r.raw("limit"), r.raw("used"), r.raw("action"))
        r.done()
        return out


# --------------------------------------------------------------------------------------------------
# adapter
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AdapterFile:
    name: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _relpath(self.name, "AdapterFile.name")
        _sha256(self.sha256, "AdapterFile.sha256")
        _int(self.size_bytes, "AdapterFile.size_bytes", minimum=0)

    def to_dict(self) -> dict:
        return {"name": self.name, "sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, d: Any) -> "AdapterFile":
        r = _Reader(d, "AdapterFile", versioned=False)
        out = cls(r.raw("name"), r.raw("sha256"), r.raw("size_bytes"))
        r.done()
        return out


def canonical_listing_sha256(files: Sequence[AdapterFile]) -> str:
    """sha256 of the canonical listing: one line ``<sha256> <size> <name>\\n`` per file, sorted by name."""
    lines = "".join(f"{f.sha256} {f.size_bytes} {f.name}\n" for f in sorted(files, key=lambda f: f.name))
    return sha256_text(lines)


def adapter_rel_path(adapter_id: str) -> str:
    return f"{ADAPTER_REL_ROOT}/{_ident(adapter_id, 'adapter_id')}"


@dataclass(frozen=True)
class AdapterRef:
    """A trained adapter (``adapter.json`` beside the weights). ``path`` is lab-root-relative (``artifacts/adapters/<adapter_id>``),
    so a record stays valid when the lab root moves. ``sha256`` is the hash of the canonical file listing."""

    path: str
    files: tuple
    sha256: str
    size_bytes: int
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        parts = PurePosixPath(_relpath(self.path, "AdapterRef.path")).parts
        if len(parts) != 3 or "/".join(parts[:2]) != ADAPTER_REL_ROOT:
            raise ContractError(f"AdapterRef.path: {self.path!r} must be {ADAPTER_REL_ROOT}/<adapter_id>")
        _ident(parts[2], "AdapterRef.path adapter id")
        files = _tuple_of(self.files, "AdapterRef.files", lambda v, w: _obj(v, AdapterFile, w), nonempty=True)
        names = [f.name for f in files]
        if len(set(names)) != len(names):
            raise ContractError("AdapterRef.files: duplicate names")
        missing = [n for n in ADAPTER_FILES if n not in names]
        if missing:
            raise ContractError(f"AdapterRef.files: missing {missing}")
        _set(self, "files", tuple(sorted(files, key=lambda f: f.name)))
        _sha256(self.sha256, "AdapterRef.sha256")
        _int(self.size_bytes, "AdapterRef.size_bytes", minimum=0)
        if self.sha256 != canonical_listing_sha256(self.files):
            raise ContractError("AdapterRef.sha256 does not match the canonical listing of files")
        if self.size_bytes != sum(f.size_bytes for f in self.files):
            raise ContractError("AdapterRef.size_bytes does not equal the sum of the file sizes")

    @property
    def adapter_id(self) -> str:
        return PurePosixPath(self.path).name

    def resolve(self, root: str | os.PathLike[str]) -> Path:
        return Path(root) / self.path

    @classmethod
    def from_files(cls, adapter_id: str, files: Sequence[AdapterFile]) -> "AdapterRef":
        fs = tuple(sorted(files, key=lambda f: f.name))
        return cls(adapter_rel_path(adapter_id), fs, canonical_listing_sha256(fs), sum(f.size_bytes for f in fs))

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "files": [f.to_dict() for f in self.files],
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "AdapterRef":
        r = _Reader(d, "AdapterRef")
        out = cls(r.raw("path"), r.seq("files", AdapterFile.from_dict), r.raw("sha256"), r.raw("size_bytes"))
        r.done()
        return out


def adapter_ref_from_dir(path: str | os.PathLike[str], *, adapter_id: str) -> AdapterRef:
    """Hash the behaviour-defining adapter files (``ADAPTER_FILES``) in directory ``path``; the ref's own path is the lab-relative one
    for ``adapter_id``. Only those files, by exact name, regular files only (no links). Raises ContractError if one is missing."""
    p = Path(path)
    files = []
    for name in ADAPTER_FILES:
        f = p / name
        st = os.lstat(f) if os.path.lexists(f) else None
        if st is None or not stat.S_ISREG(st.st_mode) or st.st_nlink > 1:
            raise ContractError(f"adapter directory {p} has no regular file {name}")
        files.append(AdapterFile(name, sha256_file(f), st.st_size))
    return AdapterRef.from_files(adapter_id, files)


def verify_adapter_dir(ref: AdapterRef, path: str | os.PathLike[str] | None = None, root: str | os.PathLike[str] | None = None) -> None:
    """Re-hash a directory and require it to equal ``ref`` (used after every copy of an adapter). Give ``path`` or the lab ``root``."""
    if path is None and root is None:
        raise ContractError("verify_adapter_dir: give a directory path or the lab root")
    where = Path(path) if path is not None else ref.resolve(root)
    actual = adapter_ref_from_dir(where, adapter_id=ref.adapter_id)
    if actual.sha256 != ref.sha256:
        raise ContractError(f"adapter at {where} has listing sha256 {actual.sha256}, expected {ref.sha256}")


# --------------------------------------------------------------------------------------------------
# packs, assignments
# --------------------------------------------------------------------------------------------------
def _revisions(v: Any, what: str, keys: Sequence[str], check: Callable[[Any, str], Any]) -> dict:
    m = _str_map(v, what)
    if set(m) != set(keys):
        raise ContractError(f"{what}: keys must be exactly {sorted(keys)}, got {sorted(m)}")
    for k in keys:
        check(m[k], f"{what}[{k!r}]")
    return m


@dataclass(frozen=True)
class Assignment:
    """One assignment: a pool of rows whose constraints all lie in ``families`` (the families of the base pool), split into train, dev
    and selection files. ``data_revisions`` keys are exactly ``DATA_REVISION_KEYS`` (sha256 of each file)."""

    assignment_id: str
    role: AssignmentRole
    families: tuple
    data_revisions: dict

    def __post_init__(self) -> None:
        _ident(self.assignment_id, "Assignment.assignment_id")
        if self.assignment_id == TEST_ASSIGNMENT_ID:
            raise ContractError(f"Assignment.assignment_id: {TEST_ASSIGNMENT_ID!r} is reserved for the test set")
        _set(self, "role", _enum(self.role, AssignmentRole, "Assignment.role"))
        _set(self, "families", _tuple_of(self.families, "Assignment.families", lambda v, w: _ident(v, w), nonempty=True, unique=True))
        _set(self, "data_revisions", _revisions(self.data_revisions, "Assignment.data_revisions", DATA_REVISION_KEYS, _sha256))

    def to_dict(self) -> dict:
        return {
            "assignment_id": self.assignment_id,
            "role": self.role.value,
            "families": list(self.families),
            "data_revisions": dict(self.data_revisions),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "Assignment":
        r = _Reader(d, "Assignment", versioned=False)
        out = cls(r.raw("assignment_id"), r.raw("role"), r.raw("families"), r.raw("data_revisions"))
        r.done()
        return out


@dataclass(frozen=True)
class AssignmentManifest:
    """Frozen before any candidate runs: assignment ids, data revisions, and the evolution / held-out family partition.

    ``source_revisions`` keys are exactly ``SOURCE_REVISION_KEYS``; the split salt itself is secret and never recorded, only its hash.
    """

    assignments: tuple
    evolution_families: tuple
    heldout_families: tuple
    mixed_row_policy: MixedRowPolicy
    item_mode: ItemMode
    source_revisions: dict
    created_utc: str
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        items = _tuple_of(self.assignments, "AssignmentManifest.assignments", lambda v, w: _obj(v, Assignment, w), nonempty=True)
        ids = [a.assignment_id for a in items]
        if len(set(ids)) != len(ids):
            raise ContractError("AssignmentManifest.assignments: duplicate assignment_id")
        _set(self, "assignments", items)
        ev = _tuple_of(self.evolution_families, "AssignmentManifest.evolution_families", lambda v, w: _ident(v, w), nonempty=True, unique=True)
        ho = _tuple_of(self.heldout_families, "AssignmentManifest.heldout_families", lambda v, w: _ident(v, w), nonempty=True, unique=True)
        if set(ev) & set(ho):
            raise ContractError("AssignmentManifest: evolution and held-out families overlap")
        _set(self, "evolution_families", ev)
        _set(self, "heldout_families", ho)
        for a in items:
            allowed = set(ev) if a.role is AssignmentRole.EVOLUTION else set(ho)
            if not set(a.families) <= allowed:
                raise ContractError(f"assignment {a.assignment_id}: families {sorted(set(a.families) - allowed)} are outside its role's partition")
        if not any(a.role is AssignmentRole.EVOLUTION for a in items) or not any(a.role is AssignmentRole.HELDOUT for a in items):
            raise ContractError("AssignmentManifest: needs at least one evolution and one held-out assignment")
        _set(self, "mixed_row_policy", _enum(self.mixed_row_policy, MixedRowPolicy, "AssignmentManifest.mixed_row_policy"))
        _set(self, "item_mode", _enum(self.item_mode, ItemMode, "AssignmentManifest.item_mode"))
        src = _str_map(self.source_revisions, "AssignmentManifest.source_revisions")
        if set(src) != set(SOURCE_REVISION_KEYS):
            raise ContractError(f"AssignmentManifest.source_revisions: keys must be exactly {sorted(SOURCE_REVISION_KEYS)}")
        _sha256(src[SRC_IFRLVR_PARQUET_SHA256], "source_revisions.ifrlvr_parquet_sha256")
        _commit(src[SRC_IFBENCH_COMMIT], "source_revisions.ifbench_commit")
        _commit(src[SRC_OPEN_INSTRUCT_COMMIT], "source_revisions.open_instruct_commit")
        _sha256(src[SRC_SPLIT_SALT_SHA256], "source_revisions.split_salt_sha256")
        _set(self, "source_revisions", src)
        _utc(self.created_utc, "AssignmentManifest.created_utc")

    def get(self, assignment_id: str) -> Assignment:
        for a in self.assignments:
            if a.assignment_id == assignment_id:
                return a
        raise KeyError(assignment_id)

    def ids_with_role(self, role: AssignmentRole) -> tuple:
        return tuple(a.assignment_id for a in self.assignments if a.role is role)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "assignments": [a.to_dict() for a in self.assignments],
            "evolution_families": list(self.evolution_families),
            "heldout_families": list(self.heldout_families),
            "mixed_row_policy": self.mixed_row_policy.value,
            "item_mode": self.item_mode.value,
            "source_revisions": dict(self.source_revisions),
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "AssignmentManifest":
        r = _Reader(d, "AssignmentManifest")
        out = cls(
            r.seq("assignments", Assignment.from_dict),
            r.raw("evolution_families"),
            r.raw("heldout_families"),
            r.raw("mixed_row_policy"),
            r.raw("item_mode"),
            r.raw("source_revisions"),
            r.raw("created_utc"),
        )
        r.done()
        return out


# --------------------------------------------------------------------------------------------------
# trials
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TrialSpec:
    """One run of one pack version on one assignment, one seed, one budget, from stock Glimmer with a clean workspace.

    Search-phase trials run on EVOLUTION assignments and have ``final_request_id`` None; final-phase trials run on HELDOUT assignments,
    have arm v0 or evolved, and carry the id of the FinalRequest that covers them (``check_trial_against_frozen`` checks both).
    ``shape`` is the GPU shape every GPU job of the trial must use (TrainJob.shape equals it).
    """

    trial_id: str
    phase: Phase
    arm: Optional[Arm]
    pack_version: str
    pack: PackRef
    assignment_id: str
    seed: int
    budget: BudgetSpec
    shape: "Shape"
    base: BaseModelRef
    data_revisions: dict
    scorer_version: str
    score_params: GenerationParams
    score_seeds: tuple
    final_request_id: Optional[str]
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.trial_id, "TrialSpec.trial_id")
        _set(self, "phase", _enum(self.phase, Phase, "TrialSpec.phase"))
        _set(self, "arm", _opt(self.arm, lambda v, w: _enum(v, Arm, w), "TrialSpec.arm"))
        if self.phase is Phase.FINAL and self.arm not in (Arm.V0, Arm.EVOLVED):
            raise ContractError("TrialSpec: a final-phase trial must have arm v0 or evolved (arm 1 has no trial)")
        if self.phase is Phase.SEARCH and self.arm is not None:
            raise ContractError("TrialSpec: a search-phase trial has no arm")
        _ident(self.pack_version, "TrialSpec.pack_version")
        _obj(self.pack, PackRef, "TrialSpec.pack")
        _ident(self.assignment_id, "TrialSpec.assignment_id")
        if self.assignment_id == TEST_ASSIGNMENT_ID:
            raise ContractError("TrialSpec: a trial cannot run on the test set")
        _int(self.seed, "TrialSpec.seed", minimum=0)
        _obj(self.budget, BudgetSpec, "TrialSpec.budget")
        _obj(self.shape, Shape, "TrialSpec.shape")
        _obj(self.base, BaseModelRef, "TrialSpec.base")
        _set(self, "data_revisions", _revisions(self.data_revisions, "TrialSpec.data_revisions", DATA_REVISION_KEYS, _sha256))
        _text(self.scorer_version, "TrialSpec.scorer_version", max_bytes=200)
        _obj(self.score_params, GenerationParams, "TrialSpec.score_params")
        _set(self, "score_seeds", _tuple_of(self.score_seeds, "TrialSpec.score_seeds", lambda v, w: _int(v, w, minimum=0), nonempty=True, unique=True))
        _opt(self.final_request_id, _ident, "TrialSpec.final_request_id")
        if self.phase is Phase.FINAL and self.final_request_id is None:
            raise ContractError("TrialSpec: a final-phase trial needs final_request_id")
        if self.phase is Phase.SEARCH and self.final_request_id is not None:
            raise ContractError("TrialSpec: a search-phase trial has no final_request_id")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "trial_id": self.trial_id,
            "phase": self.phase.value,
            "arm": None if self.arm is None else self.arm.value,
            "pack_version": self.pack_version,
            "pack": self.pack.to_dict(),
            "assignment_id": self.assignment_id,
            "seed": self.seed,
            "budget": self.budget.to_dict(),
            "shape": self.shape.to_dict(),
            "base": self.base.to_dict(),
            "data_revisions": dict(self.data_revisions),
            "scorer_version": self.scorer_version,
            "score_params": self.score_params.to_dict(),
            "score_seeds": list(self.score_seeds),
            "final_request_id": self.final_request_id,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "TrialSpec":
        r = _Reader(d, "TrialSpec")
        out = cls(
            r.raw("trial_id"),
            r.raw("phase"),
            r.raw("arm"),
            r.raw("pack_version"),
            r.obj("pack", PackRef.from_dict),
            r.raw("assignment_id"),
            r.raw("seed"),
            r.obj("budget", BudgetSpec.from_dict),
            r.obj("shape", Shape.from_dict),
            r.obj("base", BaseModelRef.from_dict),
            r.raw("data_revisions"),
            r.raw("scorer_version"),
            r.obj("score_params", GenerationParams.from_dict),
            r.raw("score_seeds"),
            r.raw("final_request_id"),
        )
        r.done()
        return out


def comparability_key(spec: TrialSpec, metric: str) -> str:
    """Two selection scores of the same assignment may be compared only if this key is equal: scorer version, metric, decoding, score
    seeds, the selection file, the budget, the GPU shape and the base checkpoint."""
    return sha256_text(
        canonical_json(
            {
                "scorer_version": spec.scorer_version,
                "metric": _text(metric, "metric", max_bytes=100),
                "params": spec.score_params.to_dict(),
                "seeds": sorted(spec.score_seeds),
                "selection_sha256": spec.data_revisions[REV_SELECTION_SHA256],
                "budget": spec.budget.to_dict(),
                "shape": spec.shape.to_dict(),
                "base": spec.base.to_dict(),
            }
        )
    )


@dataclass(frozen=True)
class Submission:
    """What a researcher team hands in: the adapter id the trainer returned, the recipe that made it, the commit that records it.

    ``dev_score`` is written by the trial runner from the control-host dev ScoreRecord (headline mean, unrounded); a value the
    researcher wrote into submission.json is discarded.
    """

    adapter_id: str
    recipe_sha256: str
    commit: str
    dev_score: Optional[float]

    def __post_init__(self) -> None:
        _ident(self.adapter_id, "Submission.adapter_id")
        _sha256(self.recipe_sha256, "Submission.recipe_sha256")
        _commit(self.commit, "Submission.commit")
        _set(self, "dev_score", _opt(self.dev_score, _frac, "Submission.dev_score"))

    def to_dict(self) -> dict:
        return {"adapter_id": self.adapter_id, "recipe_sha256": self.recipe_sha256, "commit": self.commit, "dev_score": self.dev_score}

    @classmethod
    def from_dict(cls, d: Any) -> "Submission":
        r = _Reader(d, "Submission", versioned=False)
        out = cls(r.raw("adapter_id"), r.raw("recipe_sha256"), r.raw("commit"), r.raw("dev_score"))
        r.done()
        return out


@dataclass(frozen=True)
class TrialRecord:
    """What happened in one trial (payload of ``lab.trial_finished``; also ``trials/<id>/trial.json``).

    ``initial_context_sha256`` is None only when the trial died before its workspace was checked out (then the status is ABORTED or
    FAILED); a scored trial always has it. Status mapping from the runtime and trainer results: docs/harness-design.md, "Trial".
    """

    spec: TrialSpec
    status: TrialStatus
    started_utc: str
    finished_utc: str
    initial_context_sha256: Optional[str]
    budget_used: BudgetUsage
    submission: Optional[Submission]
    adapter: Optional[AdapterRef]
    dev_score_id: Optional[str]
    selection_score_id: Optional[str]
    trace: Optional[FileRef]
    notes: Optional[FileRef]
    trial_ledger_anchor: Optional[LedgerAnchor]
    environment: dict
    failure: Optional[str]
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _obj(self.spec, TrialSpec, "TrialRecord.spec")
        _set(self, "status", _enum(self.status, TrialStatus, "TrialRecord.status"))
        _utc(self.started_utc, "TrialRecord.started_utc")
        _utc(self.finished_utc, "TrialRecord.finished_utc")
        _opt(self.initial_context_sha256, _sha256, "TrialRecord.initial_context_sha256")
        _obj(self.budget_used, BudgetUsage, "TrialRecord.budget_used")
        _opt(self.submission, lambda v, w: _obj(v, Submission, w), "TrialRecord.submission")
        _opt(self.adapter, lambda v, w: _obj(v, AdapterRef, w), "TrialRecord.adapter")
        _opt(self.dev_score_id, _ident, "TrialRecord.dev_score_id")
        _opt(self.selection_score_id, _ident, "TrialRecord.selection_score_id")
        _opt(self.trace, lambda v, w: _obj(v, FileRef, w), "TrialRecord.trace")
        _opt(self.notes, lambda v, w: _obj(v, FileRef, w), "TrialRecord.notes")
        _opt(self.trial_ledger_anchor, lambda v, w: _obj(v, LedgerAnchor, w), "TrialRecord.trial_ledger_anchor")
        _set(self, "environment", _str_map(self.environment, "TrialRecord.environment", value=lambda v, w: _text(v, w, empty=True, max_bytes=500)))
        _set(self, "failure", _opt(self.failure, _text, "TrialRecord.failure"))
        scored = self.status in (TrialStatus.COMPLETED, TrialStatus.BUDGET_EXHAUSTED)
        if scored and (self.adapter is None or self.selection_score_id is None or self.submission is None):
            raise ContractError(f"TrialRecord: status {self.status.value} requires submission, adapter and selection_score_id")
        if scored and self.initial_context_sha256 is None:
            raise ContractError(f"TrialRecord: status {self.status.value} requires initial_context_sha256")
        if not scored and self.failure is None:
            raise ContractError(f"TrialRecord: status {self.status.value} requires a failure text")
        if self.submission is not None and self.adapter is not None and self.submission.adapter_id != self.adapter.adapter_id:
            raise ContractError("TrialRecord: submission.adapter_id does not match the adapter's id")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "spec": self.spec.to_dict(),
            "status": self.status.value,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "initial_context_sha256": self.initial_context_sha256,
            "budget_used": self.budget_used.to_dict(),
            "submission": None if self.submission is None else self.submission.to_dict(),
            "adapter": None if self.adapter is None else self.adapter.to_dict(),
            "dev_score_id": self.dev_score_id,
            "selection_score_id": self.selection_score_id,
            "trace": None if self.trace is None else self.trace.to_dict(),
            "notes": None if self.notes is None else self.notes.to_dict(),
            "trial_ledger_anchor": None if self.trial_ledger_anchor is None else self.trial_ledger_anchor.to_dict(),
            "environment": dict(self.environment),
            "failure": self.failure,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "TrialRecord":
        r = _Reader(d, "TrialRecord")
        out = cls(
            r.obj("spec", TrialSpec.from_dict),
            r.raw("status"),
            r.raw("started_utc"),
            r.raw("finished_utc"),
            r.raw("initial_context_sha256"),
            r.obj("budget_used", BudgetUsage.from_dict),
            r.opt("submission", Submission.from_dict),
            r.opt("adapter", AdapterRef.from_dict),
            r.raw("dev_score_id"),
            r.raw("selection_score_id"),
            r.opt("trace", FileRef.from_dict),
            r.opt("notes", FileRef.from_dict),
            r.opt("trial_ledger_anchor", LedgerAnchor.from_dict),
            r.raw("environment"),
            r.raw("failure"),
        )
        r.done()
        return out


# --------------------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoreRequest:
    """Ask the scorer to score one base model, with an optional adapter, on one split of one assignment.

    * ``request_id`` is also the score id of the resulting ScoreRecord (``ScoreRecord.score_id == ScoreRequest.request_id``).
    * ``final_request_id`` is set exactly when ``phase`` is FINAL: every final-phase score (dev included) belongs to a FinalRequest.
    * The test split uses ``assignment_id == TEST_ASSIGNMENT_ID`` and no other split may.
    * A TrainJob's dev template has ``arm`` None and ``adapter`` None and the trial's phase, seeds and decoding.
    """

    request_id: str
    phase: Phase
    assignment_id: str
    split: Split
    trial_id: Optional[str]
    arm: Optional[Arm]
    base: BaseModelRef
    adapter: Optional[AdapterRef]
    seeds: tuple
    params: GenerationParams
    scorer_version: str
    reason: Optional[str]
    final_request_id: Optional[str]
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.request_id, "ScoreRequest.request_id")
        _set(self, "phase", _enum(self.phase, Phase, "ScoreRequest.phase"))
        _ident(self.assignment_id, "ScoreRequest.assignment_id")
        _set(self, "split", _enum(self.split, Split, "ScoreRequest.split"))
        _opt(self.trial_id, _ident, "ScoreRequest.trial_id")
        _set(self, "arm", _opt(self.arm, lambda v, w: _enum(v, Arm, w), "ScoreRequest.arm"))
        _obj(self.base, BaseModelRef, "ScoreRequest.base")
        _opt(self.adapter, lambda v, w: _obj(v, AdapterRef, w), "ScoreRequest.adapter")
        _set(self, "seeds", _tuple_of(self.seeds, "ScoreRequest.seeds", lambda v, w: _int(v, w, minimum=0), nonempty=True, unique=True))
        _obj(self.params, GenerationParams, "ScoreRequest.params")
        _text(self.scorer_version, "ScoreRequest.scorer_version", max_bytes=200)
        _set(self, "reason", _opt(self.reason, _text, "ScoreRequest.reason"))
        _opt(self.final_request_id, _ident, "ScoreRequest.final_request_id")
        if self.arm is Arm.STOCK and self.adapter is not None:
            raise ContractError("ScoreRequest: arm stock has no adapter")
        if self.arm in (Arm.V0, Arm.EVOLVED) and self.adapter is None:
            raise ContractError("ScoreRequest: arm v0 / evolved needs an adapter")
        if self.split is Split.TEST and self.phase is not Phase.FINAL:
            raise ContractError("ScoreRequest: the test split can only be requested in the final phase")
        _check_phase_ids(self.phase, self.split, self.assignment_id, self.final_request_id, "ScoreRequest")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "phase": self.phase.value,
            "assignment_id": self.assignment_id,
            "split": self.split.value,
            "trial_id": self.trial_id,
            "arm": None if self.arm is None else self.arm.value,
            "base": self.base.to_dict(),
            "adapter": None if self.adapter is None else self.adapter.to_dict(),
            "seeds": list(self.seeds),
            "params": self.params.to_dict(),
            "scorer_version": self.scorer_version,
            "reason": self.reason,
            "final_request_id": self.final_request_id,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "ScoreRequest":
        r = _Reader(d, "ScoreRequest")
        out = cls(
            r.raw("request_id"),
            r.raw("phase"),
            r.raw("assignment_id"),
            r.raw("split"),
            r.raw("trial_id"),
            r.raw("arm"),
            r.obj("base", BaseModelRef.from_dict),
            r.opt("adapter", AdapterRef.from_dict),
            r.raw("seeds"),
            r.obj("params", GenerationParams.from_dict),
            r.raw("scorer_version"),
            r.raw("reason"),
            r.raw("final_request_id"),
        )
        r.done()
        return out

    def with_adapter(self, request_id: str, adapter: AdapterRef) -> "ScoreRequest":
        """Copy for another adapter (the trainer fills a dev template this way; ``arm`` stays None for dev)."""
        return replace(self, request_id=request_id, adapter=adapter)


def _check_phase_ids(phase: Phase, split: Split, assignment_id: str, final_request_id: Optional[str], what: str) -> None:
    if (split is Split.TEST) != (assignment_id == TEST_ASSIGNMENT_ID):
        raise ContractError(f"{what}: the test split, and only the test split, uses assignment_id {TEST_ASSIGNMENT_ID!r}")
    if phase is Phase.FINAL and final_request_id is None:
        raise ContractError(f"{what}: a final-phase score needs final_request_id")
    if phase is Phase.SEARCH and final_request_id is not None:
        raise ContractError(f"{what}: a search-phase score has no final_request_id")


@dataclass(frozen=True)
class SeedAggregate:
    """Aggregates of one generation seed: every metric is a pass rate in [0, 1] over the same ``n`` prompts."""

    seed: int
    n: int
    metrics: dict
    n_truncated: int
    n_errors: int

    def __post_init__(self) -> None:
        _int(self.seed, "SeedAggregate.seed", minimum=0)
        _int(self.n, "SeedAggregate.n", minimum=1)
        m = _str_map(self.metrics, "SeedAggregate.metrics", value=_frac)
        if not m:
            raise ContractError("SeedAggregate.metrics must not be empty")
        _set(self, "metrics", m)
        _int(self.n_truncated, "SeedAggregate.n_truncated", minimum=0)
        _int(self.n_errors, "SeedAggregate.n_errors", minimum=0)
        if self.n_truncated > self.n or self.n_errors > self.n:
            raise ContractError("SeedAggregate: n_truncated and n_errors cannot exceed n")

    def to_dict(self) -> dict:
        return {"seed": self.seed, "n": self.n, "metrics": dict(self.metrics), "n_truncated": self.n_truncated, "n_errors": self.n_errors}

    @classmethod
    def from_dict(cls, d: Any) -> "SeedAggregate":
        r = _Reader(d, "SeedAggregate", versioned=False)
        out = cls(r.raw("seed"), r.raw("n"), r.raw("metrics"), r.raw("n_truncated"), r.raw("n_errors"))
        r.done()
        return out


@dataclass(frozen=True)
class ScoreRecord:
    """Result of one ScoreRequest (payload of ``lab.score_recorded``): per-seed aggregates and a reference to per-prompt results.

    The scorer writes ``scores/<id>/request.json`` and ``score.json`` itself (``write_record``) and returns this record; the trial
    runner or CLI then appends the ledger event. ``score_id == request_id`` always.
    """

    score_id: str
    request_id: str
    phase: Phase
    assignment_id: str
    split: Split
    trial_id: Optional[str]
    arm: Optional[Arm]
    final_request_id: Optional[str]
    base: BaseModelRef
    adapter_sha256: Optional[str]
    scorer_version: str
    verifier: Verifier
    verifier_pins: dict
    params: GenerationParams
    per_seed: tuple
    n_prompts: int
    per_prompt: FileRef
    headline_metric: str
    created_utc: str
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.score_id, "ScoreRecord.score_id")
        _ident(self.request_id, "ScoreRecord.request_id")
        if self.score_id != self.request_id:
            raise ContractError("ScoreRecord: score_id must equal request_id")
        _set(self, "phase", _enum(self.phase, Phase, "ScoreRecord.phase"))
        _ident(self.assignment_id, "ScoreRecord.assignment_id")
        _set(self, "split", _enum(self.split, Split, "ScoreRecord.split"))
        _opt(self.trial_id, _ident, "ScoreRecord.trial_id")
        _set(self, "arm", _opt(self.arm, lambda v, w: _enum(v, Arm, w), "ScoreRecord.arm"))
        _opt(self.final_request_id, _ident, "ScoreRecord.final_request_id")
        _obj(self.base, BaseModelRef, "ScoreRecord.base")
        _opt(self.adapter_sha256, _sha256, "ScoreRecord.adapter_sha256")
        _text(self.scorer_version, "ScoreRecord.scorer_version", max_bytes=200)
        _set(self, "verifier", _enum(self.verifier, Verifier, "ScoreRecord.verifier"))
        expected = Verifier.IFBENCH_UPSTREAM if self.split is Split.TEST else Verifier.IFEVALG
        if self.verifier is not expected:
            raise ContractError(f"ScoreRecord: split {self.split.value} must be verified with {expected.value}, not {self.verifier.value}")
        _check_phase_ids(self.phase, self.split, self.assignment_id, self.final_request_id, "ScoreRecord")
        _set(self, "verifier_pins", _str_map(self.verifier_pins, "ScoreRecord.verifier_pins"))
        _obj(self.params, GenerationParams, "ScoreRecord.params")
        seeds = _tuple_of(self.per_seed, "ScoreRecord.per_seed", lambda v, w: _obj(v, SeedAggregate, w), nonempty=True)
        _set(self, "per_seed", seeds)
        if len({s.seed for s in seeds}) != len(seeds):
            raise ContractError("ScoreRecord.per_seed: duplicate seeds")
        _int(self.n_prompts, "ScoreRecord.n_prompts", minimum=1)
        if any(s.n != self.n_prompts for s in seeds):
            raise ContractError("ScoreRecord.per_seed: every seed must cover n_prompts prompts")
        _obj(self.per_prompt, FileRef, "ScoreRecord.per_prompt")
        _text(self.headline_metric, "ScoreRecord.headline_metric", max_bytes=100)
        allowed = METRICS_BY_SPLIT[self.split]
        if self.headline_metric not in allowed:
            raise ContractError(f"ScoreRecord: headline metric {self.headline_metric!r} is not a {self.split.value} metric {allowed}")
        if any(self.headline_metric not in s.metrics for s in seeds):
            raise ContractError(f"ScoreRecord: headline metric {self.headline_metric!r} missing from a seed")
        for s in seeds:
            extra = sorted(set(s.metrics) - set(allowed))
            if extra:
                raise ContractError(f"ScoreRecord: metric(s) {extra} are not {self.split.value} metrics {allowed}")
        _utc(self.created_utc, "ScoreRecord.created_utc")

    def metric_by_seed(self, name: Optional[str] = None) -> tuple:
        key = name or self.headline_metric
        return tuple(s.metrics[key] for s in self.per_seed)

    def headline_mean(self) -> float:
        """Mean over generation seeds of the headline metric. This is the number one trial contributes to the statistics, and the
        'dev score' of a dev record (unrounded)."""
        vals = self.metric_by_seed()
        return math.fsum(vals) / len(vals)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "score_id": self.score_id,
            "request_id": self.request_id,
            "phase": self.phase.value,
            "assignment_id": self.assignment_id,
            "split": self.split.value,
            "trial_id": self.trial_id,
            "arm": None if self.arm is None else self.arm.value,
            "final_request_id": self.final_request_id,
            "base": self.base.to_dict(),
            "adapter_sha256": self.adapter_sha256,
            "scorer_version": self.scorer_version,
            "verifier": self.verifier.value,
            "verifier_pins": dict(self.verifier_pins),
            "params": self.params.to_dict(),
            "per_seed": [s.to_dict() for s in self.per_seed],
            "n_prompts": self.n_prompts,
            "per_prompt": self.per_prompt.to_dict(),
            "headline_metric": self.headline_metric,
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "ScoreRecord":
        r = _Reader(d, "ScoreRecord")
        out = cls(
            r.raw("score_id"),
            r.raw("request_id"),
            r.raw("phase"),
            r.raw("assignment_id"),
            r.raw("split"),
            r.raw("trial_id"),
            r.raw("arm"),
            r.raw("final_request_id"),
            r.obj("base", BaseModelRef.from_dict),
            r.raw("adapter_sha256"),
            r.raw("scorer_version"),
            r.raw("verifier"),
            r.raw("verifier_pins"),
            r.obj("params", GenerationParams.from_dict),
            r.seq("per_seed", SeedAggregate.from_dict),
            r.raw("n_prompts"),
            r.obj("per_prompt", FileRef.from_dict),
            r.raw("headline_metric"),
            r.raw("created_utc"),
        )
        r.done()
        return out


@dataclass(frozen=True)
class ScoreCost:
    """Payload of ``lab.score_cost``: GPU billed for one scoring session (cost centre: scorer in the search, final in the final phase)."""

    score_id: str
    gpu: "GpuUsage"
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.score_id, "ScoreCost.score_id")
        _obj(self.gpu, GpuUsage, "ScoreCost.gpu")

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version, "score_id": self.score_id, "gpu": self.gpu.to_dict()}

    @classmethod
    def from_dict(cls, d: Any) -> "ScoreCost":
        r = _Reader(d, "ScoreCost")
        out = cls(r.raw("score_id"), r.obj("gpu", GpuUsage.from_dict))
        r.done()
        return out


# --------------------------------------------------------------------------------------------------
# the keep-or-reject rule, candidates, decisions
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TrialScore:
    """One trial's contribution to the statistics: the headline-metric mean of its selection ScoreRecord, with everything needed to
    decide whether two such numbers may be compared (``comparability_sha256`` is ``comparability_key(spec, metric)``).

    ``score_id`` is None only for a trial that failed and is counted as 0.0 by ``FailedTrialPolicy.SCORED_ZERO``.
    """

    trial_id: str
    assignment_id: str
    score_id: Optional[str]
    metric: str
    scorer_version: str
    comparability_sha256: str
    score: float

    def __post_init__(self) -> None:
        _ident(self.trial_id, "TrialScore.trial_id")
        _ident(self.assignment_id, "TrialScore.assignment_id")
        _opt(self.score_id, _ident, "TrialScore.score_id")
        _text(self.metric, "TrialScore.metric", max_bytes=100)
        _text(self.scorer_version, "TrialScore.scorer_version", max_bytes=200)
        _sha256(self.comparability_sha256, "TrialScore.comparability_sha256")
        _set(self, "score", _frac(self.score, "TrialScore.score"))

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "assignment_id": self.assignment_id,
            "score_id": self.score_id,
            "metric": self.metric,
            "scorer_version": self.scorer_version,
            "comparability_sha256": self.comparability_sha256,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "TrialScore":
        r = _Reader(d, "TrialScore", versioned=False)
        out = cls(
            r.raw("trial_id"),
            r.raw("assignment_id"),
            r.raw("score_id"),
            r.raw("metric"),
            r.raw("scorer_version"),
            r.raw("comparability_sha256"),
            r.raw("score"),
        )
        r.done()
        return out


@dataclass(frozen=True)
class SpreadEstimate:
    """Trial-to-trial spread of v0 on the selection split.

    Cells are assignments with at least two v0 trials; an assignment with one trial adds no degrees of freedom and is NOT in
    ``trial_ids``. ``df = sum over cells (n_a - 1) = len(trial_ids) - n_assignments`` and ``sigma^2`` is the pooled within-cell variance.
    """

    sigma: float
    df: int
    trial_ids: tuple
    n_assignments: int

    def __post_init__(self) -> None:
        _set(self, "sigma", _nonneg(self.sigma, "SpreadEstimate.sigma"))
        _int(self.df, "SpreadEstimate.df", minimum=1)
        ids = _tuple_of(self.trial_ids, "SpreadEstimate.trial_ids", _ident, nonempty=True, unique=True)
        _set(self, "trial_ids", ids)
        _int(self.n_assignments, "SpreadEstimate.n_assignments", minimum=1)
        if self.df != len(ids) - self.n_assignments:
            raise ContractError("SpreadEstimate: df must equal the number of trials minus the number of cells (assignments)")

    def to_dict(self) -> dict:
        return {"sigma": self.sigma, "df": self.df, "trial_ids": list(self.trial_ids), "n_assignments": self.n_assignments}

    @classmethod
    def from_dict(cls, d: Any) -> "SpreadEstimate":
        r = _Reader(d, "SpreadEstimate", versioned=False)
        out = cls(r.raw("sigma"), r.raw("df"), r.raw("trial_ids"), r.raw("n_assignments"))
        r.done()
        return out


@dataclass(frozen=True)
class SpreadPlan:
    """Payload of ``lab.spread_planned``: how many v0 trials will measure the spread, declared BEFORE the first one runs, so the
    spread can never be taken from whichever trials were convenient."""

    metric: str
    scorer_version: str
    assignments: tuple
    planned_trials: int
    declared_utc: str
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        if self.metric not in METRICS_BY_SPLIT[Split.SELECTION]:
            raise ContractError(f"SpreadPlan.metric: {self.metric!r} is not a selection metric")
        _text(self.scorer_version, "SpreadPlan.scorer_version", max_bytes=200)
        _set(self, "assignments", _tuple_of(self.assignments, "SpreadPlan.assignments", _ident, nonempty=True, unique=True))
        _int(self.planned_trials, "SpreadPlan.planned_trials", minimum=2)
        _utc(self.declared_utc, "SpreadPlan.declared_utc")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "metric": self.metric,
            "scorer_version": self.scorer_version,
            "assignments": list(self.assignments),
            "planned_trials": self.planned_trials,
            "declared_utc": self.declared_utc,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "SpreadPlan":
        r = _Reader(d, "SpreadPlan")
        out = cls(r.raw("metric"), r.raw("scorer_version"), r.raw("assignments"), r.raw("planned_trials"), r.raw("declared_utc"))
        r.done()
        return out


@dataclass(frozen=True)
class SpreadMeasured:
    """Payload of ``lab.spread_measured``: ALL completed v0 trials of the declared plan, never a subset. ``n_failed`` counts planned
    trials that did not complete; ``freeze-rule`` refuses unless it is 0 (a new plan must be declared instead)."""

    metric: str
    scorer_version: str
    planned_trials: int
    n_failed: int
    trials: tuple
    estimate: SpreadEstimate
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _text(self.metric, "SpreadMeasured.metric", max_bytes=100)
        _text(self.scorer_version, "SpreadMeasured.scorer_version", max_bytes=200)
        _int(self.planned_trials, "SpreadMeasured.planned_trials", minimum=2)
        _int(self.n_failed, "SpreadMeasured.n_failed", minimum=0)
        _set(self, "trials", _tuple_of(self.trials, "SpreadMeasured.trials", lambda v, w: _obj(v, TrialScore, w), nonempty=True))
        _obj(self.estimate, SpreadEstimate, "SpreadMeasured.estimate")
        if len(self.trials) + self.n_failed != self.planned_trials:
            raise ContractError("SpreadMeasured: completed trials plus failed trials must equal the planned number")
        in_cells = {t.trial_id for t in self.trials}
        if not set(self.estimate.trial_ids) <= in_cells:
            raise ContractError("SpreadMeasured: the estimate uses trials that are not listed")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "metric": self.metric,
            "scorer_version": self.scorer_version,
            "planned_trials": self.planned_trials,
            "n_failed": self.n_failed,
            "trials": [t.to_dict() for t in self.trials],
            "estimate": self.estimate.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "SpreadMeasured":
        r = _Reader(d, "SpreadMeasured")
        out = cls(
            r.raw("metric"),
            r.raw("scorer_version"),
            r.raw("planned_trials"),
            r.raw("n_failed"),
            r.seq("trials", TrialScore.from_dict),
            r.obj("estimate", SpreadEstimate.from_dict),
        )
        r.done()
        return out


@dataclass(frozen=True)
class Rule:
    """The fixed keep-or-reject rule (``rule.json``), written before any candidate result. Screen, then confirm.

    Search rounds: ``screen_batch_size`` candidates are each run for ``screen_trials_per_candidate`` trials per assignment and ranked by
    delta against the incumbent (no threshold). Only the best one with delta > 0 is confirmed: ``confirm_trials_per_candidate`` FRESH
    trials of it and ``confirm_trials_per_incumbent`` FRESH, interleaved trials of the incumbent. Accept iff ``delta > margin`` with
    ``margin = t(alpha, df) * sigma_used * se_factor + min_gain`` (docs/harness-design.md, "Statistics"). A candidate whose trial sample
    sd exceeds ``variance_guard_ratio`` times the frozen sigma is INVALID. The rule is a heuristic filter, not evidence of a gain.

    ``minimum_detectable_gain`` is computed by stats from the measured sigma and the confirm sample sizes; the rule cannot be built if it
    exceeds ``assumed_gain`` (the gain the operator believes a good candidate has).
    """

    metric: str
    split: Split
    scorer_version: str
    assignments: tuple
    spread: SpreadEstimate
    alpha: float
    min_gain: float
    score_seeds: tuple
    score_params: GenerationParams
    screen_batch_size: int
    screen_trials_per_candidate: int
    confirm_trials_per_candidate: int
    confirm_trials_per_incumbent: int
    variance_guard_ratio: float
    failed_trial_policy: FailedTrialPolicy
    assumed_gain: float
    minimum_detectable_gain: float
    degenerate_override: Optional[str]
    max_candidates: int
    stop_after_rejections: Optional[int]
    repeated_decisions_note: str
    frozen_utc: str
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _text(self.metric, "Rule.metric", max_bytes=100)
        _set(self, "split", _enum(self.split, Split, "Rule.split"))
        if self.split is not Split.SELECTION:
            raise ContractError("Rule.split: the keep-or-reject rule is evaluated on the selection split only")
        if self.metric not in METRICS_BY_SPLIT[Split.SELECTION]:
            raise ContractError(f"Rule.metric: {self.metric!r} is not a selection metric")
        _text(self.scorer_version, "Rule.scorer_version", max_bytes=200)
        _set(self, "assignments", _tuple_of(self.assignments, "Rule.assignments", _ident, nonempty=True, unique=True))
        _obj(self.spread, SpreadEstimate, "Rule.spread")
        _set(self, "alpha", _num(self.alpha, "Rule.alpha"))
        if not 0.0 < self.alpha <= 0.5:
            raise ContractError("Rule.alpha: the one-sided significance level must be in (0, 0.5]")
        _set(self, "min_gain", _nonneg(self.min_gain, "Rule.min_gain"))
        _set(self, "score_seeds", _tuple_of(self.score_seeds, "Rule.score_seeds", lambda v, w: _int(v, w, minimum=0), nonempty=True, unique=True))
        _obj(self.score_params, GenerationParams, "Rule.score_params")
        _int(self.screen_batch_size, "Rule.screen_batch_size", minimum=1)
        if self.screen_batch_size > MAX_SCREEN_BATCH:
            raise ContractError(f"Rule.screen_batch_size: at most {MAX_SCREEN_BATCH} candidates are screened per round")
        _int(self.screen_trials_per_candidate, "Rule.screen_trials_per_candidate", minimum=1)
        _int(self.confirm_trials_per_candidate, "Rule.confirm_trials_per_candidate", minimum=2)
        _int(self.confirm_trials_per_incumbent, "Rule.confirm_trials_per_incumbent", minimum=2)
        _set(self, "variance_guard_ratio", _num(self.variance_guard_ratio, "Rule.variance_guard_ratio"))
        if self.variance_guard_ratio < 1.0:
            raise ContractError("Rule.variance_guard_ratio: must be >= 1")
        _set(self, "failed_trial_policy", _enum(self.failed_trial_policy, FailedTrialPolicy, "Rule.failed_trial_policy"))
        _set(self, "assumed_gain", _pos(self.assumed_gain, "Rule.assumed_gain"))
        _set(self, "minimum_detectable_gain", _pos(self.minimum_detectable_gain, "Rule.minimum_detectable_gain"))
        if self.minimum_detectable_gain > self.assumed_gain:
            raise ContractError("Rule: the minimum detectable gain exceeds the assumed gain; the rule could not detect a gain of that size")
        _set(self, "degenerate_override", _opt(self.degenerate_override, _text, "Rule.degenerate_override"))
        if self.spread.sigma == 0.0 and self.min_gain == 0.0 and self.degenerate_override is None:
            raise ContractError("Rule: sigma is 0 and min_gain is 0, so any positive delta would be accepted; set min_gain or record an override reason")
        if self.pooled_df() < MIN_POOLED_DF:
            raise ContractError(f"Rule: pooled degrees of freedom {self.pooled_df()} is below {MIN_POOLED_DF}; measure more v0 trials or confirm with more trials")
        _int(self.max_candidates, "Rule.max_candidates", minimum=1)
        _opt(self.stop_after_rejections, lambda v, w: _int(v, w, minimum=1), "Rule.stop_after_rejections")
        _text(self.repeated_decisions_note, "Rule.repeated_decisions_note")
        _utc(self.frozen_utc, "Rule.frozen_utc")

    def pooled_df(self) -> int:
        """Degrees of freedom of the pooled variance at decision time: frozen v0 df plus the candidate's and incumbent's own df."""
        return self.spread.df + (self.confirm_trials_per_candidate - 1) + (self.confirm_trials_per_incumbent - 1)

    def planned_search_trials(self) -> int:
        """Worst-case number of search-phase trials: the v0 spread trials, every candidate screened, every round confirmed."""
        a = len(self.assignments)
        rounds = -(-self.max_candidates // self.screen_batch_size)
        return (
            len(self.spread.trial_ids)
            + self.max_candidates * self.screen_trials_per_candidate * a
            + rounds * (self.confirm_trials_per_candidate + self.confirm_trials_per_incumbent) * a
        )

    def sha256(self) -> str:
        return record_sha256(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "metric": self.metric,
            "split": self.split.value,
            "scorer_version": self.scorer_version,
            "assignments": list(self.assignments),
            "spread": self.spread.to_dict(),
            "alpha": self.alpha,
            "min_gain": self.min_gain,
            "score_seeds": list(self.score_seeds),
            "score_params": self.score_params.to_dict(),
            "screen_batch_size": self.screen_batch_size,
            "screen_trials_per_candidate": self.screen_trials_per_candidate,
            "confirm_trials_per_candidate": self.confirm_trials_per_candidate,
            "confirm_trials_per_incumbent": self.confirm_trials_per_incumbent,
            "variance_guard_ratio": self.variance_guard_ratio,
            "failed_trial_policy": self.failed_trial_policy.value,
            "assumed_gain": self.assumed_gain,
            "minimum_detectable_gain": self.minimum_detectable_gain,
            "degenerate_override": self.degenerate_override,
            "max_candidates": self.max_candidates,
            "stop_after_rejections": self.stop_after_rejections,
            "repeated_decisions_note": self.repeated_decisions_note,
            "frozen_utc": self.frozen_utc,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "Rule":
        r = _Reader(d, "Rule")
        out = cls(
            r.raw("metric"),
            r.raw("split"),
            r.raw("scorer_version"),
            r.raw("assignments"),
            r.obj("spread", SpreadEstimate.from_dict),
            r.raw("alpha"),
            r.raw("min_gain"),
            r.raw("score_seeds"),
            r.obj("score_params", GenerationParams.from_dict),
            r.raw("screen_batch_size"),
            r.raw("screen_trials_per_candidate"),
            r.raw("confirm_trials_per_candidate"),
            r.raw("confirm_trials_per_incumbent"),
            r.raw("variance_guard_ratio"),
            r.raw("failed_trial_policy"),
            r.raw("assumed_gain"),
            r.raw("minimum_detectable_gain"),
            r.raw("degenerate_override"),
            r.raw("max_candidates"),
            r.raw("stop_after_rejections"),
            r.raw("repeated_decisions_note"),
            r.raw("frozen_utc"),
        )
        r.done()
        return out


@dataclass(frozen=True)
class SearchReplace:
    search: str
    replace: str

    def __post_init__(self) -> None:
        _text(self.search, "SearchReplace.search", max_bytes=1 << 20)
        _text(self.replace, "SearchReplace.replace", empty=True, max_bytes=1 << 20)

    def to_dict(self) -> dict:
        return {"search": self.search, "replace": self.replace}

    @classmethod
    def from_dict(cls, d: Any) -> "SearchReplace":
        r = _Reader(d, "SearchReplace", versioned=False)
        out = cls(r.raw("search"), r.raw("replace"))
        r.done()
        return out


@dataclass(frozen=True)
class PackEdit:
    """One edit to one pack file. Which paths are editable is the pack module's allowlist, not this contract's."""

    path: str
    op: EditOp
    content: Optional[str]
    blocks: tuple

    def __post_init__(self) -> None:
        _relpath(self.path, "PackEdit.path")
        _set(self, "op", _enum(self.op, EditOp, "PackEdit.op"))
        _set(self, "blocks", _tuple_of(self.blocks, "PackEdit.blocks", lambda v, w: _obj(v, SearchReplace, w)))
        if self.op is EditOp.REPLACE_FILE:
            _text(self.content, "PackEdit.content", empty=True, max_bytes=1 << 20)
            if self.blocks:
                raise ContractError("PackEdit: replace_file takes content, not blocks")
        else:
            if self.content is not None:
                raise ContractError("PackEdit: search_replace takes blocks, not content")
            if not self.blocks:
                raise ContractError("PackEdit: search_replace needs at least one block")

    def to_dict(self) -> dict:
        return {"path": self.path, "op": self.op.value, "content": self.content, "blocks": [b.to_dict() for b in self.blocks]}

    @classmethod
    def from_dict(cls, d: Any) -> "PackEdit":
        r = _Reader(d, "PackEdit", versioned=False)
        out = cls(r.raw("path"), r.raw("op"), r.raw("content"), r.seq("blocks", SearchReplace.from_dict))
        r.done()
        return out


@dataclass(frozen=True)
class Proposal:
    """What the proposer returns: ONE atomic change with its reason. Stored as ``candidates/<id>/proposal.json`` (a file, not a ledger payload)."""

    hypothesis: str
    lever: Lever
    change_summary: str
    expected_effect: str
    risk: str
    edits: tuple
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        for name in ("hypothesis", "change_summary", "expected_effect", "risk"):
            _text(getattr(self, name), f"Proposal.{name}")
        _set(self, "lever", _enum(self.lever, Lever, "Proposal.lever"))
        edits = _tuple_of(self.edits, "Proposal.edits", lambda v, w: _obj(v, PackEdit, w), nonempty=True)
        paths = [e.path for e in edits]
        if len(set(paths)) != len(paths):
            raise ContractError("Proposal.edits: one edit per path")
        _set(self, "edits", edits)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "hypothesis": self.hypothesis,
            "lever": self.lever.value,
            "change_summary": self.change_summary,
            "expected_effect": self.expected_effect,
            "risk": self.risk,
            "edits": [e.to_dict() for e in self.edits],
        }

    @classmethod
    def from_dict(cls, d: Any) -> "Proposal":
        r = _Reader(d, "Proposal")
        out = cls(r.raw("hypothesis"), r.raw("lever"), r.raw("change_summary"), r.raw("expected_effect"), r.raw("risk"), r.seq("edits", PackEdit.from_dict))
        r.done()
        return out


@dataclass(frozen=True)
class ChallengeSummary:
    """The outer challenge of a candidate (``outer_challenge``), as its ``CandidateRecord`` carries it: the full report (checker findings,
    critic and revision artefacts with hashes) is the file ``report``; here are the facts a reader needs without opening it. ``blocking``
    lists the hard checker findings new in the final candidate; a candidate is refused exactly when it is non-empty."""

    report: FileRef
    blocking: tuple
    critic_model: str
    critic_served_model: str
    critic_sha256: Optional[str]
    revised: bool
    refused: bool

    def __post_init__(self) -> None:
        _obj(self.report, FileRef, "ChallengeSummary.report")
        _set(self, "blocking", _tuple_of(self.blocking, "ChallengeSummary.blocking", lambda v, w: _text(v, w, max_bytes=200)))
        _text(self.critic_model, "ChallengeSummary.critic_model", empty=True, max_bytes=200)
        _text(self.critic_served_model, "ChallengeSummary.critic_served_model", empty=True, max_bytes=200)
        _opt(self.critic_sha256, _sha256, "ChallengeSummary.critic_sha256")
        _bool(self.revised, "ChallengeSummary.revised")
        _bool(self.refused, "ChallengeSummary.refused")
        if self.refused != bool(self.blocking):
            raise ContractError("ChallengeSummary: a candidate is refused exactly when blocking findings remain")

    def to_dict(self) -> dict:
        return {
            "report": self.report.to_dict(),
            "blocking": list(self.blocking),
            "critic_model": self.critic_model,
            "critic_served_model": self.critic_served_model,
            "critic_sha256": self.critic_sha256,
            "revised": self.revised,
            "refused": self.refused,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "ChallengeSummary":
        r = _Reader(d, "ChallengeSummary", versioned=False)
        out = cls(r.obj("report", FileRef.from_dict), r.raw("blocking"), r.raw("critic_model"), r.raw("critic_served_model"), r.raw("critic_sha256"),
                  r.raw("revised"), r.raw("refused"))
        r.done()
        return out


@dataclass(frozen=True)
class CandidateRecord:
    """Payload of ``lab.candidate_proposed``: every candidate, accepted or not, with its diff, reason and lint result. ``challenge`` is
    present only when the outer challenge ran (records written before it existed have no such key and read back as None)."""

    candidate_id: str
    version_label: str
    parent: PackRef
    parent_version: str
    pack: Optional[PackRef]
    proposal: FileRef
    diff: Optional[FileRef]
    prompt: FileRef
    response: FileRef
    lever: Lever
    hypothesis: str
    reason: str
    lint_passed: bool
    lint_messages: tuple
    proposer_model: str
    proposer_served_model: str
    fallback_used: bool
    propose_usd: float
    created_utc: str
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)
    challenge: Optional[ChallengeSummary] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.candidate_id, "CandidateRecord.candidate_id")
        _ident(self.version_label, "CandidateRecord.version_label")
        if self.version_label != self.candidate_id:
            raise ContractError("CandidateRecord: version_label must equal candidate_id (the pack tag is pack/<candidate_id>)")
        _obj(self.parent, PackRef, "CandidateRecord.parent")
        _ident(self.parent_version, "CandidateRecord.parent_version")
        _opt(self.pack, lambda v, w: _obj(v, PackRef, w), "CandidateRecord.pack")
        _obj(self.proposal, FileRef, "CandidateRecord.proposal")
        _opt(self.diff, lambda v, w: _obj(v, FileRef, w), "CandidateRecord.diff")
        _obj(self.prompt, FileRef, "CandidateRecord.prompt")
        _obj(self.response, FileRef, "CandidateRecord.response")
        _set(self, "lever", _enum(self.lever, Lever, "CandidateRecord.lever"))
        _text(self.hypothesis, "CandidateRecord.hypothesis")
        _text(self.reason, "CandidateRecord.reason")
        _bool(self.lint_passed, "CandidateRecord.lint_passed")
        _set(self, "lint_messages", _tuple_of(self.lint_messages, "CandidateRecord.lint_messages", lambda v, w: _text(v, w, max_bytes=500)))
        _text(self.proposer_model, "CandidateRecord.proposer_model", max_bytes=200)
        _text(self.proposer_served_model, "CandidateRecord.proposer_served_model", max_bytes=200)
        _bool(self.fallback_used, "CandidateRecord.fallback_used")
        _set(self, "propose_usd", _nonneg(self.propose_usd, "CandidateRecord.propose_usd"))
        _utc(self.created_utc, "CandidateRecord.created_utc")
        if self.lint_passed and (self.pack is None or self.diff is None):
            raise ContractError("CandidateRecord: a candidate that passed lint has a pack commit and a diff")
        _opt(self.challenge, lambda v, w: _obj(v, ChallengeSummary, w), "CandidateRecord.challenge")
        if self.challenge is not None and self.challenge.refused and (self.lint_passed or self.pack is not None):
            raise ContractError("CandidateRecord: a candidate the challenge refused has no pack commit and did not pass lint")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "version_label": self.version_label,
            "parent": self.parent.to_dict(),
            "parent_version": self.parent_version,
            "pack": None if self.pack is None else self.pack.to_dict(),
            "proposal": self.proposal.to_dict(),
            "diff": None if self.diff is None else self.diff.to_dict(),
            "prompt": self.prompt.to_dict(),
            "response": self.response.to_dict(),
            "lever": self.lever.value,
            "hypothesis": self.hypothesis,
            "reason": self.reason,
            "lint_passed": self.lint_passed,
            "lint_messages": list(self.lint_messages),
            "proposer_model": self.proposer_model,
            "proposer_served_model": self.proposer_served_model,
            "fallback_used": self.fallback_used,
            "propose_usd": self.propose_usd,
            "created_utc": self.created_utc,
            **({"challenge": self.challenge.to_dict()} if self.challenge is not None else {}),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "CandidateRecord":
        r = _Reader(d, "CandidateRecord")
        out = cls(
            r.raw("candidate_id"),
            r.raw("version_label"),
            r.obj("parent", PackRef.from_dict),
            r.raw("parent_version"),
            r.opt("pack", PackRef.from_dict),
            r.obj("proposal", FileRef.from_dict),
            r.opt("diff", FileRef.from_dict),
            r.obj("prompt", FileRef.from_dict),
            r.obj("response", FileRef.from_dict),
            r.raw("lever"),
            r.raw("hypothesis"),
            r.raw("reason"),
            r.raw("lint_passed"),
            r.raw("lint_messages"),
            r.raw("proposer_model"),
            r.raw("proposer_served_model"),
            r.raw("fallback_used"),
            r.raw("propose_usd"),
            r.raw("created_utc"),
            challenge=(r.opt("challenge", ChallengeSummary.from_dict) if "challenge" in d else None),
        )
        r.done()
        return out


def _check_numbers(what: str, stage: DecisionStage, outcome: Outcome, delta: Any, margin: Any, df: Any, sigma: Any) -> None:
    nums = (delta, margin, df, sigma)
    if outcome is Outcome.INVALID:
        if any(v is not None for v in nums):
            raise ContractError(f"{what}: an invalid candidate has no delta, margin, df or sigma")
    elif stage is DecisionStage.SCREEN:
        if outcome is Outcome.ACCEPTED:
            raise ContractError(f"{what}: a screen cannot accept; only a confirmation can")
        if any(v is not None for v in nums):
            raise ContractError(f"{what}: a screen decision carries screen_delta only")
    else:
        if any(v is None for v in nums):
            raise ContractError(f"{what}: an accepted or rejected confirmation needs delta, margin, df and sigma_used")
        if (delta > margin) != (outcome is Outcome.ACCEPTED):
            raise ContractError(f"{what}: outcome contradicts the rule (accepted iff delta > margin)")


@dataclass(frozen=True)
class DecisionNumbers:
    """What ``DecisionRule.decide`` returns for a confirmation (an exchange type from stats to the outer loop; not persisted)."""

    outcome: Outcome
    delta: Optional[float]
    margin: Optional[float]
    df: Optional[int]
    sigma_used: Optional[float]
    reason: str

    def __post_init__(self) -> None:
        _set(self, "outcome", _enum(self.outcome, Outcome, "DecisionNumbers.outcome"))
        _set(self, "delta", _opt(self.delta, _num, "DecisionNumbers.delta"))
        _set(self, "margin", _opt(self.margin, _nonneg, "DecisionNumbers.margin"))
        _opt(self.df, lambda v, w: _int(v, w, minimum=1), "DecisionNumbers.df")
        _set(self, "sigma_used", _opt(self.sigma_used, _nonneg, "DecisionNumbers.sigma_used"))
        _text(self.reason, "DecisionNumbers.reason")
        _check_numbers("DecisionNumbers", DecisionStage.CONFIRM, self.outcome, self.delta, self.margin, self.df, self.sigma_used)


@dataclass(frozen=True)
class Decision:
    """Payload of ``lab.candidate_decided``. Made by the controller from the Rule, never by the proposer. One per candidate.

    SCREEN: the candidate lost its round (REJECTED, ``screen_delta`` set) or could not be compared (INVALID). CONFIRM: the round's best
    candidate was tested; ``delta``, ``margin``, ``df`` (pooled) and ``sigma_used`` are recorded and accepted iff ``delta > margin``.
    ``candidate_version`` equals ``candidate_id``.
    """

    candidate_id: str
    rule_sha256: str
    stage: DecisionStage
    round_index: int
    outcome: Outcome
    reason: str
    incumbent_version: str
    candidate_version: str
    incumbent_scores: tuple
    candidate_scores: tuple
    screen_delta: Optional[float]
    delta: Optional[float]
    margin: Optional[float]
    df: Optional[int]
    sigma_used: Optional[float]
    cost: CostBreakdown
    decided_utc: str
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.candidate_id, "Decision.candidate_id")
        _sha256(self.rule_sha256, "Decision.rule_sha256")
        _set(self, "stage", _enum(self.stage, DecisionStage, "Decision.stage"))
        _int(self.round_index, "Decision.round_index", minimum=0)
        _set(self, "outcome", _enum(self.outcome, Outcome, "Decision.outcome"))
        _text(self.reason, "Decision.reason")
        _ident(self.incumbent_version, "Decision.incumbent_version")
        _ident(self.candidate_version, "Decision.candidate_version")
        if self.candidate_version != self.candidate_id:
            raise ContractError("Decision: candidate_version must equal candidate_id")
        _set(self, "incumbent_scores", _tuple_of(self.incumbent_scores, "Decision.incumbent_scores", lambda v, w: _obj(v, TrialScore, w)))
        _set(self, "candidate_scores", _tuple_of(self.candidate_scores, "Decision.candidate_scores", lambda v, w: _obj(v, TrialScore, w)))
        _set(self, "screen_delta", _opt(self.screen_delta, _num, "Decision.screen_delta"))
        _set(self, "delta", _opt(self.delta, _num, "Decision.delta"))
        _set(self, "margin", _opt(self.margin, _nonneg, "Decision.margin"))
        _opt(self.df, lambda v, w: _int(v, w, minimum=1), "Decision.df")
        _set(self, "sigma_used", _opt(self.sigma_used, _nonneg, "Decision.sigma_used"))
        _obj(self.cost, CostBreakdown, "Decision.cost")
        _utc(self.decided_utc, "Decision.decided_utc")
        _check_numbers("Decision", self.stage, self.outcome, self.delta, self.margin, self.df, self.sigma_used)
        if self.outcome is not Outcome.INVALID and self.screen_delta is None:
            raise ContractError("Decision: a valid decision records the screen delta")

    @property
    def accepted(self) -> bool:
        return self.outcome is Outcome.ACCEPTED

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "rule_sha256": self.rule_sha256,
            "stage": self.stage.value,
            "round_index": self.round_index,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "incumbent_version": self.incumbent_version,
            "candidate_version": self.candidate_version,
            "incumbent_scores": [s.to_dict() for s in self.incumbent_scores],
            "candidate_scores": [s.to_dict() for s in self.candidate_scores],
            "screen_delta": self.screen_delta,
            "delta": self.delta,
            "margin": self.margin,
            "df": self.df,
            "sigma_used": self.sigma_used,
            "cost": self.cost.to_dict(),
            "decided_utc": self.decided_utc,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "Decision":
        r = _Reader(d, "Decision")
        out = cls(
            r.raw("candidate_id"),
            r.raw("rule_sha256"),
            r.raw("stage"),
            r.raw("round_index"),
            r.raw("outcome"),
            r.raw("reason"),
            r.raw("incumbent_version"),
            r.raw("candidate_version"),
            r.seq("incumbent_scores", TrialScore.from_dict),
            r.seq("candidate_scores", TrialScore.from_dict),
            r.raw("screen_delta"),
            r.raw("delta"),
            r.raw("margin"),
            r.raw("df"),
            r.raw("sigma_used"),
            r.obj("cost", CostBreakdown.from_dict),
            r.raw("decided_utc"),
        )
        r.done()
        return out


# --------------------------------------------------------------------------------------------------
# remaining ledger payloads
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LabInit:
    lab_id: str
    created_utc: str
    host: str
    layout_version: int = LAYOUT_VERSION
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.lab_id, "LabInit.lab_id")
        _utc(self.created_utc, "LabInit.created_utc")
        _text(self.host, "LabInit.host", max_bytes=200)
        _int(self.layout_version, "LabInit.layout_version", minimum=1)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "lab_id": self.lab_id,
            "created_utc": self.created_utc,
            "host": self.host,
            "layout_version": self.layout_version,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "LabInit":
        r = _Reader(d, "LabInit")
        out = cls(r.raw("lab_id"), r.raw("created_utc"), r.raw("host"), r.raw("layout_version"))
        r.done()
        return out


@dataclass(frozen=True)
class FrozenFile:
    """Payload of ``lab.assignments_frozen``, ``lab.budget_frozen`` and ``lab.rule_frozen``: a file frozen by hash."""

    name: str
    file: FileRef
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.name, "FrozenFile.name")
        _obj(self.file, FileRef, "FrozenFile.file")

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version, "name": self.name, "file": self.file.to_dict()}

    @classmethod
    def from_dict(cls, d: Any) -> "FrozenFile":
        r = _Reader(d, "FrozenFile")
        out = cls(r.raw("name"), r.obj("file", FileRef.from_dict))
        r.done()
        return out


@dataclass(frozen=True)
class GateRecord:
    """Payload of ``lab.gate_recorded``. The outer loop refuses to propose until the terms gate has one. ``CLARIFIED`` closes the gate;
    ``PROCEED_WITH_RISK`` is a waiver that lets work start while the gate stays open and must be disclosed as open."""

    gate: Gate
    outcome: GateOutcome
    statement: str
    decided_by: str
    recorded_utc: str
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _set(self, "gate", _enum(self.gate, Gate, "GateRecord.gate"))
        _set(self, "outcome", _enum(self.outcome, GateOutcome, "GateRecord.outcome"))
        _text(self.statement, "GateRecord.statement")
        _text(self.decided_by, "GateRecord.decided_by", max_bytes=200)
        _utc(self.recorded_utc, "GateRecord.recorded_utc")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "gate": self.gate.value,
            "outcome": self.outcome.value,
            "statement": self.statement,
            "decided_by": self.decided_by,
            "recorded_utc": self.recorded_utc,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "GateRecord":
        r = _Reader(d, "GateRecord")
        out = cls(r.raw("gate"), r.raw("outcome"), r.raw("statement"), r.raw("decided_by"), r.raw("recorded_utc"))
        r.done()
        return out


@dataclass(frozen=True)
class PackCommitted:
    """Payload of ``lab.pack_committed``: a pack commit exists (v0, a candidate, an accepted version)."""

    version_label: str
    pack: PackRef
    parent: Optional[PackRef]
    candidate_id: Optional[str]
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.version_label, "PackCommitted.version_label")
        _obj(self.pack, PackRef, "PackCommitted.pack")
        _opt(self.parent, lambda v, w: _obj(v, PackRef, w), "PackCommitted.parent")
        _opt(self.candidate_id, _ident, "PackCommitted.candidate_id")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "version_label": self.version_label,
            "pack": self.pack.to_dict(),
            "parent": None if self.parent is None else self.parent.to_dict(),
            "candidate_id": self.candidate_id,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "PackCommitted":
        r = _Reader(d, "PackCommitted")
        out = cls(r.raw("version_label"), r.obj("pack", PackRef.from_dict), r.opt("parent", PackRef.from_dict), r.raw("candidate_id"))
        r.done()
        return out


@dataclass(frozen=True)
class FinalRequest:
    """Payload of ``lab.final_requested``, appended BEFORE any final-phase score. One per (arm, pack commit); it covers every planned
    trial of that pack (the plan is the frozen FinalAnalysis, bound here by ``analysis_sha256``). A second request for the same pack
    commit needs ``repeat_of`` and a reason. Arm 1 (stock) has no pack: ``arm is STOCK`` if and only if ``pack`` and ``pack_version``
    are None. ``splits`` are the splits this request unlocks (never dev)."""

    request_id: str
    arm: Arm
    pack_version: Optional[str]
    pack: Optional[PackRef]
    analysis_sha256: str
    splits: tuple
    repeat_of: Optional[str]
    reason: Optional[str]
    requested_utc: str
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.request_id, "FinalRequest.request_id")
        _set(self, "arm", _enum(self.arm, Arm, "FinalRequest.arm"))
        _opt(self.pack_version, _ident, "FinalRequest.pack_version")
        _opt(self.pack, lambda v, w: _obj(v, PackRef, w), "FinalRequest.pack")
        if (self.arm is Arm.STOCK) != (self.pack is None) or (self.pack is None) != (self.pack_version is None):
            raise ContractError("FinalRequest: arm stock has no pack and no pack_version; arm v0 / evolved has both")
        _sha256(self.analysis_sha256, "FinalRequest.analysis_sha256")
        splits = _tuple_of(self.splits, "FinalRequest.splits", lambda v, w: _enum(v, Split, w), nonempty=True, unique=True)
        if Split.DEV in splits:
            raise ContractError("FinalRequest.splits: dev is not a final split")
        _set(self, "splits", splits)
        _opt(self.repeat_of, _ident, "FinalRequest.repeat_of")
        _set(self, "reason", _opt(self.reason, _text, "FinalRequest.reason"))
        _utc(self.requested_utc, "FinalRequest.requested_utc")
        if self.repeat_of is not None and self.reason is None:
            raise ContractError("FinalRequest: a repeat of a final comparison needs a recorded reason")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "arm": self.arm.value,
            "pack_version": self.pack_version,
            "pack": None if self.pack is None else self.pack.to_dict(),
            "analysis_sha256": self.analysis_sha256,
            "splits": [s.value for s in self.splits],
            "repeat_of": self.repeat_of,
            "reason": self.reason,
            "requested_utc": self.requested_utc,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "FinalRequest":
        r = _Reader(d, "FinalRequest")
        out = cls(
            r.raw("request_id"),
            r.raw("arm"),
            r.raw("pack_version"),
            r.opt("pack", PackRef.from_dict),
            r.raw("analysis_sha256"),
            r.raw("splits"),
            r.raw("repeat_of"),
            r.raw("reason"),
            r.raw("requested_utc"),
        )
        r.done()
        return out


@dataclass(frozen=True)
class AssignmentWeight:
    assignment_id: str
    weight: float

    def __post_init__(self) -> None:
        _ident(self.assignment_id, "AssignmentWeight.assignment_id")
        _set(self, "weight", _pos(self.weight, "AssignmentWeight.weight"))
        if self.weight > 1.0:
            raise ContractError("AssignmentWeight.weight: must be <= 1")

    def to_dict(self) -> dict:
        return {"assignment_id": self.assignment_id, "weight": self.weight}

    @classmethod
    def from_dict(cls, d: Any) -> "AssignmentWeight":
        r = _Reader(d, "AssignmentWeight", versioned=False)
        out = cls(r.raw("assignment_id"), r.raw("weight"))
        r.done()
        return out


@dataclass(frozen=True)
class PlannedTrial:
    """One trial the final analysis plans: nothing outside this list may be scored on held-out or test data."""

    trial_id: str
    arm: Arm
    assignment_id: str
    seed: int

    def __post_init__(self) -> None:
        _ident(self.trial_id, "PlannedTrial.trial_id")
        _set(self, "arm", _enum(self.arm, Arm, "PlannedTrial.arm"))
        if self.arm not in (Arm.V0, Arm.EVOLVED):
            raise ContractError("PlannedTrial.arm: arm 1 has no trials")
        _ident(self.assignment_id, "PlannedTrial.assignment_id")
        _int(self.seed, "PlannedTrial.seed", minimum=0)

    def to_dict(self) -> dict:
        return {"trial_id": self.trial_id, "arm": self.arm.value, "assignment_id": self.assignment_id, "seed": self.seed}

    @classmethod
    def from_dict(cls, d: Any) -> "PlannedTrial":
        r = _Reader(d, "PlannedTrial", versioned=False)
        out = cls(r.raw("trial_id"), r.raw("arm"), r.raw("assignment_id"), r.raw("seed"))
        r.done()
        return out


@dataclass(frozen=True)
class FinalAnalysis:
    """The confirmatory plan (``frozen/final_analysis.json``), frozen before any held-out or test score exists.

    Primary comparison: arm 3 minus arm 2 on ``heldout_metric`` over the weighted held-out assignments. Secondary: ``test_metric`` on the
    IFBench test set for arms 1, 2 and 3. Both arms have the same number of planned trials in every assignment (budget matched), at
    least ``MIN_FINAL_TRIALS_PER_PACK`` per pack in total. ``method`` is BLOCKED_T with two or more assignments, otherwise WELCH.
    """

    heldout_metric: str
    test_metric: str
    assignments: tuple
    planned_trials: tuple
    confidence: float
    method: ComparisonMethod
    score_seeds: tuple
    score_params: GenerationParams
    failed_trial_policy: FailedTrialPolicy
    retry_policy: str
    budget_sha256: str
    frozen_utc: str
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)
    # EXPLORATORY: an owner-approved small comparison below the confirmatory floor (LIVE 2026-11-04, Tier B "2 + 2 on held-out"); the arms
    # are scored on the held-out selection split only and every report states that the result is exploratory, not confirmatory
    exploratory: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        if self.heldout_metric not in METRICS_BY_SPLIT[Split.SELECTION]:
            raise ContractError(f"FinalAnalysis.heldout_metric: {self.heldout_metric!r} is not a selection metric")
        if self.test_metric not in METRICS_BY_SPLIT[Split.TEST]:
            raise ContractError(f"FinalAnalysis.test_metric: {self.test_metric!r} is not a test metric")
        ws = _tuple_of(self.assignments, "FinalAnalysis.assignments", lambda v, w: _obj(v, AssignmentWeight, w), nonempty=True)
        ids = [w.assignment_id for w in ws]
        if len(set(ids)) != len(ids):
            raise ContractError("FinalAnalysis.assignments: duplicate assignment id")
        if abs(math.fsum(w.weight for w in ws) - 1.0) > 1e-9:
            raise ContractError("FinalAnalysis.assignments: weights must sum to 1")
        _set(self, "assignments", ws)
        planned = _tuple_of(self.planned_trials, "FinalAnalysis.planned_trials", lambda v, w: _obj(v, PlannedTrial, w), nonempty=True)
        if len({t.trial_id for t in planned}) != len(planned):
            raise ContractError("FinalAnalysis.planned_trials: duplicate trial id")
        cells: dict = {}
        for t in planned:
            cells.setdefault((t.arm, t.assignment_id), []).append(t)
        expected = {(a, i) for a in (Arm.V0, Arm.EVOLVED) for i in ids}
        if set(cells) != expected:
            raise ContractError("FinalAnalysis.planned_trials: every assignment needs trials for both v0 and evolved, and no others")
        if len({len(v) for v in cells.values()}) != 1:
            raise ContractError("FinalAnalysis.planned_trials: both arms need the same number of trials in every assignment")
        n = len(next(iter(cells.values())))
        if not isinstance(self.exploratory, bool):
            raise ContractError("FinalAnalysis.exploratory must be a boolean")
        if n * len(ids) < (2 if self.exploratory else MIN_FINAL_TRIALS_PER_PACK):
            raise ContractError(f"FinalAnalysis: at least {MIN_FINAL_TRIALS_PER_PACK} trials per pack are required, got {n * len(ids)}")
        for v in cells.values():
            if len({t.seed for t in v}) != len(v):
                raise ContractError("FinalAnalysis.planned_trials: seeds must differ within an (arm, assignment) cell")
        _set(self, "planned_trials", planned)
        _set(self, "confidence", _num(self.confidence, "FinalAnalysis.confidence"))
        if not 0.5 < self.confidence < 1.0:
            raise ContractError("FinalAnalysis.confidence: the two-sided level must be in (0.5, 1)")
        _set(self, "method", _enum(self.method, ComparisonMethod, "FinalAnalysis.method"))
        if (self.method is ComparisonMethod.BLOCKED_T) != (len(ids) >= 2):
            raise ContractError("FinalAnalysis.method: blocked_t needs two or more held-out assignments, otherwise welch")
        _set(self, "score_seeds", _tuple_of(self.score_seeds, "FinalAnalysis.score_seeds", lambda v, w: _int(v, w, minimum=0), nonempty=True, unique=True))
        _obj(self.score_params, GenerationParams, "FinalAnalysis.score_params")
        _set(self, "failed_trial_policy", _enum(self.failed_trial_policy, FailedTrialPolicy, "FinalAnalysis.failed_trial_policy"))
        _text(self.retry_policy, "FinalAnalysis.retry_policy")
        _sha256(self.budget_sha256, "FinalAnalysis.budget_sha256")
        _utc(self.frozen_utc, "FinalAnalysis.frozen_utc")

    def trials_per_pack_per_assignment(self) -> int:
        return len(self.planned_trials) // (2 * len(self.assignments))

    def planned(self, trial_id: str) -> PlannedTrial:
        for t in self.planned_trials:
            if t.trial_id == trial_id:
                return t
        raise KeyError(trial_id)

    def sha256(self) -> str:
        return record_sha256(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "heldout_metric": self.heldout_metric,
            "test_metric": self.test_metric,
            "assignments": [w.to_dict() for w in self.assignments],
            "planned_trials": [t.to_dict() for t in self.planned_trials],
            "confidence": self.confidence,
            "method": self.method.value,
            "score_seeds": list(self.score_seeds),
            "score_params": self.score_params.to_dict(),
            "failed_trial_policy": self.failed_trial_policy.value,
            "retry_policy": self.retry_policy,
            "budget_sha256": self.budget_sha256,
            "frozen_utc": self.frozen_utc,
            **({"exploratory": True} if self.exploratory else {}),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "FinalAnalysis":
        r = _Reader(d, "FinalAnalysis")
        out = cls(
            r.raw("heldout_metric"),
            r.raw("test_metric"),
            r.seq("assignments", AssignmentWeight.from_dict),
            r.seq("planned_trials", PlannedTrial.from_dict),
            r.raw("confidence"),
            r.raw("method"),
            r.raw("score_seeds"),
            r.obj("score_params", GenerationParams.from_dict),
            r.raw("failed_trial_policy"),
            r.raw("retry_policy"),
            r.raw("budget_sha256"),
            r.raw("frozen_utc"),
            exploratory=(r.raw("exploratory") if "exploratory" in d else False),
        )
        r.done()
        return out


@dataclass(frozen=True)
class SearchClosed:
    """Payload of ``lab.search_closed``: the search is over, the incumbent is the evolved pack, no search trial is pending. Final scoring
    cannot start before it (and not before ``lab.final_analysis_frozen``)."""

    incumbent_version: str
    incumbent: PackRef
    n_candidates: int
    n_search_trials: int
    reason: str
    closed_utc: str
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.incumbent_version, "SearchClosed.incumbent_version")
        _obj(self.incumbent, PackRef, "SearchClosed.incumbent")
        _int(self.n_candidates, "SearchClosed.n_candidates", minimum=0)
        _int(self.n_search_trials, "SearchClosed.n_search_trials", minimum=0)
        _text(self.reason, "SearchClosed.reason")
        _utc(self.closed_utc, "SearchClosed.closed_utc")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "incumbent_version": self.incumbent_version,
            "incumbent": self.incumbent.to_dict(),
            "n_candidates": self.n_candidates,
            "n_search_trials": self.n_search_trials,
            "reason": self.reason,
            "closed_utc": self.closed_utc,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "SearchClosed":
        r = _Reader(d, "SearchClosed")
        out = cls(r.raw("incumbent_version"), r.obj("incumbent", PackRef.from_dict), r.raw("n_candidates"), r.raw("n_search_trials"), r.raw("reason"), r.raw("closed_utc"))
        r.done()
        return out


@dataclass(frozen=True)
class FinalReveal:
    """Payload of ``lab.final_revealed``: appended by the ``reveal`` command after the WHOLE plan has completed. Until then report and
    status print no per-trial held-out or test score, so trial counts cannot be adapted to results."""

    analysis_sha256: str
    n_scores: int
    revealed_utc: str
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _sha256(self.analysis_sha256, "FinalReveal.analysis_sha256")
        _int(self.n_scores, "FinalReveal.n_scores", minimum=1)
        _utc(self.revealed_utc, "FinalReveal.revealed_utc")

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version, "analysis_sha256": self.analysis_sha256, "n_scores": self.n_scores, "revealed_utc": self.revealed_utc}

    @classmethod
    def from_dict(cls, d: Any) -> "FinalReveal":
        r = _Reader(d, "FinalReveal")
        out = cls(r.raw("analysis_sha256"), r.raw("n_scores"), r.raw("revealed_utc"))
        r.done()
        return out


@dataclass(frozen=True)
class ScoreFailure:
    """Payload of ``lab.score_failed``: a scoring attempt ended without a result. A second request for the same adapter, split and
    assignment is allowed only after one of these with ``result_revealed`` False (no number was seen), so a score cannot be re-rolled."""

    request_id: str
    split: Split
    assignment_id: str
    trial_id: Optional[str]
    adapter_sha256: Optional[str]
    failure: str
    result_revealed: bool
    recorded_utc: str
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.request_id, "ScoreFailure.request_id")
        _set(self, "split", _enum(self.split, Split, "ScoreFailure.split"))
        _ident(self.assignment_id, "ScoreFailure.assignment_id")
        if (self.split is Split.TEST) != (self.assignment_id == TEST_ASSIGNMENT_ID):
            raise ContractError("ScoreFailure: the test split, and only the test split, uses the test assignment id")
        _opt(self.trial_id, _ident, "ScoreFailure.trial_id")
        _opt(self.adapter_sha256, _sha256, "ScoreFailure.adapter_sha256")
        _text(self.failure, "ScoreFailure.failure")
        _bool(self.result_revealed, "ScoreFailure.result_revealed")
        _utc(self.recorded_utc, "ScoreFailure.recorded_utc")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "split": self.split.value,
            "assignment_id": self.assignment_id,
            "trial_id": self.trial_id,
            "adapter_sha256": self.adapter_sha256,
            "failure": self.failure,
            "result_revealed": self.result_revealed,
            "recorded_utc": self.recorded_utc,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "ScoreFailure":
        r = _Reader(d, "ScoreFailure")
        out = cls(r.raw("request_id"), r.raw("split"), r.raw("assignment_id"), r.raw("trial_id"), r.raw("adapter_sha256"), r.raw("failure"), r.raw("result_revealed"), r.raw("recorded_utc"))
        r.done()
        return out


@dataclass(frozen=True)
class ContaminationReport:
    """Payload of ``lab.contamination_scanned``: the control host scanned the processed training data of a job (and the recipe tarball)
    against the selection, held-out and test prompts, by exact prompt match and by ``ngram_n``-gram match. A hit makes the trial
    CONTAMINATED, which counts as a failed trial."""

    trial_id: str
    job_id: str
    scanned: FileRef
    n_rows: int
    n_exact_hits: int
    n_ngram_hits: int
    ngram_n: int
    reference_sha256: tuple
    passed: bool
    recorded_utc: str
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.trial_id, "ContaminationReport.trial_id")
        _ident(self.job_id, "ContaminationReport.job_id")
        _obj(self.scanned, FileRef, "ContaminationReport.scanned")
        _int(self.n_rows, "ContaminationReport.n_rows", minimum=0)
        _int(self.n_exact_hits, "ContaminationReport.n_exact_hits", minimum=0)
        _int(self.n_ngram_hits, "ContaminationReport.n_ngram_hits", minimum=0)
        if self.ngram_n != CONTAMINATION_NGRAM:
            raise ContractError(f"ContaminationReport.ngram_n: must be {CONTAMINATION_NGRAM}")
        _set(self, "reference_sha256", _tuple_of(self.reference_sha256, "ContaminationReport.reference_sha256", _sha256, nonempty=True))
        _bool(self.passed, "ContaminationReport.passed")
        if self.passed != (self.n_exact_hits == 0 and self.n_ngram_hits == 0):
            raise ContractError("ContaminationReport: passed must be true exactly when there are no hits")
        _utc(self.recorded_utc, "ContaminationReport.recorded_utc")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "trial_id": self.trial_id,
            "job_id": self.job_id,
            "scanned": self.scanned.to_dict(),
            "n_rows": self.n_rows,
            "n_exact_hits": self.n_exact_hits,
            "n_ngram_hits": self.n_ngram_hits,
            "ngram_n": self.ngram_n,
            "reference_sha256": list(self.reference_sha256),
            "passed": self.passed,
            "recorded_utc": self.recorded_utc,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "ContaminationReport":
        r = _Reader(d, "ContaminationReport")
        out = cls(
            r.raw("trial_id"),
            r.raw("job_id"),
            r.obj("scanned", FileRef.from_dict),
            r.raw("n_rows"),
            r.raw("n_exact_hits"),
            r.raw("n_ngram_hits"),
            r.raw("ngram_n"),
            r.raw("reference_sha256"),
            r.raw("passed"),
            r.raw("recorded_utc"),
        )
        r.done()
        return out


@dataclass(frozen=True)
class OperatorChange:
    """Payload of ``lab.operator_change``: any change by the operator after the first candidate was proposed (pack lint ranges, rule
    inputs, proposer prompt, held-out families, v0 notes). The operator is an adaptive channel too; this makes it visible."""

    target: str
    detail: str
    candidates_so_far: int
    recorded_utc: str
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _text(self.target, "OperatorChange.target", max_bytes=100)
        _text(self.detail, "OperatorChange.detail")
        _int(self.candidates_so_far, "OperatorChange.candidates_so_far", minimum=0)
        _utc(self.recorded_utc, "OperatorChange.recorded_utc")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "target": self.target,
            "detail": self.detail,
            "candidates_so_far": self.candidates_so_far,
            "recorded_utc": self.recorded_utc,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "OperatorChange":
        r = _Reader(d, "OperatorChange")
        out = cls(r.raw("target"), r.raw("detail"), r.raw("candidates_so_far"), r.raw("recorded_utc"))
        r.done()
        return out


@dataclass(frozen=True)
class LintResult:
    """What ``PackStore.lint`` returns (an exchange type; the messages are copied into CandidateRecord)."""

    passed: bool
    messages: tuple
    changed_paths: tuple

    def __post_init__(self) -> None:
        _bool(self.passed, "LintResult.passed")
        _set(self, "messages", _tuple_of(self.messages, "LintResult.messages", lambda v, w: _text(v, w, max_bytes=500)))
        _set(self, "changed_paths", _tuple_of(self.changed_paths, "LintResult.changed_paths", _relpath))
        if not self.passed and not self.messages:
            raise ContractError("LintResult: a failed lint explains itself")


@dataclass(frozen=True)
class FinalComparison:
    """Result of the confirmatory contrast (``reports/final_comparison.json``), made by ``stats.compare_final`` from a FinalAnalysis
    and the planned trials' scores. ``contrast`` is always ``evolved_minus_v0``; ``df`` is the t degrees of freedom (Welch-Satterthwaite
    for WELCH, floored)."""

    analysis_sha256: str
    split: Split
    metric: str
    contrast: str
    method: ComparisonMethod
    delta: float
    ci_low: float
    ci_high: float
    level: float
    df: float
    n_assignments: int
    trials_v0: int
    trials_evolved: int
    mean_v0: float
    mean_evolved: float
    includes_zero: bool
    note: str
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _sha256(self.analysis_sha256, "FinalComparison.analysis_sha256")
        _set(self, "split", _enum(self.split, Split, "FinalComparison.split"))
        if self.split is Split.DEV:
            raise ContractError("FinalComparison.split: dev is not a final split")
        _text(self.metric, "FinalComparison.metric", max_bytes=100)
        _text(self.contrast, "FinalComparison.contrast", max_bytes=100)
        _set(self, "method", _enum(self.method, ComparisonMethod, "FinalComparison.method"))
        for name in ("delta", "ci_low", "ci_high", "mean_v0", "mean_evolved"):
            _set(self, name, _num(getattr(self, name), f"FinalComparison.{name}"))
        if not self.ci_low <= self.delta <= self.ci_high:
            raise ContractError("FinalComparison: the interval must contain the estimate")
        _set(self, "level", _num(self.level, "FinalComparison.level"))
        _set(self, "df", _pos(self.df, "FinalComparison.df"))
        _int(self.n_assignments, "FinalComparison.n_assignments", minimum=1)
        _int(self.trials_v0, "FinalComparison.trials_v0", minimum=1)
        _int(self.trials_evolved, "FinalComparison.trials_evolved", minimum=1)
        _bool(self.includes_zero, "FinalComparison.includes_zero")
        if self.includes_zero != (self.ci_low <= 0.0 <= self.ci_high):
            raise ContractError("FinalComparison.includes_zero contradicts the interval")
        _text(self.note, "FinalComparison.note")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "analysis_sha256": self.analysis_sha256,
            "split": self.split.value,
            "metric": self.metric,
            "contrast": self.contrast,
            "method": self.method.value,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "level": self.level,
            "df": self.df,
            "n_assignments": self.n_assignments,
            "trials_v0": self.trials_v0,
            "trials_evolved": self.trials_evolved,
            "mean_v0": self.mean_v0,
            "mean_evolved": self.mean_evolved,
            "includes_zero": self.includes_zero,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "FinalComparison":
        r = _Reader(d, "FinalComparison")
        out = cls(
            r.raw("analysis_sha256"),
            r.raw("split"),
            r.raw("metric"),
            r.raw("contrast"),
            r.raw("method"),
            r.raw("delta"),
            r.raw("ci_low"),
            r.raw("ci_high"),
            r.raw("level"),
            r.raw("df"),
            r.raw("n_assignments"),
            r.raw("trials_v0"),
            r.raw("trials_evolved"),
            r.raw("mean_v0"),
            r.raw("mean_evolved"),
            r.raw("includes_zero"),
            r.raw("note"),
        )
        r.done()
        return out


_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


@dataclass(frozen=True)
class LabConfig:
    """``<lab>/lab.json``: hosts, pins and the NAMES of secret environment variables (never their values). Only the CLI reads the file;
    it hands each component its slice. ``lab_budget_usd`` caps researcher inference plus Wonder across the whole lab;
    ``affordable_trials`` is the total number of trials the operator can pay for."""

    lab_id: str
    control_host: str
    researcher_host: str
    researcher_user: str
    base: BaseModelRef
    shape: "Shape"
    pins: dict
    secret_env: dict
    upstream_base_url: str
    lab_budget_usd: float
    affordable_trials: int
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.lab_id, "LabConfig.lab_id")
        for name in ("control_host", "researcher_host", "researcher_user", "upstream_base_url"):
            _text(getattr(self, name), f"LabConfig.{name}", max_bytes=300)
        _obj(self.base, BaseModelRef, "LabConfig.base")
        _obj(self.shape, Shape, "LabConfig.shape")
        pins = _str_map(self.pins, "LabConfig.pins")
        missing = [k for k in CONFIG_PIN_KEYS if k not in pins]
        if missing:
            raise ContractError(f"LabConfig.pins: missing {missing}")
        _set(self, "pins", pins)
        env = _str_map(self.secret_env, "LabConfig.secret_env")
        missing = [k for k in CONFIG_SECRET_NAMES if k not in env]
        if missing:
            raise ContractError(f"LabConfig.secret_env: missing {missing}")
        for k, v in env.items():
            if not _ENV_NAME.match(v):
                raise ContractError(f"LabConfig.secret_env[{k!r}]: {v!r} is not an environment variable name")
        _set(self, "secret_env", env)
        _set(self, "lab_budget_usd", _pos(self.lab_budget_usd, "LabConfig.lab_budget_usd"))
        _int(self.affordable_trials, "LabConfig.affordable_trials", minimum=1)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "lab_id": self.lab_id,
            "control_host": self.control_host,
            "researcher_host": self.researcher_host,
            "researcher_user": self.researcher_user,
            "base": self.base.to_dict(),
            "shape": self.shape.to_dict(),
            "pins": dict(self.pins),
            "secret_env": dict(self.secret_env),
            "upstream_base_url": self.upstream_base_url,
            "lab_budget_usd": self.lab_budget_usd,
            "affordable_trials": self.affordable_trials,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "LabConfig":
        r = _Reader(d, "LabConfig")
        out = cls(
            r.raw("lab_id"),
            r.raw("control_host"),
            r.raw("researcher_host"),
            r.raw("researcher_user"),
            r.obj("base", BaseModelRef.from_dict),
            r.obj("shape", Shape.from_dict),
            r.raw("pins"),
            r.raw("secret_env"),
            r.raw("upstream_base_url"),
            r.raw("lab_budget_usd"),
            r.raw("affordable_trials"),
        )
        r.done()
        return out


def load_frozen(layout: "LabLayout", frozen: FrozenFile, cls: type) -> Any:
    """Load a frozen file and verify it against the ledger's FileRef (sha256 of the BYTES written, and size) before parsing.

    Two different hashes exist and must not be compared with each other: ``FrozenFile.file.sha256`` is the hash of the file bytes
    (what this function checks); ``Decision.rule_sha256`` is ``Rule.sha256()``, the hash of the canonical JSON of the parsed record.
    """
    data = read_regular_bytes(layout.root, frozen.file.path, max_bytes=1 << 24)
    if hashlib.sha256(data).hexdigest() != frozen.file.sha256 or len(data) != frozen.file.size_bytes:
        raise ContractError(f"frozen file {frozen.file.path} does not match the hash recorded in the ledger")
    return cls.from_dict(loads_strict(data.decode("utf-8")))


def validate_rule_against_manifest(rule: Rule, manifest: AssignmentManifest) -> None:
    """Every assignment the rule judges candidates on must be an EVOLUTION assignment."""
    for a in rule.assignments:
        try:
            role = manifest.get(a).role
        except KeyError:
            raise ContractError(f"rule assignment {a!r} is not in the manifest") from None
        if role is not AssignmentRole.EVOLUTION:
            raise ContractError(f"rule assignment {a!r} is held out; the rule may only use evolution assignments")


def validate_analysis_against_manifest(analysis: FinalAnalysis, manifest: AssignmentManifest) -> None:
    """Every assignment of the final analysis must be a HELDOUT assignment."""
    for w in analysis.assignments:
        try:
            role = manifest.get(w.assignment_id).role
        except KeyError:
            raise ContractError(f"final assignment {w.assignment_id!r} is not in the manifest") from None
        if role is not AssignmentRole.HELDOUT:
            raise ContractError(f"final assignment {w.assignment_id!r} is an evolution assignment; the final comparison uses held-out assignments only")


def check_trial_against_frozen(
    spec: TrialSpec,
    *,
    manifest: AssignmentManifest,
    budget: BudgetSpec,
    rule: Optional[Rule] = None,
    analysis: Optional[FinalAnalysis] = None,
) -> None:
    """The trial command and the trial runner call this before anything is spent. Raises ContractError on the first mismatch:
    search-phase trials run on evolution assignments, final-phase trials on held-out ones; the spec carries the manifest's data
    revisions and exactly the frozen budget; with a rule, the scorer version, score seeds and decoding are the rule's; a final-phase
    trial is one of the analysis's planned trials (same arm, assignment, seed)."""
    try:
        assignment = manifest.get(spec.assignment_id)
    except KeyError:
        raise ContractError(f"trial assignment {spec.assignment_id!r} is not in the manifest") from None
    want = AssignmentRole.EVOLUTION if spec.phase is Phase.SEARCH else AssignmentRole.HELDOUT
    if assignment.role is not want:
        raise ContractError(f"a {spec.phase.value}-phase trial runs on {want.value} assignments, not {assignment.assignment_id!r} ({assignment.role.value})")
    if spec.data_revisions != assignment.data_revisions:
        raise ContractError("TrialSpec.data_revisions differ from the manifest's revisions of the assignment")
    if spec.budget != budget:
        raise ContractError("TrialSpec.budget is not the frozen budget")
    if spec.phase is Phase.SEARCH and rule is not None:
        if (spec.scorer_version, spec.score_seeds, spec.score_params) != (rule.scorer_version, rule.score_seeds, rule.score_params):
            raise ContractError("TrialSpec scorer version, score seeds or decoding differ from the frozen rule")
    if spec.phase is Phase.FINAL and analysis is not None:
        try:
            planned = analysis.planned(spec.trial_id)
        except KeyError:
            raise ContractError(f"trial {spec.trial_id!r} is not in the frozen final analysis") from None
        if (planned.arm, planned.assignment_id, planned.seed) != (spec.arm, spec.assignment_id, spec.seed):
            raise ContractError(f"trial {spec.trial_id!r} differs from its planned arm, assignment or seed")
        if (spec.score_seeds, spec.score_params) != (analysis.score_seeds, analysis.score_params):
            raise ContractError("TrialSpec score seeds or decoding differ from the frozen final analysis")
        if record_sha256(spec.budget) != analysis.budget_sha256:
            raise ContractError("TrialSpec.budget differs from the budget the final analysis froze")


# --------------------------------------------------------------------------------------------------
# GPU, trainer, serving
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Shape:
    """A rentable machine shape. Price and availability are the provider's (``GpuProvider.quote``)."""

    gpu: str
    num_gpus: int
    vcpus: int
    disk_gb: int
    template: str = "base"

    def __post_init__(self) -> None:
        _text(self.gpu, "Shape.gpu", max_bytes=50)
        _int(self.num_gpus, "Shape.num_gpus", minimum=1)
        _int(self.vcpus, "Shape.vcpus", minimum=1)
        _int(self.disk_gb, "Shape.disk_gb", minimum=1)
        _text(self.template, "Shape.template", max_bytes=100)

    def to_dict(self) -> dict:
        return {"gpu": self.gpu, "num_gpus": self.num_gpus, "vcpus": self.vcpus, "disk_gb": self.disk_gb, "template": self.template}

    @classmethod
    def from_dict(cls, d: Any) -> "Shape":
        r = _Reader(d, "Shape", versioned=False)
        out = cls(r.raw("gpu"), r.raw("num_gpus"), r.raw("vcpus"), r.raw("disk_gb"), r.raw("template"))
        r.done()
        return out


@dataclass(frozen=True)
class Quote:
    """Provider price for a shape. ``available`` is None when the provider cannot say."""

    usd_per_hour: float
    fixed_usd: float
    available: Optional[bool]
    source: str

    def __post_init__(self) -> None:
        _set(self, "usd_per_hour", _nonneg(self.usd_per_hour, "Quote.usd_per_hour"))
        _set(self, "fixed_usd", _nonneg(self.fixed_usd, "Quote.fixed_usd"))
        _opt(self.available, _bool, "Quote.available")
        _text(self.source, "Quote.source", max_bytes=200)


@dataclass(frozen=True)
class GpuInstance:
    """One row of the provider's account listing. ``handle`` is the provider's stable id, never a list position. ``owner_tag`` is
    ``owner_tag_of_key(public key)`` for a rental we created (the provider derives it from the key; there is no separate tag)."""

    handle: str
    status: str
    shape: Optional[Shape]
    created_epoch: float
    owner_tag: Optional[str]
    host: Optional[str] = None
    port: Optional[int] = None
    user: str = "ubuntu"


@dataclass(frozen=True)
class CreateReceipt:
    handle: Optional[str]
    attempt_epoch: float


@dataclass(frozen=True)
class GpuUsage:
    """Billed GPU use, from the rental ledger: minutes are rounded up per rental."""

    minutes: float
    usd: float
    rental_ids: tuple

    def __post_init__(self) -> None:
        _set(self, "minutes", _nonneg(self.minutes, "GpuUsage.minutes"))
        _set(self, "usd", _nonneg(self.usd, "GpuUsage.usd"))
        _set(self, "rental_ids", _tuple_of(self.rental_ids, "GpuUsage.rental_ids", _ident))

    def to_dict(self) -> dict:
        return {"minutes": self.minutes, "usd": self.usd, "rental_ids": list(self.rental_ids)}

    @classmethod
    def from_dict(cls, d: Any) -> "GpuUsage":
        r = _Reader(d, "GpuUsage", versioned=False)
        out = cls(r.raw("minutes"), r.raw("usd"), r.raw("rental_ids"))
        r.done()
        return out


@dataclass(frozen=True)
class TrainJob:
    """Payload of ``lab.gpu_job_started``. The recipe is the researcher's code; the trainer provides the GPU and the stack.

    ``dev_request`` is a template: arm None, adapter None, the trial's phase (and final_request_id), score seeds and decoding. ``shape``
    equals ``TrialSpec.shape``.
    """

    job_id: str
    trial_id: str
    recipe: FileRef
    entrypoint: tuple
    max_gpu_minutes: float
    shape: Shape
    dev_request: ScoreRequest
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.job_id, "TrainJob.job_id")
        _ident(self.trial_id, "TrainJob.trial_id")
        _obj(self.recipe, FileRef, "TrainJob.recipe")
        _set(self, "entrypoint", _tuple_of(self.entrypoint, "TrainJob.entrypoint", lambda v, w: _text(v, w, max_bytes=500), nonempty=True))
        _set(self, "max_gpu_minutes", _pos(self.max_gpu_minutes, "TrainJob.max_gpu_minutes"))
        _obj(self.shape, Shape, "TrainJob.shape")
        _obj(self.dev_request, ScoreRequest, "TrainJob.dev_request")
        if self.dev_request.split is not Split.DEV:
            raise ContractError("TrainJob.dev_request must be a dev-split request")
        if self.dev_request.arm is not None or self.dev_request.adapter is not None:
            raise ContractError("TrainJob.dev_request is a template: arm and adapter are null (the trainer fills the adapter with with_adapter)")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "trial_id": self.trial_id,
            "recipe": self.recipe.to_dict(),
            "entrypoint": list(self.entrypoint),
            "max_gpu_minutes": self.max_gpu_minutes,
            "shape": self.shape.to_dict(),
            "dev_request": self.dev_request.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "TrainJob":
        r = _Reader(d, "TrainJob")
        out = cls(
            r.raw("job_id"),
            r.raw("trial_id"),
            r.obj("recipe", FileRef.from_dict),
            r.raw("entrypoint"),
            r.raw("max_gpu_minutes"),
            r.obj("shape", Shape.from_dict),
            r.obj("dev_request", ScoreRequest.from_dict),
        )
        r.done()
        return out


@dataclass(frozen=True)
class TrainResult:
    """Payload of ``lab.gpu_job_finished``. ``timings`` keys are a subset of ``TIMING_KEYS`` (seconds). ``train_data`` is the processed
    training dataset the glimmer helper wrote into the job output; an OK job has one (it is what the contamination scan reads).
    ``dev_score_id`` names a ScoreRecord the scorer already wrote under ``scores/<id>/``."""

    job_id: str
    trial_id: str
    status: JobStatus
    adapter_id: Optional[str]
    adapter: Optional[AdapterRef]
    dev_score_id: Optional[str]
    gpu: GpuUsage
    timings: dict
    log: Optional[FileRef]
    train_data: Optional[FileRef]
    failure: Optional[str]
    schema_version: int = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _check_version(self)
        _ident(self.job_id, "TrainResult.job_id")
        _ident(self.trial_id, "TrainResult.trial_id")
        _set(self, "status", _enum(self.status, JobStatus, "TrainResult.status"))
        _opt(self.adapter_id, _ident, "TrainResult.adapter_id")
        _opt(self.adapter, lambda v, w: _obj(v, AdapterRef, w), "TrainResult.adapter")
        _opt(self.dev_score_id, _ident, "TrainResult.dev_score_id")
        _obj(self.gpu, GpuUsage, "TrainResult.gpu")
        timings = _str_map(self.timings, "TrainResult.timings", value=_nonneg)
        extra = sorted(set(timings) - set(TIMING_KEYS))
        if extra:
            raise ContractError(f"TrainResult.timings: unknown key(s) {extra}; allowed {TIMING_KEYS}")
        _set(self, "timings", timings)
        _opt(self.log, lambda v, w: _obj(v, FileRef, w), "TrainResult.log")
        _opt(self.train_data, lambda v, w: _obj(v, FileRef, w), "TrainResult.train_data")
        _set(self, "failure", _opt(self.failure, _text, "TrainResult.failure"))
        if self.status is JobStatus.OK:
            if self.adapter is None or self.adapter_id is None or self.dev_score_id is None or self.train_data is None:
                raise ContractError("TrainResult: an ok job returns adapter_id, adapter, dev_score_id and train_data")
            if self.adapter_id != self.adapter.adapter_id:
                raise ContractError("TrainResult: adapter_id does not match the adapter's id")
        elif self.failure is None:
            raise ContractError("TrainResult: a job that did not succeed needs a failure text")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "trial_id": self.trial_id,
            "status": self.status.value,
            "adapter_id": self.adapter_id,
            "adapter": None if self.adapter is None else self.adapter.to_dict(),
            "dev_score_id": self.dev_score_id,
            "gpu": self.gpu.to_dict(),
            "timings": dict(self.timings),
            "log": None if self.log is None else self.log.to_dict(),
            "train_data": None if self.train_data is None else self.train_data.to_dict(),
            "failure": self.failure,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "TrainResult":
        r = _Reader(d, "TrainResult")
        out = cls(
            r.raw("job_id"),
            r.raw("trial_id"),
            r.raw("status"),
            r.raw("adapter_id"),
            r.opt("adapter", AdapterRef.from_dict),
            r.raw("dev_score_id"),
            r.obj("gpu", GpuUsage.from_dict),
            r.raw("timings"),
            r.opt("log", FileRef.from_dict),
            r.opt("train_data", FileRef.from_dict),
            r.raw("failure"),
        )
        r.done()
        return out


@dataclass(frozen=True)
class ServeSpec:
    """Ask a ModelServer for an OpenAI-compatible endpoint serving the base model and zero or more adapters."""

    base: BaseModelRef
    adapters: tuple
    max_model_len: int
    max_gpu_minutes: float
    purpose: str
    trial_id: Optional[str] = None

    def __post_init__(self) -> None:
        _obj(self.base, BaseModelRef, "ServeSpec.base")
        _set(self, "adapters", _tuple_of(self.adapters, "ServeSpec.adapters", lambda v, w: _obj(v, AdapterRef, w)))
        _int(self.max_model_len, "ServeSpec.max_model_len", minimum=1)
        _set(self, "max_gpu_minutes", _pos(self.max_gpu_minutes, "ServeSpec.max_gpu_minutes"))
        _text(self.purpose, "ServeSpec.purpose", max_bytes=200)
        _opt(self.trial_id, _ident, "ServeSpec.trial_id")


@dataclass(frozen=True)
class ServeEndpoint:
    """A live endpoint. ``adapter_names`` maps an adapter's listing sha256 to the model name to send in requests. ``base_url`` has NO
    ``/v1`` suffix (a generator appends ``/v1/chat/completions``)."""

    base_url: str
    base_name: str
    adapter_names: dict

    def __post_init__(self) -> None:
        _text(self.base_url, "ServeEndpoint.base_url", max_bytes=500)
        _text(self.base_name, "ServeEndpoint.base_name", max_bytes=200)
        _set(self, "adapter_names", _str_map(self.adapter_names, "ServeEndpoint.adapter_names"))

    def model_for(self, adapter: Optional[AdapterRef]) -> str:
        if adapter is None:
            return self.base_name
        try:
            return self.adapter_names[adapter.sha256]
        except KeyError:
            raise ContractError(f"endpoint does not serve adapter {adapter.sha256}") from None


# --------------------------------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GenPrompt:
    prompt_id: str
    text: str

    def __post_init__(self) -> None:
        _text(self.prompt_id, "GenPrompt.prompt_id", max_bytes=200)
        _text(self.text, "GenPrompt.text", empty=True, max_bytes=1 << 22)


@dataclass(frozen=True)
class GenResult:
    """One generation. ``text`` is the final answer only (never reasoning). A per-prompt failure sets ``error`` and empty text."""

    prompt_id: str
    text: str
    finish_reason: str
    completion_tokens: int
    error: Optional[str] = None

    def __post_init__(self) -> None:
        _text(self.prompt_id, "GenResult.prompt_id", max_bytes=200)
        _text(self.text, "GenResult.text", empty=True, max_bytes=1 << 22)
        _text(self.finish_reason, "GenResult.finish_reason", empty=True, max_bytes=50)
        _int(self.completion_tokens, "GenResult.completion_tokens", minimum=0)
        _opt(self.error, lambda v, w: _text(v, w, empty=True, max_bytes=1000), "GenResult.error")

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


# --------------------------------------------------------------------------------------------------
# researcher runtime and proposals (in-memory exchange types)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TrialContext:
    """What the trial runner hands the researcher runtime. Not persisted. ``llm_token`` and ``control_endpoint`` are trial-scoped
    secrets and are excluded from ``repr`` so a log line or traceback cannot print them.

    ``llm_base_url`` and ``control_endpoint`` are control-host loopback URLs served on ONE port, and ``llm_token`` is the single per-trial
    token for both the metering proxy and the control API; the runtime rewrites the URLs for the researcher side (an ssh reverse tunnel)."""

    spec: TrialSpec
    trial_dir: str
    llm_base_url: str
    llm_token: str = field(repr=False)
    control_endpoint: str = field(repr=False)

    def __post_init__(self) -> None:
        _obj(self.spec, TrialSpec, "TrialContext.spec")
        for name in ("trial_dir", "llm_base_url", "llm_token", "control_endpoint"):
            _text(getattr(self, name), f"TrialContext.{name}", max_bytes=1024)


@dataclass(frozen=True)
class SubmittedAttempt:
    """One attempt that left a ``submission.json`` and the time CORAL committed it. The trial runner (not the researcher) picks the
    submission to score: the attempt whose adapter has the highest control-host dev score, ties to the earliest ``committed_utc``."""

    submission: Submission
    committed_utc: str

    def __post_init__(self) -> None:
        _obj(self.submission, Submission, "SubmittedAttempt.submission")
        _utc(self.committed_utc, "SubmittedAttempt.committed_utc")


@dataclass(frozen=True)
class RuntimeResult:
    exit_reason: ExitReason
    attempts: int
    submissions: tuple
    trace: Optional[FileRef]
    notes: Optional[FileRef]
    initial_context_sha256: str
    environment: dict

    def __post_init__(self) -> None:
        _set(self, "exit_reason", _enum(self.exit_reason, ExitReason, "RuntimeResult.exit_reason"))
        _int(self.attempts, "RuntimeResult.attempts", minimum=0)
        _set(self, "submissions", _tuple_of(self.submissions, "RuntimeResult.submissions", lambda v, w: _obj(v, SubmittedAttempt, w)))
        _sha256(self.initial_context_sha256, "RuntimeResult.initial_context_sha256")
        _set(self, "environment", _str_map(self.environment, "RuntimeResult.environment", value=lambda v, w: _text(v, w, empty=True, max_bytes=500)))


@dataclass(frozen=True)
class ProposalContext:
    """Everything the proposer sees, already redacted by the outer loop (see docs/harness-design.md, "Outer loop")."""

    candidate_id: str
    incumbent: PackRef
    incumbent_version: str
    system_prompt: str
    user_prompt: str

    def __post_init__(self) -> None:
        _ident(self.candidate_id, "ProposalContext.candidate_id")
        _obj(self.incumbent, PackRef, "ProposalContext.incumbent")
        _ident(self.incumbent_version, "ProposalContext.incumbent_version")
        _text(self.system_prompt, "ProposalContext.system_prompt", max_bytes=1 << 20)
        _text(self.user_prompt, "ProposalContext.user_prompt", max_bytes=1 << 22)


@dataclass(frozen=True)
class ProposalResult:
    proposal: Proposal
    model: str
    served_model: str
    fallback_used: bool
    usd: float
    prompt_tokens: int
    completion_tokens: int
    raw_response: str

    def __post_init__(self) -> None:
        _obj(self.proposal, Proposal, "ProposalResult.proposal")
        _text(self.model, "ProposalResult.model", max_bytes=200)
        _text(self.served_model, "ProposalResult.served_model", max_bytes=200)
        _bool(self.fallback_used, "ProposalResult.fallback_used")
        _set(self, "usd", _nonneg(self.usd, "ProposalResult.usd"))
        _int(self.prompt_tokens, "ProposalResult.prompt_tokens", minimum=0)
        _int(self.completion_tokens, "ProposalResult.completion_tokens", minimum=0)
        _text(self.raw_response, "ProposalResult.raw_response", empty=True, max_bytes=1 << 22)


# --------------------------------------------------------------------------------------------------
# ledger event kinds and payload types
# --------------------------------------------------------------------------------------------------
class LabKind:
    """Kind strings the lab appends to nightshift.ledger.Ledger (which accepts any string). Never ``experiment.proposed``."""

    # master ledger (<lab>/ledger.jsonl). The coordinator is ONE process; every master append goes through one thread-safe LabLedger.
    INITIALIZED = "lab.initialized"
    ASSIGNMENTS_FROZEN = "lab.assignments_frozen"
    BUDGET_FROZEN = "lab.budget_frozen"
    SPREAD_PLANNED = "lab.spread_planned"
    SPREAD_MEASURED = "lab.spread_measured"
    RULE_FROZEN = "lab.rule_frozen"
    GATE_RECORDED = "lab.gate_recorded"
    PACK_COMMITTED = "lab.pack_committed"
    TRIAL_STARTED = "lab.trial_started"
    TRIAL_FINISHED = "lab.trial_finished"
    SCORE_RECORDED = "lab.score_recorded"
    SCORE_COST = "lab.score_cost"
    SCORE_FAILED = "lab.score_failed"
    CONTAMINATION_SCANNED = "lab.contamination_scanned"
    CANDIDATE_PROPOSED = "lab.candidate_proposed"
    CANDIDATE_DECIDED = "lab.candidate_decided"
    OPERATOR_CHANGE = "lab.operator_change"
    SEARCH_CLOSED = "lab.search_closed"
    FINAL_ANALYSIS_FROZEN = "lab.final_analysis_frozen"
    FINAL_REQUESTED = "lab.final_requested"
    FINAL_REVEALED = "lab.final_revealed"
    # per-trial ledger (<lab>/trials/<id>/ledger.jsonl): created and closed inside TrialRunner.run, which executes in a thread of the
    # coordinator process; the thread that runs the trial is not the only writer (proxy, control API and GPU tick threads charge too),
    # so the trial LabLedger is thread-safe as well.
    BUDGET_CHARGED = "lab.budget_charged"
    BUDGET_EXHAUSTED = "lab.budget_exhausted"
    # the worst case of one LLM call, written before the request is forwarded (payload BudgetCharge: resource researcher_tokens, amount
    # the worst-case tokens, detail["usd"] the worst-case dollars); a reservation with no ``<call_id>:tokens`` charge after it is unsettled
    BUDGET_RESERVED = "lab.budget_reserved"
    GPU_JOB_STARTED = "lab.gpu_job_started"
    GPU_JOB_FINISHED = "lab.gpu_job_finished"

    MASTER = (
        INITIALIZED,
        ASSIGNMENTS_FROZEN,
        BUDGET_FROZEN,
        SPREAD_PLANNED,
        SPREAD_MEASURED,
        RULE_FROZEN,
        GATE_RECORDED,
        PACK_COMMITTED,
        TRIAL_STARTED,
        TRIAL_FINISHED,
        SCORE_RECORDED,
        SCORE_COST,
        SCORE_FAILED,
        CONTAMINATION_SCANNED,
        CANDIDATE_PROPOSED,
        CANDIDATE_DECIDED,
        OPERATOR_CHANGE,
        SEARCH_CLOSED,
        FINAL_ANALYSIS_FROZEN,
        FINAL_REQUESTED,
        FINAL_REVEALED,
    )
    TRIAL = (BUDGET_CHARGED, BUDGET_EXHAUSTED, BUDGET_RESERVED, GPU_JOB_STARTED, GPU_JOB_FINISHED)
    # kinds written by the reused BudgetGate / MeteredClient into <lab>/outer/ledger.jsonl (Wonder calls), unchanged
    REUSED = ("budget.charged", "budget.exhausted")
    # kinds that may appear at most once in the master ledger ("first wins"): a second one is malformed and fails verify
    FREEZE_ONCE = (
        ASSIGNMENTS_FROZEN,
        BUDGET_FROZEN,
        SPREAD_PLANNED,
        SPREAD_MEASURED,
        RULE_FROZEN,
        SEARCH_CLOSED,
        FINAL_ANALYSIS_FROZEN,
        FINAL_REVEALED,
    )

    @classmethod
    def all(cls) -> frozenset:
        return frozenset(cls.MASTER) | frozenset(cls.TRIAL)


PAYLOAD_TYPES: dict[str, type] = {
    LabKind.INITIALIZED: LabInit,
    LabKind.ASSIGNMENTS_FROZEN: FrozenFile,
    LabKind.BUDGET_FROZEN: FrozenFile,
    LabKind.SPREAD_PLANNED: SpreadPlan,
    LabKind.SPREAD_MEASURED: SpreadMeasured,
    LabKind.RULE_FROZEN: FrozenFile,
    LabKind.GATE_RECORDED: GateRecord,
    LabKind.PACK_COMMITTED: PackCommitted,
    LabKind.TRIAL_STARTED: TrialSpec,
    LabKind.TRIAL_FINISHED: TrialRecord,
    LabKind.SCORE_RECORDED: ScoreRecord,
    LabKind.SCORE_COST: ScoreCost,
    LabKind.SCORE_FAILED: ScoreFailure,
    LabKind.CONTAMINATION_SCANNED: ContaminationReport,
    LabKind.CANDIDATE_PROPOSED: CandidateRecord,
    LabKind.CANDIDATE_DECIDED: Decision,
    LabKind.OPERATOR_CHANGE: OperatorChange,
    LabKind.SEARCH_CLOSED: SearchClosed,
    LabKind.FINAL_ANALYSIS_FROZEN: FrozenFile,
    LabKind.FINAL_REQUESTED: FinalRequest,
    LabKind.FINAL_REVEALED: FinalReveal,
    LabKind.BUDGET_CHARGED: BudgetCharge,
    LabKind.BUDGET_EXHAUSTED: BudgetExhaustedEvent,
    LabKind.BUDGET_RESERVED: BudgetCharge,
    LabKind.GPU_JOB_STARTED: TrainJob,
    LabKind.GPU_JOB_FINISHED: TrainResult,
}


def payload_from_dict(kind: str, data: Any) -> Any:
    """Parse a ledger event's ``data`` into its contract record. Raises ContractError for unknown kinds or bad payloads."""
    cls = PAYLOAD_TYPES.get(kind)
    if cls is None:
        raise ContractError(f"{kind!r} is not a lab event kind")
    return cls.from_dict(data)


@dataclass
class LabState:
    """The read model: what ``fold.fold_events`` makes of a master ledger. Malformed events are listed, never fatal.

    Keys: ``packs`` by version_label (a label that maps to two commits is malformed); ``trials_started`` and ``trials`` by trial_id;
    ``scores`` and ``score_costs`` by score_id; ``contamination`` by trial_id; ``candidates`` and ``decisions`` by candidate_id.
    ``malformed`` holds ``(seq, reason)``. The frozen files are FileRefs here; ``fold.load_frozen_all`` parses them and re-hashes.
    """

    initialized: Optional[LabInit] = None
    manifest: Optional[FrozenFile] = None
    budget: Optional[FrozenFile] = None
    spread_plan: Optional[SpreadPlan] = None
    spread: Optional[SpreadMeasured] = None
    rule: Optional[FrozenFile] = None
    final_analysis: Optional[FrozenFile] = None
    search_closed: Optional[SearchClosed] = None
    reveal: Optional[FinalReveal] = None
    gates: list[GateRecord] = field(default_factory=list)
    packs: dict[str, PackCommitted] = field(default_factory=dict)
    trials_started: dict[str, TrialSpec] = field(default_factory=dict)
    trials: dict[str, TrialRecord] = field(default_factory=dict)
    scores: dict[str, ScoreRecord] = field(default_factory=dict)
    score_costs: dict[str, ScoreCost] = field(default_factory=dict)
    score_failures: list[ScoreFailure] = field(default_factory=list)
    contamination: dict[str, ContaminationReport] = field(default_factory=dict)
    candidates: dict[str, CandidateRecord] = field(default_factory=dict)
    decisions: dict[str, Decision] = field(default_factory=dict)
    finals: list[FinalRequest] = field(default_factory=list)
    operator_changes: list[OperatorChange] = field(default_factory=list)
    malformed: list[tuple[int, str]] = field(default_factory=list)
    head: Optional[LedgerAnchor] = None
    # ``{(kind, key): seq}``: the sequence number of the first event of each kind and key (key is "" for FREEZE_ONCE kinds, the id otherwise).
    event_seq: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------------------
# on-disk layout (control host)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LabLayout:
    """Pure path arithmetic for the lab root on the control host. Identifiers are validated, so no path escapes the root.
    Every module builds paths with this class and never by hand."""

    root: Path

    def __post_init__(self) -> None:
        _set(self, "root", Path(self.root))

    # master ledger directory is the root itself: Ledger.create(root) writes <root>/ledger.jsonl
    @property
    def master_ledger_dir(self) -> Path:
        return self.root

    @property
    def lab_config_path(self) -> Path:
        return self.root / "lab.json"

    @property
    def anchors_path(self) -> Path:
        """Master head anchors, one line per anchoring event; committed and pushed to git so a rewritten ledger is detectable."""
        return self.root / "anchors" / "master.anchors"

    @property
    def frozen_dir(self) -> Path:
        return self.root / "frozen"

    @property
    def rule_path(self) -> Path:
        return self.frozen_dir / "rule.json"

    @property
    def budget_path(self) -> Path:
        return self.frozen_dir / "budget.json"

    @property
    def manifest_path(self) -> Path:
        return self.frozen_dir / "assignments.json"

    @property
    def final_analysis_path(self) -> Path:
        return self.frozen_dir / "final_analysis.json"

    @property
    def packs_dir(self) -> Path:
        return self.root / "packs"

    @property
    def pack_repo(self) -> Path:
        return self.packs_dir / "pack.git"

    @property
    def outer_dir(self) -> Path:
        """Ledger of the reused BudgetGate / MeteredClient for Wonder calls (``budget.charged`` events)."""
        return self.root / "outer"

    @property
    def gpu_dir(self) -> Path:
        return self.root / "gpu"

    @property
    def rental_ledger(self) -> Path:
        return self.gpu_dir / "rentals.json"

    def gpu_session_dir(self, session: str) -> Path:
        return self.gpu_dir / "sessions" / _ident(session, "session")

    @property
    def third_party_dir(self) -> Path:
        """Pinned checkouts of IFBench and open-instruct (fetched at setup, never vendored)."""
        return self.root / "third_party"

    @property
    def public_data_dir(self) -> Path:
        """Train and dev data: what researchers may see. Copied to researcher hosts per trial."""
        return self.root / "data"

    @property
    def private_dir(self) -> Path:
        """Selection and test data, the split salt. Mode 0700, owned by the lab user on the control host; never copied to a researcher host."""
        return self.root / "private"

    @property
    def split_salt_path(self) -> Path:
        return self.private_dir / "split_salt"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def final_comparison_path(self) -> Path:
        """Written by ``reveal`` once the whole final plan has its scores."""
        return self.reports_dir / "final_comparison.json"

    @property
    def split_report_path(self) -> Path:
        """Counts and hashes of the files the split builder wrote (control host only)."""
        return self.private_dir / "splits_report.json"

    @property
    def template_ngrams_path(self) -> Path:
        """Boilerplate 13-grams the split builder and the contamination scan ignore (control host only)."""
        return self.private_dir / "template_ngrams.json"

    # trials ---------------------------------------------------------------------------------
    def trial_dir(self, trial_id: str) -> Path:
        return self.root / "trials" / _ident(trial_id, "trial_id")

    def trial_ledger_dir(self, trial_id: str) -> Path:
        return self.trial_dir(trial_id)

    def trial_spec_path(self, trial_id: str) -> Path:
        return self.trial_dir(trial_id) / "spec.json"

    def trial_record_path(self, trial_id: str) -> Path:
        return self.trial_dir(trial_id) / "trial.json"

    def trial_pack_dir(self, trial_id: str) -> Path:
        return self.trial_dir(trial_id) / "pack"

    def trial_run_dir(self, trial_id: str) -> Path:
        return self.trial_dir(trial_id) / "run"

    def trial_trace_dir(self, trial_id: str) -> Path:
        return self.trial_dir(trial_id) / "trace"

    def trial_recipe_dir(self, trial_id: str) -> Path:
        return self.trial_dir(trial_id) / "recipe"

    def trial_train_data_path(self, trial_id: str, job_id: str) -> Path:
        return self.trial_dir(trial_id) / "train_data" / (_ident(job_id, "job_id") + ".jsonl")

    def trial_job_log_path(self, trial_id: str, job_id: str) -> Path:
        return self.trial_dir(trial_id) / "job_logs" / (_ident(job_id, "job_id") + ".log")

    # scores ---------------------------------------------------------------------------------
    def score_dir(self, score_id: str) -> Path:
        return self.root / "scores" / _ident(score_id, "score_id")

    def score_request_path(self, score_id: str) -> Path:
        return self.score_dir(score_id) / "request.json"

    def score_record_path(self, score_id: str) -> Path:
        return self.score_dir(score_id) / "score.json"

    def per_prompt_path(self, score_id: str) -> Path:
        return self.score_dir(score_id) / "per_prompt.jsonl"

    def gen_responses_path(self, score_id: str, seed: int) -> Path:
        return self.score_dir(score_id) / f"gen_seed{_int(seed, 'seed', minimum=0)}.jsonl"

    # candidates -----------------------------------------------------------------------------
    def candidate_dir(self, candidate_id: str) -> Path:
        return self.root / "candidates" / _ident(candidate_id, "candidate_id")

    def candidate_proposal_path(self, candidate_id: str) -> Path:
        return self.candidate_dir(candidate_id) / "proposal.json"

    def candidate_prompt_path(self, candidate_id: str) -> Path:
        return self.candidate_dir(candidate_id) / "prompt.txt"

    def candidate_response_path(self, candidate_id: str) -> Path:
        return self.candidate_dir(candidate_id) / "response.txt"

    def candidate_diff_path(self, candidate_id: str) -> Path:
        return self.candidate_dir(candidate_id) / "diff.patch"

    def candidate_lint_path(self, candidate_id: str) -> Path:
        return self.candidate_dir(candidate_id) / "lint.json"

    # adapters and data ----------------------------------------------------------------------
    def adapter_dir(self, adapter_id: str) -> Path:
        return self.artifacts_dir / "adapters" / _ident(adapter_id, "adapter_id")

    def adapter_json_path(self, adapter_id: str) -> Path:
        return self.adapter_dir(adapter_id) / "adapter.json"

    def dev_data(self, assignment_id: str) -> Path:
        return self.public_data_dir / "assignments" / _ident(assignment_id, "assignment_id") / "dev.jsonl"

    def train_data(self, assignment_id: str) -> Path:
        return self.public_data_dir / "assignments" / _ident(assignment_id, "assignment_id") / "train.jsonl"

    def selection_data(self, assignment_id: str) -> Path:
        return self.private_dir / "assignments" / _ident(assignment_id, "assignment_id") / "selection.jsonl"

    def test_data(self) -> Path:
        return self.private_dir / "test" / "IFBench_test.jsonl"

    def file_ref(self, path: str | os.PathLike[str]) -> FileRef:
        return file_ref(self.root, path)


# --------------------------------------------------------------------------------------------------
# protocols the implementers must satisfy
# --------------------------------------------------------------------------------------------------
Clock = Callable[[], float]  # epoch seconds; tests inject a FakeClock. Convert with utc_iso.
PortAllocator = Callable[[], int]  # returns a free loopback port, never reused within a minute
ScoreGuard = Callable[["ScoreRequest"], None]  # raises AccessDenied / FinalComparisonRefused; see docs/harness-design.md "Scorer"


@runtime_checkable
class LedgerWriter(Protocol):
    """The only object that touches a lab ledger. Thread-safe: every method may be called from any thread of the coordinator process."""

    def append(self, kind: str, actor: str, record: Any) -> Any:
        """Append one contract record under ``kind`` (must be in ``LabKind.all()``, payload of ``PAYLOAD_TYPES[kind]``). Returns the Event."""

    def anchor(self) -> LedgerAnchor: ...

    def seal(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class BudgetMeter(Protocol):
    """Per-trial hard stop. One instance per trial, writing to that trial's LedgerWriter. Thread-safe and idempotent per ``call_id``.

    LLM calls are reserve-then-settle: ``reserve_llm`` books the worst case (estimated prompt tokens plus ``max_tokens``, priced) before
    the request is forwarded and raises BudgetExhausted if the cap would be crossed; ``charge_llm`` replaces the reservation with the
    actual usage as TWO events with call ids ``<call_id>:tokens`` and ``<call_id>:usd`` (both are booked before BudgetExhausted is
    raised). A reservation that is never settled (stream without usage, client disconnect) stays booked at its worst case.
    GPU ticks use call id ``gpu:<rental_id>:<n>``. Wall time is derived from the injected clock since trial start (``check`` only).
    """

    @property
    def spec(self) -> BudgetSpec: ...

    def check(self, resource: Resource) -> None:
        """Raise BudgetExhausted BEFORE a spend if the axis ``AXIS_OF[resource]`` is already used up."""

    def charge(self, resource: Resource, amount: float, call_id: str, detail: Optional[Mapping[str, Any]] = None) -> BudgetUsage:
        """Book ``amount`` (idempotent per ``call_id``), append ``lab.budget_charged``, return usage. Raises BudgetExhausted after crossing a cap."""

    def reserve_llm(self, model: str, prompt_tokens_estimate: int, max_tokens: int, call_id: str) -> None: ...

    def charge_llm(self, model: str, usage: LlmUsage, call_id: str, detail: Optional[Mapping[str, Any]] = None) -> BudgetUsage:
        """``detail`` (scalars, for example ``{DETAIL_USAGE_ESTIMATED: True}``) is added to both events."""

    def usage(self) -> BudgetUsage: ...


@runtime_checkable
class GpuProvider(Protocol):
    """One GPU rental provider. The money rule, ledger, ownership and watchdog live in the rental manager, not here."""

    name: str

    def quote(self, shape: Shape) -> Quote:
        """Pure read. Raises ProviderUnavailable on network error; never returns a stale rate silently."""

    def list_instances(self) -> list[GpuInstance]:
        """The whole account. Raises ProviderUnavailable on ANY error or malformed output; never returns [] on error."""

    def create(self, shape: Shape, ssh_public_key: str) -> CreateReceipt:
        """Never retried automatically. The ownership tag is ``owner_tag_of_key(ssh_public_key)``. Raises CreateRejected (nothing exists)
        or CreateAmbiguous (something may exist)."""

    def find_created(self, before: set[str], shape: Shape, attempt_epoch: float, owner_tag: str) -> Optional[str]:
        """The single new instance carrying ``owner_tag`` created at or after attempt_epoch - 120 s; None for zero or several."""

    def delete(self, handle: str, expect_tag: Optional[str]) -> None:
        """Idempotent. Raises OwnershipError if the row's tag differs, ProviderUnavailable if the outcome is unknown."""

    def exists(self, handle: str) -> Optional[bool]:
        """True, False, or None when it could not be determined. Only False may be recorded as deleted."""


@runtime_checkable
class ServeSession(Protocol):
    @property
    def endpoint(self) -> ServeEndpoint:
        """``endpoint.base_url`` is a control-host loopback port forwarded over ssh to the rental; the control host never talks to a public port."""

    def close(self) -> GpuUsage:
        """Stop serving and delete the rental (confirmed absent). Idempotent. Returns the billed usage."""

    def __enter__(self) -> "ServeSession": ...

    def __exit__(self, *exc: Any) -> None: ...


@runtime_checkable
class ModelServer(Protocol):
    """Serves Glimmer (+ adapters) behind an OpenAI-compatible endpoint on a GPU the server rents and always deletes."""

    def serve(self, spec: ServeSpec, meter: Optional[BudgetMeter] = None) -> ServeSession:
        """Raises BudgetExhausted if ``meter`` has no GPU room, GpuError / TrainerError on provider or host failure."""


@runtime_checkable
class Generator(Protocol):
    """Produces final answers for a list of prompts from one served model. Same code path for all three arms."""

    def generate(self, model: str, prompts: Sequence[GenPrompt], params: GenerationParams, seed: int) -> list[GenResult]:
        """Same length and order as ``prompts``. A per-prompt failure sets ``GenResult.error``; a systemic failure raises ScorerError."""


GeneratorFactory = Callable[[ServeEndpoint], Generator]  # the one spelling: score_session, GlimmerTrainer and the CLI all use it


@runtime_checkable
class Scorer(Protocol):
    """Scores a ScoreRequest using ``generator`` (already bound to a live endpoint) and the verifiers on the control host."""

    @property
    def scorer_version(self) -> str:
        """``scorer_version_of(verifier pins, metric names, module version)``; the CLI reads it for freeze-rule and trial."""

    def score(self, request: ScoreRequest, generator: Generator, model: str) -> ScoreRecord:
        """Writes ``scores/<id>/request.json`` and ``score.json`` (``write_record``) and returns the record; the caller appends the event.
        Raises AccessDenied (split/phase/assignment violation or a failed ScoreGuard) or ScorerError. Never builds a record from missing responses."""


@runtime_checkable
class Trainer(Protocol):
    """Runs one researcher training job on a rented GPU and scores the result on dev. Blocking; safe to call from a worker thread."""

    def run_job(self, job: TrainJob, meter: BudgetMeter) -> TrainResult:
        """Failures of the job itself come back as a TrainResult with status != ok; TrainerError is for infrastructure faults. The dev
        ScoreRecord is already on disk under ``scores/<result.dev_score_id>/``; the caller reads it and appends ``lab.score_recorded``."""


@runtime_checkable
class RuntimeHandle(Protocol):
    def poll(self) -> Optional[ExitReason]:
        """None while running."""

    def stop(self, reason: ExitReason) -> None:
        """Stop the whole run (all agents, gateway, grader). Idempotent."""

    def collect(self) -> RuntimeResult:
        """After exit: harvest attempts, submissions, trace and notes (symlinks, hard links and devices refused). Raises RuntimeFailure
        if nothing can be harvested."""

    def close(self) -> None:
        """Kill leftovers, close tunnels and delete the researcher host's trial directory. Idempotent; never raises. Called after collect."""


@runtime_checkable
class ResearcherRuntime(Protocol):
    def start(self, context: TrialContext) -> RuntimeHandle:
        """Launch the pack's researcher team from a clean workspace. Raises RuntimeFailure."""


@runtime_checkable
class ProposalSource(Protocol):
    def propose(self, context: ProposalContext) -> ProposalResult:
        """One revision. Raises ProposalInvalid if the reply cannot be parsed into a Proposal; BudgetExhausted if the outer budget is spent."""


@runtime_checkable
class TrialExecutor(Protocol):
    """The trial runner as the outer loop and the CLI see it. ``run`` executes in a thread of the coordinator process; trials may run
    concurrently."""

    def run(self, spec: TrialSpec) -> TrialRecord: ...


@runtime_checkable
class PackStore(Protocol):
    """Pack git operations (owner 4), as the trial runner, the outer loop and the CLI use them."""

    def init_v0(self, template_dir: str | os.PathLike[str]) -> PackRef: ...

    def checkout(self, pack: PackRef, dest: str | os.PathLike[str]) -> None:
        """Clean checkout of ``pack`` into the empty directory ``dest``."""

    def tree_sha256(self, pack: PackRef) -> str: ...

    def apply(self, proposal: Proposal, parent: PackRef, version_label: str) -> tuple[PackRef, str]:
        """Strict apply of ``proposal`` on ``parent``: the new commit (tagged ``pack/<version_label>``) and the diff text. Raises
        ProposalInvalid on an unmatched block, a forbidden path, a symlink or an over-size file. Spends nothing."""

    def lint(self, pack: PackRef) -> LintResult: ...

    def accept(self, pack: PackRef, version_label: str) -> None:
        """Fast-forward linear ``main`` to ``pack``. Candidates stay tagged whether accepted or not; nothing is deleted or rewritten."""


@runtime_checkable
class DecisionRule(Protocol):
    """Statistics (owner 3), as the outer loop and the CLI use them. Pure and deterministic; never raises for data problems."""

    def trial_spread(self, scores: Sequence[TrialScore]) -> SpreadEstimate:
        """Pooled within-cell sd. Raises ContractError 'too few trials to estimate spread' when no assignment has two trials."""

    def screen(self, rule: Rule, incumbent: Sequence[TrialScore], candidates: Mapping[str, Sequence[TrialScore]]) -> dict[str, Optional[float]]:
        """Delta of each candidate against the incumbent; None when not comparable (comparability keys or trial counts differ)."""

    def decide(self, rule: Rule, incumbent: Sequence[TrialScore], candidate: Sequence[TrialScore]) -> DecisionNumbers:
        """The confirmation. INVALID on any comparability mismatch or when the candidate's sd exceeds the variance guard."""


@runtime_checkable
class StateReader(Protocol):
    """The fold queries (owner 6), as the outer loop, the trial runner, the guards and the report use them."""

    def state(self) -> LabState: ...

    def incumbent_version(self, state: LabState) -> str:
        """The version label of the latest accepted candidate, else ``v0``."""

    def selection_trial_scores(self, state: LabState, rule: Rule, pack_version: str) -> list[TrialScore]:
        """Search-phase trials of ``pack_version`` on ``rule.assignments``: ``rule.metric`` of the first selection score (extra ones are
        malformed), scorer version, decoding, seeds and verifier equal to the rule's. BUDGET_EXHAUSTED trials with a scored adapter count;
        other non-scored trials follow ``rule.failed_trial_policy``."""

    def cost_breakdown(self, state: LabState, trial_ids: Sequence[str], candidate_id: Optional[str]) -> CostBreakdown:
        """Sum of the trials' budget_used (ticks), the candidate's propose cost and the scoring GPU. Money reporting also reads the rental ledger."""

    def next_id(self, state: LabState, prefix: str, *, taken: Iterable[str] = (), reserve: bool = True) -> str:
        """The next unused id of the form ``<prefix>-<NNNN>`` (prefixes: t, cand, ad, a, j, sc, f); allocated by the coordinator only. Ids in
        ``taken`` and ids already issued by this reader (when ``reserve``) are skipped; ``reserve=False`` only peeks."""

    def has_gate(self, state: LabState, gate: Gate) -> bool: ...

    def gate_status(self, state: LabState, gate: Gate) -> str: ...

    def final_requests(self, state: LabState, pack: Optional[PackRef]) -> list[FinalRequest]: ...

    def load_frozen_all(self, state: LabState) -> tuple[Rule, AssignmentManifest, BudgetSpec]:
        """Parse the three frozen files and re-hash them against the ledger (``load_frozen``); ContractError on a mismatch."""

    def load_final_analysis(self, state: LabState) -> FinalAnalysis:
        """Parse ``frozen/final_analysis.json`` and re-hash it against the ledger; ContractError if it is not frozen or was rewritten."""


MeterFactory = Callable[[TrialSpec, LedgerWriter], BudgetMeter]  # (spec, that trial's ledger writer) -> meter
TrialLedgerFactory = Callable[[str], LedgerWriter]  # trial_id -> a new thread-safe writer for trials/<id>/ledger.jsonl; closed by TrialRunner.run


@dataclass
class LabWorld:
    """The dependency container the CLI builds. ``cli.main(argv, *, world=None)`` uses ``world`` when given; tests build a world of fakes
    in ``tests/support/lab_world.py``. There is no ``--fake`` flag and no fake in production code. Not persisted."""

    layout: LabLayout
    config: Optional[LabConfig]
    clock: Clock
    ledger: LedgerWriter
    state_reader: StateReader
    new_trial_ledger: TrialLedgerFactory
    meter_factory: MeterFactory
    scorer: Scorer
    generator_factory: GeneratorFactory
    trainer: Trainer
    model_server: ModelServer
    runtime: ResearcherRuntime
    packs: PackStore
    decision_rule: DecisionRule
    trial_executor: TrialExecutor
    proposal_source: Optional[ProposalSource]
