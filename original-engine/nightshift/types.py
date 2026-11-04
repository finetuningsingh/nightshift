"""Shared contract for Nightshift.

Every cross-module type lives here. Modules import cross-module types from this file only, never from each
other's private internals. Stdlib only. Python 3.11+.

Conventions
-----------
* All ledger payloads are plain JSON-serialisable dicts. Dataclasses here have ``to_dict`` / ``from_dict``.
* Time is always Unix seconds (float) and always comes from an injectable ``Clock`` so tests are deterministic.
* Nothing in this file performs I/O.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence, runtime_checkable

SCHEMA_VERSION = 1
Clock = Callable[[], float]
JSON = Any  # anything json.dumps accepts


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def canonical_json(obj: Any) -> str:
    """Deterministic JSON used for hashing and for the ledger file."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def short_hash(data: str | bytes, n: int = 12) -> str:
    return sha256_hex(data)[:n]


def format_exp_id(n: int) -> str:
    return f"e{n:03d}"


def _filter_fields(cls: type, d: Mapping[str, Any]) -> dict[str, Any]:
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in names}


# --------------------------------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------------------------------
class NightshiftError(Exception):
    """Base class."""


class BudgetExceeded(NightshiftError):
    def __init__(self, reason: str, kind: str = "unknown", limit: float | None = None, used: float | None = None):
        super().__init__(reason)
        self.reason, self.kind, self.limit, self.used = reason, kind, limit, used


class HaltRequested(NightshiftError):
    """A HALT file exists in the run directory, or the operator interrupted."""


class DuplicateExperiment(NightshiftError):
    def __init__(self, idem_key: str, dup_of: str):
        super().__init__(f"duplicate of {dup_of}")
        self.idem_key, self.dup_of = idem_key, dup_of


class ProtectedPathViolation(NightshiftError):
    def __init__(self, paths: Sequence[str], kind: str = "workspace"):
        super().__init__(f"protected {kind} paths changed: {', '.join(paths)}")
        self.paths, self.kind = list(paths), kind


class InvalidComposition(NightshiftError):
    def __init__(self, errors: Sequence[str]):
        super().__init__("; ".join(errors))
        self.errors = list(errors)


class SimulatedCrash(NightshiftError):
    """Raised by the test/demo crash hook. The engine stops all lanes and the orchestrator closes the ledger."""


class LedgerCorrupt(NightshiftError):
    """A line other than a torn final line failed to parse. Never auto-repaired."""

    def __init__(self, message: str, first_bad_line: int):
        super().__init__(message)
        self.first_bad_line = first_bad_line


class ModelError(NightshiftError):
    """Raised by ModelClient implementations. ``retryable`` follows the Meta docs (429/500/503)."""

    def __init__(self, message: str, *, status: int | None = None, error_type: str | None = None,
                 code: str | None = None, retryable: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.status, self.error_type, self.code = status, error_type, code
        self.retryable, self.retry_after = retryable, retry_after


# --------------------------------------------------------------------------------------------------
# ledger events
# --------------------------------------------------------------------------------------------------
class Kind:
    """Event kind constants. The authoritative payload schemas are in docs/ARCHITECTURE.md section 4."""

    RUN_STARTED = "run.started"
    RUN_RESUMED = "run.resumed"
    RUN_FINISHED = "run.finished"
    SUP_TURN = "supervisor.turn"
    SUP_NUDGE = "supervisor.nudge"
    SUP_FINISH_REQUESTED = "supervisor.finish_requested"
    SUP_FINISH_REFUSED = "supervisor.finish_refused"
    CLAIM = "agent.claim"
    EXP_PROPOSED = "experiment.proposed"
    EXP_BLOCKED = "experiment.blocked"
    EXP_WORKSPACE = "experiment.workspace_ready"
    EXP_IMPLEMENTED = "experiment.implemented"
    EXP_PROTECTED = "experiment.protected_violation"
    EXP_RUN_STARTED = "experiment.run_started"
    EXP_HEARTBEAT = "experiment.heartbeat"
    EXP_RUN_FINISHED = "experiment.run_finished"
    EXP_GRADED = "experiment.graded"
    EXP_REVIEWED = "experiment.reviewed"
    EXP_REPAIRED = "experiment.repaired"
    EXP_RECORDED = "experiment.recorded"
    INCUMBENT_PROMOTED = "incumbent.promoted"
    LESSON = "memory.lesson"
    REFLECTION = "memory.reflection"
    BUDGET_CHARGED = "budget.charged"
    BUDGET_EXHAUSTED = "budget.exhausted"
    AUDIT_WINDOW = "audit.window"
    AUDIT_FINDING = "audit.finding"
    AUDIT_ACK = "audit.ack"
    WATCHDOG_STALLED = "watchdog.stalled"

    @classmethod
    def all(cls) -> set[str]:
        return {v for k, v in vars(cls).items() if k.isupper() and isinstance(v, str)}


@dataclass(frozen=True)
class Event:
    """One line of the append-only ledger. ``hash`` chains each event to its predecessor."""

    seq: int
    ts: float
    run_id: str
    kind: str
    actor: str
    data: dict
    prev_hash: str
    hash: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Event":
        return cls(**_filter_fields(cls, d))


GENESIS_HASH = "0" * 64


def compute_event_hash(prev_hash: str, seq: int, ts: float, run_id: str, kind: str, actor: str, data: dict) -> str:
    return sha256_hex(canonical_json([prev_hash, seq, ts, run_id, kind, actor, data]))


@runtime_checkable
class LedgerProtocol(Protocol):
    """What every other module may rely on. Implemented by ``nightshift.ledger.Ledger``."""

    run_id: str

    def append(self, kind: str, actor: str, data: dict) -> Event: ...
    def events(self, since_seq: int = 0) -> list[Event]: ...   # events with seq > since_seq
    def view(self) -> "LedgerView": ...
    def next_exp_id(self) -> str: ...                            # thread-safe, e000 is the baseline
    def verify_chain(self) -> tuple[bool, Optional[int]]: ...   # (ok, first_bad_seq)


class Purpose:
    """Why a run or a grade happened. Which split a purpose uses is fixed here and nowhere else."""

    BASELINE = "baseline"                        # dev split, task.seeds, only the unmodified template
    PRIMARY = "primary"                          # dev split, task.seeds
    REPLICATION = "replication"                  # SEALED split, task.replication_seeds, a candidate
    REPLICATION_BASELINE = "replication_baseline"  # SEALED split, task.replication_seeds, the baseline exp

    AGENT_VISIBLE = (BASELINE, PRIMARY)
    SEALED = (REPLICATION, REPLICATION_BASELINE)

    @classmethod
    def split(cls, purpose: str) -> str:
        return "sealed" if purpose in cls.SEALED else "dev"


# --------------------------------------------------------------------------------------------------
# task specification (task.toml, parsed by nightshift.task)
# --------------------------------------------------------------------------------------------------
@dataclass
class TaskSpec:
    name: str
    description: str
    brief: str                    # text safe to show to models: the research objective and rules
    task_dir: Path                # directory containing task.toml (trusted)
    template_dir: Path            # workspace template, copied for every experiment (untrusted afterwards)
    trusted_dir: Path             # grader, labels, held-out data. NEVER copied into a workspace.
    objective: str                # "minimize" | "maximize"
    primary_metric: str           # held-out metric the grader returns as ``score``
    train_metric: Optional[str]   # in-sample metric, used for the overfit invariant
    editable: list[str]           # globs (workspace-relative) an agent may create or modify
    scratch: list[str]            # globs the run may write (outputs/**), ignored by protection checks
    run_cmd: list[str]            # executed in the workspace; untrusted
    run_timeout_sec: float
    grade_cmd: list[str]          # executed with cwd=task_dir; trusted. Tokens: {workspace} {seed} {task_dir}
    grade_timeout_sec: float
    seeds: list[int]              # evaluation seeds for primary grading
    replication_seeds: list[int]  # disjoint seeds for replication
    noise_margin: float           # absolute minimum delta that can count as an improvement
    requires: list[str]           # python modules the run needs; a missing one is an infra failure
    scripted_dir: Optional[Path] = None   # tasks/<name>/scripted (offline proposal bank), if present
    sealed: bool = True           # the grader supports --split sealed, used only for replication
    sigma_floor: float = 0.0      # minimum per-seed std assumed when computing the margin
    max_output_bytes: int = 8 * 1024 * 1024   # per staged output file
    stall_sec: float = 10.0       # no output growth for this long -> watchdog.stalled, then kill
    extra: dict = field(default_factory=dict)

    @property
    def minimize(self) -> bool:
        return self.objective == "minimize"


@dataclass
class GradeResult:
    """Output of the trusted grader, aggregated over seeds."""

    valid: bool
    score: Optional[float]                    # mean primary metric over seeds
    std: float = 0.0
    per_seed: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)        # means, includes train metric if any
    per_seed_metrics: list[dict[str, float]] = field(default_factory=list)
    grader_hash: str = ""
    notes: str = ""
    error: Optional[str] = None
    split: str = "dev"                        # dev | sealed
    n_eval: int = 0                           # evaluated rows per seed (0 if unknown)
    seeds: list[int] = field(default_factory=list)   # aligned with per_seed. ``std`` is the sample std (ddof=1), 0.0 if n<2

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GradeResult":
        return cls(**_filter_fields(cls, d))


@dataclass
class OutcomeDecision:
    """Result of comparing a candidate grade to an incumbent grade under the same conditions."""

    outcome: str                  # improved | no_gain | worse | invalid
    delta: Optional[float]        # mean paired improvement; positive means better regardless of objective direction
    margin: float                 # delta must exceed this to count
    train_delta: Optional[float]  # same sign convention, None if no train metric
    overfit_suspect: bool         # train improved beyond train_margin while held-out did not beat margin
    reason: str = ""
    train_margin: Optional[float] = None
    n: int = 0                    # number of paired seeds
    s_eff: float = 0.0            # std actually used: max(sample std of paired deltas, sigma_prior, task.sigma_floor)
    t: float = 0.0                # critical value used


# --------------------------------------------------------------------------------------------------
# materialised ledger view
# --------------------------------------------------------------------------------------------------
class ExpStatus:
    PROPOSED = "proposed"
    BLOCKED = "blocked"
    WORKSPACE = "workspace"
    IMPLEMENTED = "implemented"
    VIOLATION = "violation"
    RUNNING = "running"
    RAN = "ran"
    GRADED = "graded"
    REVIEWED = "reviewed"
    RECORDED = "recorded"
    TERMINAL = {BLOCKED, VIOLATION, RECORDED}


class Outcome:
    BASELINE = "baseline"
    IMPROVED = "improved"
    NO_GAIN = "no_gain"
    WORSE = "worse"
    FAILED = "failed"          # the experiment's own code failed (a model-side failure)
    INVALID = "invalid"        # grader said the output is not valid
    INFRA = "infra_error"      # the environment failed, says nothing about the hypothesis
    UNRESOLVED = "unresolved"  # review/repair loop gave up


@dataclass
class RunRecord:
    run_idx: int
    purpose: str                       # primary | replication | baseline
    seed: Optional[int]
    started_seq: int
    started_ts: float
    pid: Optional[int] = None
    finished_seq: Optional[int] = None
    finished_ts: Optional[float] = None
    exit_code: Optional[int] = None
    duration_sec: Optional[float] = None
    classification: Optional[str] = None     # ok | timeout | crash | infra | interrupted
    last_heartbeat_ts: Optional[float] = None
    last_output_bytes: Optional[int] = None
    stalled: bool = False
    isolation: str = ""                      # userns | unshare-net | none, as actually applied


@dataclass
class GradeRecord:
    purpose: str                       # primary | replication | baseline | replication_baseline
    seq: int
    ts: float
    grade: GradeResult
    seeds: list[int]
    conditions_hash: str               # hash of (task, grader_hash, seeds, run_cmd): equal means comparable
    grader_hash: str


@dataclass
class ReviewRecord:
    round: int
    seq: int
    reviewer: str
    verdict: str                       # accept | reject | repair
    findings: list[dict]
    independent: bool


@dataclass
class ExperimentView:
    exp_id: str
    lane: str = ""
    hypothesis: str = ""
    plan: str = ""
    parent_exp_id: Optional[str] = None
    base: str = "template"
    replicate_of: Optional[str] = None
    hyp_fingerprint: str = ""
    proposed_seq: int = 0
    proposed_ts: float = 0.0
    status: str = ExpStatus.PROPOSED
    worker: str = ""
    diff_hash: str = ""
    idem_key: str = ""
    files_changed: list[str] = field(default_factory=list)
    diff_summary: str = ""
    attempts: int = 0
    workspace_path: str = ""
    runs: list[RunRecord] = field(default_factory=list)
    grades: list[GradeRecord] = field(default_factory=list)
    reviews: list[ReviewRecord] = field(default_factory=list)
    repairs: list[dict] = field(default_factory=list)
    protected_violations: list[dict] = field(default_factory=list)
    blocked: Optional[dict] = None
    outcome: Optional[str] = None
    delta: Optional[float] = None
    vs_exp_id: Optional[str] = None
    noise_margin: Optional[float] = None
    train_delta: Optional[float] = None
    replicated: bool = False
    claim_ids: list[str] = field(default_factory=list)
    recorded_seq: Optional[int] = None
    recorded_ts: Optional[float] = None
    scripted_id: Optional[str] = None
    planted_fault: Optional[str] = None      # ONLY for display and the fault-injection scoreboard; auditors must not read it
    tree_hash: str = ""
    dev_outcome: Optional[str] = None        # outcome of the lane (dev split) before any replication
    sealed_delta: Optional[float] = None     # human-only: agent-facing code must never read these three
    sealed_margin: Optional[float] = None
    sealed_outcome: Optional[str] = None
    overfit_suspect: bool = False
    train_margin: Optional[float] = None

    @property
    def awaiting_replication(self) -> bool:
        return self.dev_outcome == "improved" and self.grade_for("replication") is None and self.status == ExpStatus.RECORDED

    def grade_for(self, purpose: str) -> Optional[GradeRecord]:
        found = [g for g in self.grades if g.purpose == purpose]
        return found[-1] if found else None

    @property
    def primary_grade(self) -> Optional[GradeResult]:
        g = self.grade_for("primary") or self.grade_for("baseline")
        return g.grade if g else None

    @property
    def replication_grade(self) -> Optional[GradeResult]:
        g = self.grade_for("replication")
        return g.grade if g else None

    @property
    def last_review(self) -> Optional[ReviewRecord]:
        return self.reviews[-1] if self.reviews else None

    @property
    def terminal(self) -> bool:
        return self.status in ExpStatus.TERMINAL


class Verdict:
    VIOLATED = "violated"
    COMPLIANT = "compliant"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT = "insufficient_evidence"
    ALL = (VIOLATED, COMPLIANT, NOT_APPLICABLE, INSUFFICIENT)


@dataclass
class Finding:
    finding_id: str                    # "<invariant_id>@<subject>"
    invariant_id: str
    verdict: str                       # Verdict.*
    severity: str                      # block | warn | info
    source: str                        # deterministic | llm
    rationale: str
    evidence_seqs: list[int] = field(default_factory=list)
    confidence: Optional[float] = None
    subject: str = ""
    window_from: int = 0
    window_to: int = 0
    status: str = "open"               # open | addressed | dismissed (derived from audit.ack events)
    created_seq: int = 0
    judge_model: Optional[str] = None
    exp_id: Optional[str] = None       # experiment the finding is about, when attributable

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Finding":
        return cls(**_filter_fields(cls, d))


@dataclass
class LedgerView:
    """Everything derived from the ledger. Built by ``nightshift.view.build_view``. Read-only by convention."""

    run: dict = field(default_factory=dict)                 # run.started payload plus run_id
    finished: Optional[dict] = None                         # run.finished payload
    experiments: dict[str, ExperimentView] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)          # exp ids in proposal order
    baseline_id: Optional[str] = None
    incumbent_id: Optional[str] = None
    promotions: list[dict] = field(default_factory=list)
    claims: dict[str, dict] = field(default_factory=dict)   # claim_id -> payload + {seq, ts, actor}
    lessons: list[dict] = field(default_factory=list)
    reflections: list[dict] = field(default_factory=list)
    findings: dict[str, Finding] = field(default_factory=dict)
    acks: list[dict] = field(default_factory=list)
    budget_used: dict[str, float] = field(default_factory=dict)   # usd, experiments, llm_calls
    charged_call_ids: set[str] = field(default_factory=set)
    supervisor_turns: int = 0
    turns_since_experiment: int = 0
    finish_requests: list[dict] = field(default_factory=list)
    finish_refusals: list[dict] = field(default_factory=list)
    stalls: list[dict] = field(default_factory=list)
    audit_windows: list[dict] = field(default_factory=list)
    blocks: list[dict] = field(default_factory=list)        # every experiment.blocked payload (+ seq, ts)
    turns: list[dict] = field(default_factory=list)         # every supervisor.turn payload (+ seq, ts)
    nudges: list[dict] = field(default_factory=list)
    malformed: list[dict] = field(default_factory=list)     # events whose payload could not be applied (seq, kind, error)
    last_seq: int = 0
    first_ts: float = 0.0
    last_ts: float = 0.0
    event_count: int = 0

    # convenience ---------------------------------------------------------------------------------
    def exp(self, exp_id: str) -> Optional[ExperimentView]:
        return self.experiments.get(exp_id)

    @property
    def incumbent(self) -> Optional[ExperimentView]:
        return self.experiments.get(self.incumbent_id) if self.incumbent_id else None

    @property
    def baseline(self) -> Optional[ExperimentView]:
        return self.experiments.get(self.baseline_id) if self.baseline_id else None

    def open_findings(self, severity: str | None = None) -> list[Finding]:
        return [f for f in self.findings.values()
                if f.status == "open" and f.verdict == Verdict.VIOLATED and (severity is None or f.severity == severity)]

    def in_flight(self) -> list[ExperimentView]:
        return [e for e in self.experiments.values() if not e.terminal]

    def recorded(self) -> list[ExperimentView]:
        return [e for e in self.experiments.values() if e.status == ExpStatus.RECORDED]


# --------------------------------------------------------------------------------------------------
# model layer
# --------------------------------------------------------------------------------------------------
@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict                    # JSON Schema object

    def to_openai(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict                     # parsed; {} if the model sent invalid JSON (see ``raw_arguments``)
    raw_arguments: str = ""
    parse_error: Optional[str] = None


@dataclass
class ChatMessage:
    role: str                           # system | user | assistant | tool
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResponse:
    message: ChatMessage
    finish_reason: str
    usage: Usage
    model: str
    request_id: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.message.content or ""


@runtime_checkable
class ModelClient(Protocol):
    """One chat model endpoint. Implementations: MetaChatClient (real), ScriptedClient and fakes (tests)."""

    model: str

    def chat(self, messages: Sequence[ChatMessage], *, tools: Sequence[ToolSpec] | None = None,
             response_schema: dict | None = None, schema_name: str = "response",
             reasoning_effort: str | None = None, max_completion_tokens: int | None = None,
             temperature: float | None = None, cache_key: str | None = None,
             timeout: float | None = None) -> ChatResponse: ...


# --------------------------------------------------------------------------------------------------
# agent roles
# --------------------------------------------------------------------------------------------------
@dataclass
class LaneSpec:
    """A request from the supervisor to explore one hypothesis."""

    lane: str                           # short label, unique within a launch (e.g. "A")
    hypothesis: str
    approach: str = ""                  # optional hint for the designer
    base: str = "incumbent"             # "incumbent" | "template" | <exp_id>
    replicate_of: Optional[str] = None  # exp_id to re-run on replication seeds
    scripted_id: Optional[str] = None   # offline proposal id (scripted backend only)


@dataclass
class LaneResult:
    exp_id: str                         # "" for a block before any experiment was proposed
    lane: str
    status: str                         # ExpStatus.*
    outcome: Optional[str] = None
    delta: Optional[float] = None
    score: Optional[float] = None
    replicated: Optional[bool] = False  # None when a replication was refused or could not run
    review_verdict: Optional[str] = None
    blocked_reason: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class ExperimentPlan:
    hypothesis: str
    approach: str
    expected_effect: str = ""
    risks: str = ""
    files_to_touch: list[str] = field(default_factory=list)


@dataclass
class DesignContext:
    task: TaskSpec
    lane: LaneSpec
    memory_digest: str
    incumbent_summary: str              # score, key facts about the current best
    editable_files: dict[str, str]      # path -> current content in the base workspace
    exp_id: str = ""                    # the experiment this lane became (only to attribute the designer's spend)


@dataclass
class ImplementContext:
    task: TaskSpec
    exp_id: str
    plan: ExperimentPlan
    lane: LaneSpec
    workspace: Path
    memory_digest: str
    repair_findings: list[dict] = field(default_factory=list)   # non-empty when repairing
    prior_diff: str = ""
    emit_claim: Optional[Callable[[str, str], str]] = None      # (claim_type, text) -> claim_id


@dataclass
class ImplementResult:
    summary: str
    steps: int = 0
    worker: str = ""
    raw_log: str = ""
    self_report: str = ""               # what the agent says happened; NEVER treated as evidence


@dataclass
class ReviewContext:
    task: TaskSpec
    exp_id: str
    plan: ExperimentPlan
    diff_text: str
    grade_summary: dict
    incumbent_summary: str
    run_summary: str
    implementer_identity: str           # reviewer must not share it (independence)
    repair_round: int = 0
    prior_findings: list[dict] = field(default_factory=list)


@dataclass
class ReviewVerdict:
    verdict: str                        # accept | reject | repair
    findings: list[dict] = field(default_factory=list)   # {id, severity, text, evidence}
    summary: str = ""
    reviewer: str = ""


@dataclass
class ReflectContext:
    task: TaskSpec
    trigger: str                        # stall | manual | periodic
    experiments_table: str
    lessons: list[dict]
    incumbent_summary: str


@dataclass
class ReflectionResult:
    lessons: list[dict]                 # {text, kind, source_exps, classification}
    pivots: list[str]
    summary: str = ""


@runtime_checkable
class Designer(Protocol):
    def design(self, ctx: DesignContext) -> ExperimentPlan: ...


@runtime_checkable
class Implementer(Protocol):
    identity: str

    def implement(self, ctx: ImplementContext) -> ImplementResult: ...


@runtime_checkable
class Reviewer(Protocol):
    identity: str

    def review(self, ctx: ReviewContext) -> ReviewVerdict: ...


@runtime_checkable
class Reflector(Protocol):
    def reflect(self, ctx: ReflectContext) -> ReflectionResult: ...


@runtime_checkable
class SupervisorEnv(Protocol):
    """The only surface a supervisor (LLM or scripted) uses to act. Implemented by the orchestrator.

    Nothing returned here may be derived from a sealed-split grade (see ``Purpose.SEALED``), except the boolean
    ``replicated``. The one exception is that the dashboard/report, which are human-only, may read the ledger directly.
    """

    def tool_specs(self) -> list[ToolSpec]: ...
    def call_tool(self, name: str, arguments: dict) -> dict: ...      # JSON-serialisable result, never raises
    def state(self) -> dict: ...                                      # compact state, shape in docs/ARCHITECTURE.md 6.8
    def launch(self, template: str, lanes: list[LaneSpec], stages: list[str] | None = None) -> list[LaneResult]: ...
    def replicate(self, exp_id: str) -> LaneResult: ...               # sealed re-grade; score and delta are always None
    def reflect(self) -> dict: ...
    def acknowledge(self, finding_id: str, action: str, reason: str) -> dict: ...
    def claim(self, claim_type: str, text: str, exp_id: str | None = None, cites: Sequence[int] = (),
              alias: str | None = None) -> str: ...                   # writes agent.claim (actor "supervisor"), returns claim_id
    def finish(self, summary: str, claim_ids: Sequence[str] = ()) -> dict: ...   # {"accepted": bool, "reasons": [...]}
    def note_turn(self, model: str, summary: str, *, say: str | None = None, tool_calls: Sequence[dict] | None = None,
                  reasoning_effort: str | None = None, latency_ms: float | None = None, text: str | None = None,
                  source: str = "supervisor") -> None: ...            # writes supervisor.turn. source: supervisor | script
    def nudge(self, text: str) -> None: ...                           # writes supervisor.nudge (templated text only)
    def should_stop(self) -> Optional[str]: ...                       # reason string, or None to continue


@runtime_checkable
class Supervisor(Protocol):
    identity: str

    def run(self, env: SupervisorEnv) -> None: ...


@dataclass
class AgentSuite:
    """Dependency-injection bundle chosen by backend (scripted | meta)."""

    backend: str
    supervisor: Supervisor
    designer: Designer
    implementer: Implementer
    reviewer: Reviewer
    reflector: Reflector
    judge: Optional["Judge"] = None
    reviewer_hard: Optional[Reviewer] = None     # used by the optional review_hard stage; None = stage is skipped with a note
    final_judge: Optional["Judge"] = None        # used by Auditor.final_sweep (auditor_final role); None = same as ``judge``
    control: bool = False                        # scripted suite running the clean negative-control manifest
    planted: Optional["AgentSuite"] = None       # scripted attackers (designer + implementer) for lanes that carry a scripted_id
                                                 # in a meta-backend run started with --plant; the real reviewer still reviews them
    models: dict[str, str] = field(default_factory=dict)   # role -> model id, recorded in run.started


# --------------------------------------------------------------------------------------------------
# auditing
# --------------------------------------------------------------------------------------------------
@dataclass
class Invariant:
    id: str                             # e.g. "I01"
    slug: str
    title: str
    statement: str                      # operational, falsifiable statement
    source: str                         # provenance in the design conversation
    severity: str                       # block | warn
    kind: str                           # deterministic | llm | both
    question: str = ""                  # narrow question for the LLM judge, if kind != deterministic


@dataclass
class JudgePacket:
    invariant: Invariant
    goal: str                           # current goal and authority, plain language
    state_fields: dict                  # named, bounded state the judge needs
    evidence: list[dict]                # [{"seq": int, "kind": str, "actor": str, "text": str}], bounded
    window_from: int
    window_to: int


@dataclass
class JudgeResult:
    verdict: str                        # Verdict.*
    confidence: Optional[float]
    evidence_seqs: list[int]
    rationale: str
    model: str = ""


@runtime_checkable
class Judge(Protocol):
    def judge(self, packet: JudgePacket) -> JudgeResult: ...    # raises ModelError on failure


# --------------------------------------------------------------------------------------------------
# budget
# --------------------------------------------------------------------------------------------------
@dataclass
class BudgetLimits:
    max_experiments: int = 16           # counts every proposed experiment except the baseline
    max_wall_sec: float = 1800.0
    max_llm_usd: float = 5.0
    max_parallel: int = 3
    max_supervisor_turns: int = 40
    max_replications: int = 6           # sealed-split queries; each one leaks one bit, so they are capped
    max_repairs: int = 2                # review -> repair loops per experiment
    stall_after: int = 3                # consecutive non-improving experiments before auto-reflection

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BudgetLimits":
        return cls(**_filter_fields(cls, d))
