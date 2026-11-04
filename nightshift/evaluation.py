"""Inspect AI evaluation (spec 3.8).

One task definition parameterised by split; the scorer applies the split
rows' pinned constraint checks and carries its own id+version (configured
separately from the training verifier set). Failure is not zero. Fake mode
runs Inspect for real against `mockllm/model` with deterministic outputs
derived from the fake checkpoint. Real mode takes an operator-supplied
Inspect model string for a served base+adapter on a fresh trusted eval host
and is NOT YET VALIDATED LIVE.

Offline note: Inspect estimates token counts with tiktoken (`o200k_base`),
which downloads its encoding on first use. The CPU Docker image pre-caches
it at build time (see docker/Dockerfile.cpu and docs/reproducibility.md),
and fake model outputs carry explicit token usage.
"""

from __future__ import annotations

import inspect as py_inspect
import json
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate

from . import CODE_REVISION
from .checkpoints import validate_checkpoint_dir
from .data import ControllerDataAccess, DatasetRegistry, TestAccessGate
from .errors import NightshiftError, PolicyError
from .generation import checks_for_row, synthesize_constrained_text, violating_text
from .hashing import file_sha256, hash_object, sha256_hex
from .ledger import EventLedger
from .verifiers import VERIFIER_SET_VERSION, check_text

TASK_NAME = "nightshift_instruction_following"
TASK_VERSION = "1.0.0"
SCORER_ID = "nightshift_constraint_scorer"
SCORER_VERSION = "1.0.0"

_ACTIVE_CHECKS: dict[str, list[dict[str, Any]]] = {}


class EvaluationError(NightshiftError):
    """An evaluation could not be run or its log could not be trusted."""


@dataclass(frozen=True)
class FailurePolicy:
    """Prespecified policy, recorded in the ledger at run start (spec 3.8)."""

    max_retries: int = 1
    unresolved_selection_failure_accepts: bool = False
    failed_trials_stay_in_denominator: bool = True
    failed_scores_imputed: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "FailurePolicy":
        return cls(max_retries=int(config.evaluation.max_retries))

    def to_record(self) -> dict[str, Any]:
        return {
            "max_retries": int(self.max_retries),
            "unresolved_selection_failure_accepts": self.unresolved_selection_failure_accepts,
            "failed_trials_stay_in_denominator": self.failed_trials_stay_in_denominator,
            "failed_scores_imputed": self.failed_scores_imputed,
        }


def record_failure_policy(
    ledger: EventLedger, policy: FailurePolicy, fake: bool = False
) -> dict[str, Any]:
    payload = policy.to_record()
    payload["fake"] = bool(fake)
    return ledger.append("failure_policy", "controller", payload)


@dataclass
class EvaluationRecord:
    evaluation_id: str
    status: str
    checkpoint_hash: str
    trial_id: str = ""
    pack_version: str = ""
    dataset_manifest_id: str = ""
    dataset_revision: str = ""
    dataset_sha256: str = ""
    split: str = ""
    task_name: str = TASK_NAME
    task_version: str = TASK_VERSION
    scorer_id: str = SCORER_ID
    scorer_version: str = SCORER_VERSION
    verifier_set: str = VERIFIER_SET_VERSION
    code_revision: str = CODE_REVISION
    decoding_params: dict[str, Any] = field(default_factory=dict)
    inspect_version: str = ""
    inspect_log_path: str = ""
    inspect_log_sha256: str | None = None
    model: str = ""
    score: float | None = None
    n: int = 0
    attempts: int = 0
    error: str = ""
    fake: bool = False

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "evaluation_id": self.evaluation_id,
            "status": self.status,
            "checkpoint_hash": self.checkpoint_hash,
            "trial_id": self.trial_id,
            "pack_version": self.pack_version,
            "dataset_manifest_id": self.dataset_manifest_id,
            "dataset_revision": self.dataset_revision,
            "dataset_sha256": self.dataset_sha256,
            "split": self.split,
            "task_name": self.task_name,
            "task_version": self.task_version,
            "scorer_id": self.scorer_id,
            "scorer_version": self.scorer_version,
            "verifier_set": self.verifier_set,
            "code_revision": self.code_revision,
            "decoding_params": dict(self.decoding_params),
            "inspect_version": self.inspect_version,
            "inspect_log_path": self.inspect_log_path,
            "inspect_log_sha256": self.inspect_log_sha256,
            "model": self.model,
            "n": int(self.n),
            "attempts": int(self.attempts),
            "fake": bool(self.fake),
        }
        if self.status == "succeeded" and self.score is not None:
            record["score"] = float(self.score)
        else:
            record["error"] = self.error or "evaluation failed"
        return record


def _inspect_version() -> str:
    try:
        return importlib_metadata.version("inspect-ai")
    except Exception:
        try:
            import inspect_ai

            return str(getattr(inspect_ai, "__version__", "unknown"))
        except Exception:
            return "unknown"


def _completion_text(state: Any) -> str:
    output = getattr(state, "output", None)
    if output is not None:
        completion = getattr(output, "completion", None)
        if completion:
            return str(completion)
    messages = getattr(state, "messages", []) or []
    for message in reversed(messages):
        if getattr(message, "role", None) == "assistant":
            text = getattr(message, "text", None) or getattr(message, "content", "")
            return text if isinstance(text, str) else str(text)
    return ""


@scorer(metrics=[accuracy(), stderr()])
def nightshift_constraint_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        checks: list[dict[str, Any]] = []
        state_metadata = getattr(state, "metadata", None) or {}
        if isinstance(state_metadata, dict):
            checks = list(state_metadata.get("checks") or [])
        if not checks:
            sample = getattr(state, "sample", None)
            sample_metadata = getattr(sample, "metadata", None) or {}
            if isinstance(sample_metadata, dict):
                checks = list(sample_metadata.get("checks") or [])
        if not checks:
            checks = list(_ACTIVE_CHECKS.get(str(getattr(state, "sample_id", "")), []))
        answer = _completion_text(state)
        results = check_text(answer, checks)
        fraction = (
            sum(1 for result in results if result.passed) / len(results)
            if results
            else 1.0
        )
        return Score(
            value=float(fraction),
            answer=answer,
            explanation="; ".join(
                f"{r.verifier_id}@{r.version}={'pass' if r.passed else 'fail'}"
                for r in results
            ),
            metadata={
                "scorer_id": SCORER_ID,
                "scorer_version": SCORER_VERSION,
                "verifier_results": [r.to_record() for r in results],
            },
        )

    return score


def build_task(split: str, rows: list[dict[str, Any]]) -> Task:
    _ACTIVE_CHECKS.clear()
    samples: list[Sample] = []
    for row in rows:
        checks = checks_for_row(row)
        _ACTIVE_CHECKS[str(row["id"])] = checks
        samples.append(
            Sample(
                input=str(row["prompt"]),
                target="satisfy all stated constraints",
                id=str(row["id"]),
                metadata={"checks": checks, "split": split},
            )
        )
    return Task(
        name=TASK_NAME,
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=nightshift_constraint_scorer(),
    )


def _prompt_from_model_input(model_input: Any) -> str:
    messages = model_input
    if hasattr(model_input, "messages"):
        messages = model_input.messages
    if isinstance(messages, (list, tuple)):
        for message in reversed(messages):
            text = getattr(message, "text", None)
            if isinstance(text, str) and text:
                return text
            content = getattr(message, "content", "")
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                for part in content:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str) and part_text:
                        return part_text
    return str(model_input)


def _model_output(content: str) -> ModelOutput:
    """Fake output with explicit usage so Inspect need not estimate tokens."""
    try:
        from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelUsage

        output_tokens = max(1, len(content.split()))
        return ModelOutput(
            model="mockllm/model",
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=content),
                    stop_reason="stop",
                )
            ],
            usage=ModelUsage(
                input_tokens=1,
                output_tokens=output_tokens,
                total_tokens=1 + output_tokens,
            ),
        )
    except Exception:
        return ModelOutput.from_content(model="mockllm/model", content=content)


def fake_outputs_callable(
    rows: list[dict[str, Any]], checkpoint_hash: str, quality: float
):
    """mockllm outputs derived deterministically from the fake checkpoint.

    The per-row threshold is a function of the row (not of the checkpoint),
    so a higher fake_quality checkpoint always passes a superset of rows.
    """
    by_prompt: dict[str, str] = {}
    for row in rows:
        uniform = int(sha256_hex(f"nightshift-fake-eval:{row['id']}")[:8], 16) / 0xFFFFFFFF
        by_prompt[str(row["prompt"])] = (
            synthesize_constrained_text(checks_for_row(row))
            if uniform <= quality
            else violating_text()
        )

    def custom(model_input: Any = None, *args: Any, **kwargs: Any) -> ModelOutput:
        prompt = _prompt_from_model_input(model_input)
        content = by_prompt.get(prompt)
        if content is None:
            for known_prompt, known_output in by_prompt.items():
                if known_prompt in prompt:
                    content = known_output
                    break
        return _model_output(content if content is not None else violating_text())

    return custom


def _score_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return {"C": 1.0, "CORRECT": 1.0, "I": 0.0, "INCORRECT": 0.0}.get(value.upper())
    if isinstance(value, dict) and value:
        values = [_score_value(v) for v in value.values()]
        numbers = [v for v in values if v is not None]
        return sum(numbers) / len(numbers) if numbers else None
    return None


def _locate_log(eval_log: Any, log_dir: Path) -> Path | None:
    location = getattr(eval_log, "location", None) if eval_log is not None else None
    if location:
        path = Path(str(location).removeprefix("file://"))
        if path.is_file():
            return path
    candidates = [
        p for p in log_dir.rglob("*") if p.is_file() and p.suffix in (".eval", ".json")
    ] if log_dir.exists() else []
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _error_text(error: Any) -> str:
    if error is None:
        return ""
    message = getattr(error, "message", None) or getattr(error, "type", None)
    text = str(message) if message else str(error)
    return text[:600]


def _extract_scores(eval_log: Any) -> tuple[list[float], int, str]:
    scores: list[float] = []
    errors = 0
    first_error = _error_text(getattr(eval_log, "error", None))
    for sample in getattr(eval_log, "samples", None) or []:
        sample_error = getattr(sample, "error", None)
        if sample_error:
            errors += 1
            if not first_error:
                first_error = _error_text(sample_error)
            continue
        sample_scores = getattr(sample, "scores", None) or {}
        value: float | None = None
        if isinstance(sample_scores, dict):
            for score_obj in sample_scores.values():
                value = _score_value(getattr(score_obj, "value", score_obj))
                if value is not None:
                    break
        if value is None:
            errors += 1
        else:
            scores.append(value)
    if not scores and getattr(eval_log, "results", None) is not None:
        result_scores = getattr(eval_log.results, "scores", []) or []
        for result_score in result_scores:
            metrics = getattr(result_score, "metrics", {}) or {}
            accuracy_metric = metrics.get("accuracy") if isinstance(metrics, dict) else None
            metric_value = _score_value(getattr(accuracy_metric, "value", None))
            if metric_value is not None:
                scores.append(metric_value)
    return scores, errors, first_error


class Evaluator:
    """Runs Inspect evaluations under the prespecified failure policy."""

    def __init__(
        self,
        registry: DatasetRegistry,
        log_dir: str | Path,
        test_gate: TestAccessGate | None = None,
        ledger: EventLedger | None = None,
        failure_policy: FailurePolicy | None = None,
        decoding_params: dict[str, Any] | None = None,
        code_revision: str = CODE_REVISION,
        records_dir: str | Path | None = None,
    ):
        self.registry = registry
        self.controller = ControllerDataAccess(registry, test_gate)
        self.test_gate = test_gate
        self.log_dir = Path(log_dir)
        self.records_dir = Path(records_dir) if records_dir else self.log_dir / "records"
        self.ledger = ledger
        self.failure_policy = failure_policy or FailurePolicy()
        self.decoding_params = dict(
            decoding_params or {"temperature": 0.0, "max_tokens": 128}
        )
        self.code_revision = code_revision

    @classmethod
    def from_config(
        cls,
        config: Any,
        registry: DatasetRegistry,
        run_dir: str | Path,
        ledger: EventLedger | None = None,
    ) -> "Evaluator":
        root = Path(run_dir)
        return cls(
            registry=registry,
            log_dir=root / "inspect_logs",
            test_gate=TestAccessGate(root / "test_access.json"),
            ledger=ledger,
            failure_policy=FailurePolicy.from_config(config),
            decoding_params={
                "temperature": float(config.evaluation.temperature),
                "max_tokens": int(config.evaluation.max_tokens),
            },
            records_dir=root / "evaluations",
        )

    def claim_test_rows(self, comparison_id: str) -> list[dict[str, Any]]:
        """Claim `test` exactly once and return its rows for all arms."""
        return self.controller.rows("test", "final_comparison", comparison_id)

    def _event(self, record: EvaluationRecord) -> None:
        if self.ledger is not None:
            self.ledger.append(
                "evaluation_finished" if record.succeeded else "evaluation_failed",
                "controller",
                record.to_record(),
            )

    def evaluate_checkpoint(
        self,
        checkpoint_path: str | Path,
        split: str,
        trial_id: str = "",
        pack_version: str = "",
        purpose: str | None = None,
        comparison_id: str | None = None,
        rows: list[dict[str, Any]] | None = None,
        fake: bool | None = None,
        model: str | None = None,
        evaluation_id: str | None = None,
        checkpoint_hash: str | None = None,
    ) -> EvaluationRecord:
        info = validate_checkpoint_dir(checkpoint_path)
        resolved_hash = checkpoint_hash or info.checkpoint_hash
        config_path = Path(checkpoint_path) / "adapter_config.json"
        adapter_config: dict[str, Any] = {}
        if config_path.is_file():
            try:
                adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                adapter_config = {}
        is_fake = bool(adapter_config.get("fake")) if fake is None else bool(fake)
        quality = float(adapter_config.get("fake_quality", 0.5))

        manifest = self.registry.manifest(split)
        inferred = {
            "dev": "dev_feedback",
            "train_prompts": "controller_generation_check",
            "selection": "selection_evaluation",
            "test": "final_comparison",
        }[split]
        resolved_purpose = purpose or inferred
        if rows is None:
            eval_rows = self.controller.rows(split, resolved_purpose, comparison_id)
        else:
            self.controller.check_purpose(split, resolved_purpose, comparison_id)
            eval_rows = list(rows)

        eval_id = evaluation_id or "eval-" + hash_object(
            {"checkpoint": resolved_hash, "split": split, "trial": trial_id,
             "rows": [r.get("id") for r in eval_rows]}
        )[:12]
        record = EvaluationRecord(
            evaluation_id=eval_id,
            status="failed",
            checkpoint_hash=resolved_hash,
            trial_id=trial_id,
            pack_version=pack_version,
            dataset_manifest_id=manifest.id,
            dataset_revision=manifest.revision,
            dataset_sha256=manifest.sha256,
            split=split,
            code_revision=self.code_revision,
            decoding_params=dict(self.decoding_params),
            inspect_version=_inspect_version(),
            model="mockllm/model" if is_fake else str(model or ""),
            n=len(eval_rows),
            fake=is_fake,
        )
        max_attempts = 1 + max(0, int(self.failure_policy.max_retries))
        for attempt in range(1, max_attempts + 1):
            record.attempts = attempt
            attempt_dir = self.log_dir / eval_id / f"attempt-{attempt}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            try:
                succeeded, score, log_path, error = self._run_once(
                    eval_rows, split, resolved_hash, quality, is_fake, model, attempt_dir
                )
            except PolicyError:
                raise
            except Exception as exc:
                succeeded, score, log_path, error = (
                    False, None, _locate_log(None, attempt_dir), f"{type(exc).__name__}: {exc}",
                )
            if log_path is not None:
                record.inspect_log_path = str(log_path)
                record.inspect_log_sha256 = file_sha256(log_path)
            if succeeded and score is not None:
                record.status = "succeeded"
                record.score = float(score)
                record.error = ""
                break
            record.status = "failed"
            record.score = None
            record.error = error or "evaluation failed"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        (self.records_dir / f"{eval_id}.json").write_text(
            json.dumps(record.to_record(), indent=2, sort_keys=True), encoding="utf-8"
        )
        self._event(record)
        return record

    def evaluate_rows(
        self,
        checkpoint_path: str | Path,
        split: str,
        rows: list[dict[str, Any]],
        **kwargs: Any,
    ) -> EvaluationRecord:
        return self.evaluate_checkpoint(checkpoint_path, split, rows=rows, **kwargs)

    def _run_once(
        self,
        rows: list[dict[str, Any]],
        split: str,
        checkpoint_hash: str,
        quality: float,
        fake: bool,
        model: str | None,
        attempt_dir: Path,
    ) -> tuple[bool, float | None, Path | None, str]:
        task = build_task(split, rows)
        if fake:
            inspect_model = get_model(
                "mockllm/model",
                custom_outputs=fake_outputs_callable(rows, checkpoint_hash, quality),
            )
        elif model:
            inspect_model = model
        else:
            raise EvaluationError(
                "real evaluation needs an operator-supplied Inspect model "
                "string for a trusted eval host (not yet validated live)"
            )
        kwargs: dict[str, Any] = {"log_dir": str(attempt_dir)}
        supported = py_inspect.signature(inspect_eval).parameters
        if "fail_on_error" in supported:
            kwargs["fail_on_error"] = False
        if "display" in supported:
            kwargs["display"] = "none"
        logs = inspect_eval(task, model=inspect_model, **kwargs)
        eval_log = logs[0] if logs else None
        log_path = _locate_log(eval_log, attempt_dir)
        if eval_log is None:
            return False, None, log_path, "Inspect returned no evaluation log"
        status = str(getattr(eval_log, "status", ""))
        scores, sample_errors, first_error = _extract_scores(eval_log)
        if status != "success" or sample_errors or len(scores) != len(rows):
            detail = (
                f"status={status or 'unknown'}, scored={len(scores)}/{len(rows)}, "
                f"sample_errors={sample_errors}"
            )
            if first_error:
                detail += f"; first sample error: {first_error}"
            return False, None, log_path, detail
        return True, sum(scores) / len(scores), log_path, ""


def _record_dict(record: Any) -> dict[str, Any]:
    if isinstance(record, EvaluationRecord):
        return record.to_record()
    return dict(record)


def mean_score(records: list[Any]) -> float | None:
    """Mean over succeeded records only; failures never enter as 0 (A13)."""
    scores = [
        float(rec["score"])
        for rec in (_record_dict(r) for r in records)
        if rec.get("status") == "succeeded" and isinstance(rec.get("score"), (int, float))
    ]
    return sum(scores) / len(scores) if scores else None


def summarize_evaluations(records: list[Any]) -> dict[str, Any]:
    dicts = [_record_dict(r) for r in records]
    succeeded = [r for r in dicts if r.get("status") == "succeeded"]
    summary: dict[str, Any] = {
        "n": len(dicts),
        "succeeded": len(succeeded),
        "failures": len(dicts) - len(succeeded),
        "fake": any(bool(r.get("fake")) for r in dicts),
    }
    mean = mean_score(dicts)
    if mean is not None:
        summary["mean_score"] = mean
    return summary


def selection_acceptable(records: list[Any], policy: FailurePolicy | None = None) -> bool:
    """An unresolved selection failure blocks acceptance (spec 3.8)."""
    effective = policy or FailurePolicy()
    if effective.unresolved_selection_failure_accepts:
        return True
    return all(_record_dict(r).get("status") == "succeeded" for r in records)
