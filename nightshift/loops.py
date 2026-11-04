"""Trial runner, outer loop, and final matched comparison.

Inner trials follow CORAL's collaboration pattern (isolated workspaces,
trial-scoped shared notes); the outer loop follows AIDE² (WeCo,
arXiv:2609.26457): edit the researcher pack, screen on a private held-out
split with a greedy accept-if-better gate, expect most proposals to be
rejected, and report only the final paired comparison as evidence.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agents import (
    AgentError,
    agent_record,
    context_for,
    make_researcher,
    make_reviser,
)
from .budget import BudgetStore, LeaseStore, ResourceManager
from .checkpoints import store_checkpoint
from .config import NightshiftConfig
from .data import (
    DatasetRegistry,
    PublicDataView,
    assert_payload_is_public,
    private_markers,
)
from .errors import NightshiftError, PolicyError, ReservationRejected
from .evaluation import (
    EvaluationRecord,
    Evaluator,
    FailurePolicy,
    mean_score,
    record_failure_policy,
)
from .generation import (
    FakeGenerator,
    RealLocalGenerator,
    filter_sample_set,
    load_sample_set,
    make_generator,
    write_sample_set,
)
from .hashing import hash_object
from .ledger import EventLedger
from .pack import (
    check_pack_policy,
    create_pack,
    freeze_pack,
    load_pack_metadata,
    make_scratch_copy,
)
from .providers import JobSpec, RentalRequest, make_provider
from .report import write_report
from .training import FakeTrainer, RealTrainer
from .workspaces import (
    TrialWorkspace,
    append_shared_note,
    create_trial_workspace,
    load_outbox,
    read_shared_notes,
    write_inbox_result,
)

ProgressFn = Callable[[str], None]


def _noop(_message: str) -> None:
    pass


@dataclass
class TrialRecord:
    trial_id: str
    phase: str
    pack_version: str
    seed: int
    researchers: list[str] = field(default_factory=list)
    status: str = "failed"
    end_reason: str = ""
    submitted_checkpoint_hash: str = ""
    submitted_checkpoint_path: str = ""
    trial_gpu_seconds: float = 0.0
    spent_usd: float = 0.0
    request_count: int = 0
    samples_generated: int = 0
    samples_accepted: int = 0
    dev_feedback: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    shared_notes: str = ""
    agent_records: list[dict[str, Any]] = field(default_factory=list)
    fake: bool = False

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded" and bool(self.submitted_checkpoint_hash)

    def to_record(self, public: bool = False) -> dict[str, Any]:
        record: dict[str, Any] = {
            "trial_id": self.trial_id,
            "phase": self.phase,
            "pack_version": self.pack_version,
            "seed": self.seed,
            "researchers": list(self.researchers),
            "status": self.status,
            "end_reason": self.end_reason,
            "submitted_checkpoint_hash": self.submitted_checkpoint_hash,
            "trial_gpu_seconds": float(self.trial_gpu_seconds),
            "spent_usd": float(self.spent_usd),
            "request_count": int(self.request_count),
            "samples_generated": int(self.samples_generated),
            "samples_accepted": int(self.samples_accepted),
            "dev_feedback": list(self.dev_feedback),
            "fake": bool(self.fake),
        }
        if not public:
            record["submitted_checkpoint_path"] = self.submitted_checkpoint_path
            record["history"] = list(self.history)
            record["shared_notes"] = self.shared_notes
            record["agent_records"] = list(self.agent_records)
        return record


class TrialRunner:
    """Controller side of one inner-loop trial (spec 3.6)."""

    def __init__(
        self,
        config: NightshiftConfig,
        run_dir: str | Path,
        ledger: EventLedger,
        resources: ResourceManager,
        registry: DatasetRegistry,
        evaluator: Evaluator,
        generator: Any,
        trainer: Any,
        progress: ProgressFn = _noop,
        max_steps_per_researcher: int = 40,
        wall_clock_seconds: float = 7200.0,
    ):
        self.config = config
        self.run_dir = Path(run_dir)
        self.ledger = ledger
        self.resources = resources
        self.registry = registry
        self.public = PublicDataView(registry)
        self.evaluator = evaluator
        self.generator = generator
        self.trainer = trainer
        self.progress = progress
        self.max_steps = int(max_steps_per_researcher)
        self.wall_clock_seconds = float(wall_clock_seconds)

    def _event(self, event_type: str, payload: dict[str, Any], actor: str = "controller") -> None:
        record = dict(payload)
        record.setdefault("fake", bool(self.config.fake))
        self.ledger.append(event_type, actor, record)

    def _rental_request(
        self, trial_id: str, kind: str, compute: float, setup: float = 2.0,
        transfer: float = 1.0, evaluation: float = 0.0,
    ) -> RentalRequest:
        price = float(self.config.provider.options.get("hourly_price_usd", 1.0))
        return RentalRequest(
            gpu_type=str(self.config.provider.options.get("gpu_type", "fake-gpu")),
            hourly_price_usd=price,
            compute_seconds=float(compute),
            setup_seconds=float(setup),
            transfer_seconds=float(transfer),
            evaluation_seconds=float(evaluation),
            trial_id=trial_id,
            run_id=self.config.run_name,
            job_id=f"{trial_id}-{kind}",
        )

    def _run_provider_job(self, trial_id: str, kind: str, compute: float, **kw: float) -> float:
        request = self._rental_request(trial_id, kind, compute, **kw)
        job = JobSpec(job_id=request.job_id, trial_id=trial_id,
                      duration_seconds=float(compute), kind=kind)
        execution = self.resources.run_and_destroy(request, job)
        if not execution.result.succeeded:
            raise NightshiftError(f"{kind} job failed: {execution.result.reason}")
        return execution.charged_usd

    def run_trial(
        self,
        pack_dir: str | Path,
        pack_version: str,
        trial_id: str,
        seed: int,
        phase: str,
    ) -> TrialRecord:
        researcher_ids = [
            f"researcher-{i + 1}"
            for i in range(max(1, self.config.outer.researchers_per_trial))
        ]
        pack_path = Path(pack_dir)
        trial = create_trial_workspace(
            self.run_dir / "trials", trial_id, researcher_ids,
            pack_version=pack_version, seed=seed,
            initial_notes_dir=pack_path / "initial_notes",
        )
        record = TrialRecord(
            trial_id=trial_id, phase=phase, pack_version=pack_version,
            seed=int(seed), researchers=list(researcher_ids),
            fake=bool(self.config.fake),
        )
        allowed_keys = set(load_pack_metadata(pack_path).editable_keys)
        train_rows = self.public.train_prompts()
        state = _TrialState(self, trial, record, pack_path, train_rows)
        agents = {
            rid: make_researcher(self.config, rid, primary=(index == 0))
            for index, rid in enumerate(researcher_ids)
        }
        histories: dict[str, list[dict[str, Any]]] = {rid: [] for rid in researcher_ids}
        self._event("trial_started", {
            "trial_id": trial_id, "phase": phase, "pack_version": pack_version,
            "seed": int(seed), "researchers": researcher_ids,
        })
        self.progress(
            f"phase={phase} trial={trial_id} pack={pack_version[:12]} seed={seed}"
        )
        start = time.monotonic()
        active = list(researcher_ids)
        step = 0
        end_reason = "all_finished"
        while active:
            if step >= self.max_steps:
                end_reason = "max_steps"
                break
            if time.monotonic() - start > self.wall_clock_seconds:
                end_reason = "wall_clock_exceeded"
                break
            step += 1
            progressed = False
            for rid in list(active):
                agent = agents[rid]
                context = context_for(
                    trial, rid, pack_path, train_rows,
                    histories[rid], read_shared_notes(trial), step,
                )
                agent_error = ""
                try:
                    agent.act(context)
                except (AgentError, Exception) as exc:  # agent failure ends only that agent
                    agent_error = f"{type(exc).__name__}: {exc}"
                record.agent_records.append(agent_record(agent, step))
                requests_made = state.drain_outbox(rid, allowed_keys, histories[rid])
                progressed = progressed or requests_made > 0
                if state.budget_exhausted:
                    end_reason = "budget_exhausted"
                    active = []
                    break
                if agent_error:
                    histories[rid].append({
                        "request": {"type": "agent"},
                        "result": {"status": "failed", "reason": agent_error,
                                   "fake": record.fake},
                    })
                    active.remove(rid)
                elif getattr(agent, "done", False):
                    active.remove(rid)
                elif requests_made == 0:
                    active.remove(rid)
            if not progressed and not active:
                break
        record.end_reason = end_reason
        record.history = [entry for rid in researcher_ids for entry in histories[rid]]
        record.shared_notes = read_shared_notes(trial)
        record.request_count = len(record.history)
        record.trial_gpu_seconds = self.resources.budget.trial_charged_seconds(trial_id)
        state.choose_submission()
        record.status = "succeeded" if record.submitted_checkpoint_hash else "failed"
        if not record.submitted_checkpoint_hash and end_reason == "all_finished":
            record.end_reason = "no_submission"
        controller_dir = trial.controller_dir()
        (controller_dir / "trial_record.json").write_text(
            json.dumps(record.to_record(), indent=2, sort_keys=True), encoding="utf-8"
        )
        self._event("trial_finished", {
            "trial_id": trial_id, "phase": phase, "status": record.status,
            "end_reason": record.end_reason,
            "submitted_checkpoint_hash": record.submitted_checkpoint_hash,
            "trial_gpu_seconds": record.trial_gpu_seconds,
            "spent_usd": record.spent_usd,
        })
        return record


class _TrialState:
    """Per-trial registries and validated request execution."""

    def __init__(self, runner: TrialRunner, trial: TrialWorkspace,
                 record: TrialRecord, pack_path: Path, train_rows: list[dict[str, Any]]):
        self.runner = runner
        self.trial = trial
        self.record = record
        self.pack_path = pack_path
        self.train_rows = train_rows
        self.sample_sets: dict[str, Path] = {}
        self.checkpoints: dict[str, Path] = {}
        self.submissions: list[dict[str, Any]] = []
        self.dev_scores: dict[str, float] = {}
        self.budget_exhausted = False

    def drain_outbox(self, researcher_id: str, allowed_keys: set[str],
                     history: list[dict[str, Any]]) -> int:
        runner = self.runner
        outbox = self.trial.outbox(researcher_id)
        valid, errors = load_outbox(self.trial, researcher_id, allowed_keys)
        processed = self.trial.controller_dir() / "processed" / researcher_id
        processed.mkdir(parents=True, exist_ok=True)
        made = 0
        valid_ids = {request.request_id for request in valid}
        for request in valid:
            result = self.execute(request)
            write_inbox_result(self.trial, researcher_id, request.request_id, result)
            history.append({"request": request.to_record(), "result": result})
            made += 1
        for error in errors:
            result = {"status": "failed", "reason": error["error"],
                      "file": error["file"], "fake": self.record.fake}
            history.append({"request": {"type": "invalid"}, "result": result})
            made += 1
        for path in sorted(outbox.glob("*.json")):
            dest = processed / path.name
            counter = 1
            while dest.exists():
                dest = processed / f"{path.stem}-{counter}{path.suffix}"
                counter += 1
            shutil.move(str(path), str(dest))
        return made if (valid_ids or errors) else 0

    def execute(self, request: Any) -> dict[str, Any]:
        runner = self.runner
        trial_id = self.record.trial_id
        result: dict[str, Any] = {
            "request_id": request.request_id, "type": request.type,
            "status": "succeeded", "fake": self.record.fake,
        }
        try:
            if request.type == "generate":
                result.update(self._generate(request))
            elif request.type == "filter":
                result.update(self._filter(request))
            elif request.type == "train":
                result.update(self._train(request))
            elif request.type == "evaluate_dev":
                result.update(self._evaluate_dev(request))
            elif request.type == "submit":
                result.update(self._submit(request))
            elif request.type == "note":
                assert_payload_is_public(
                    {"text": request.payload["text"]}, runner.registry
                )
                append_shared_note(self.trial, request.researcher_id, request.payload["text"])
                result["note"] = "appended"
                runner._event("shared_note", {
                    "trial_id": trial_id, "researcher": request.researcher_id,
                }, actor=request.researcher_id)
        except ReservationRejected as exc:
            self.budget_exhausted = True
            result.update({"status": "failed", "reason": f"budget: {exc}"})
        except Exception as exc:
            result.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
        return result

    def _generate(self, request: Any) -> dict[str, Any]:
        runner = self.runner
        payload = request.payload
        prompt_ids = list(payload["prompt_ids"])
        compute = 2.0 + float(len(prompt_ids) * int(payload["count"]))
        if isinstance(runner.generator, RealLocalGenerator) and not runner.config.fake:
            rental = runner._rental_request(self.record.trial_id, "generate", compute)
            samples = runner.generator.generate(
                self.train_rows, prompt_ids, rental,
                count=int(payload["count"]), seed=self.record.seed,
                sampling_params=payload["sampling_params"],
            )
        else:
            self.record.spent_usd += runner._run_provider_job(
                self.record.trial_id, "generate", compute
            )
            samples = runner.generator.generate(
                self.train_rows, prompt_ids, count=int(payload["count"]),
                seed=self.record.seed, sampling_params=payload["sampling_params"],
            )
        dest = self.trial.controller_dir() / "sample_sets" / request.request_id
        sample_set = write_sample_set(samples, dest)
        self.sample_sets[sample_set.sample_set_id] = dest
        self.record.samples_generated += len(samples)
        runner._event("generation_finished", {
            "trial_id": self.record.trial_id, "sample_set_id": sample_set.sample_set_id,
            "n": len(samples), "fake": sample_set.fake,
        }, actor=request.researcher_id)
        return {"sample_set_id": sample_set.sample_set_id, "n": len(samples),
                "fake": sample_set.fake}

    def _load_samples(self, sample_set_id: str) -> Any:
        path = self.sample_sets.get(sample_set_id)
        if path is None:
            raise NightshiftError(f"unknown sample set for this trial: {sample_set_id}")
        return load_sample_set(path)

    def _filter(self, request: Any) -> dict[str, Any]:
        runner = self.runner
        sample_set = self._load_samples(request.payload["sample_set_id"])
        verifier_ids = request.payload.get("verifier_ids") or None
        outcome = filter_sample_set(sample_set.samples, self.train_rows, verifier_ids)
        dest = self.trial.controller_dir() / "sample_sets" / request.request_id
        accepted_set = write_sample_set(outcome.accepted, dest)
        self.sample_sets[accepted_set.sample_set_id] = dest
        self.record.samples_accepted += len(outcome.accepted)
        runner._event("filter_finished", {
            "trial_id": self.record.trial_id,
            "sample_set_id": accepted_set.sample_set_id,
            **outcome.to_record(fake=accepted_set.fake),
        }, actor=request.researcher_id)
        return {"sample_set_id": accepted_set.sample_set_id,
                **outcome.to_record(fake=accepted_set.fake)}

    def _train(self, request: Any) -> dict[str, Any]:
        runner = self.runner
        sample_set = self._load_samples(request.payload["sample_set_id"])
        checkpoint_dir = (
            self.trial.controller_dir() / "checkpoints" / request.request_id
        )
        work_dir = self.trial.controller_dir() / "training" / request.request_id
        overrides = request.payload.get("recipe_overrides") or {}
        if isinstance(runner.trainer, RealTrainer) and not runner.config.fake:
            rental = runner._rental_request(
                self.record.trial_id, "train", 60.0, setup=10.0,
                transfer=5.0, evaluation=10.0,
            )
            trained = runner.trainer.train(
                sample_set.samples, checkpoint_dir=checkpoint_dir,
                pack_dir=self.pack_path, request=rental,
                overrides=overrides, seed=self.record.seed, work_dir=work_dir,
            )
        else:
            self.record.spent_usd += runner._run_provider_job(
                self.record.trial_id, "train", 60.0, setup=10.0,
                transfer=5.0, evaluation=10.0,
            )
            trained = runner.trainer.train(
                sample_set.samples, checkpoint_dir=checkpoint_dir,
                pack_dir=self.pack_path, config=runner.config,
                overrides=overrides, seed=self.record.seed, work_dir=work_dir,
            )
        stored = store_checkpoint(
            trained.checkpoint_path, runner.run_dir / "checkpoints", fake=trained.fake
        )
        self.checkpoints[stored.checkpoint_hash] = stored.path
        runner._event("training_finished", {
            "trial_id": self.record.trial_id,
            "checkpoint_hash": stored.checkpoint_hash,
            "n_samples": trained.n_samples, "fake": trained.fake,
        }, actor=request.researcher_id)
        runner._event("checkpoint_validated", {
            "trial_id": self.record.trial_id,
            "checkpoint_hash": stored.checkpoint_hash, "fake": trained.fake,
        })
        return {"checkpoint_id": stored.checkpoint_hash,
                "checkpoint_hash": stored.checkpoint_hash,
                "n_samples": trained.n_samples, "fake": trained.fake}

    def _evaluate_dev(self, request: Any) -> dict[str, Any]:
        runner = self.runner
        checkpoint_id = request.payload["checkpoint_id"]
        path = self.checkpoints.get(checkpoint_id)
        if path is None:
            raise NightshiftError(f"unknown checkpoint for this trial: {checkpoint_id}")
        if runner.config.fake:
            self.record.spent_usd += runner._run_provider_job(
                self.record.trial_id, "evaluate_dev", 10.0, evaluation=10.0
            )
        evaluation = runner.evaluator.evaluate_checkpoint(
            path, split="dev", trial_id=self.record.trial_id,
            pack_version=self.record.pack_version,
            model=_eval_model_string(runner.config),
            fake=runner.config.fake,
        )
        feedback = PublicDataView.dev_feedback(
            evaluation.score, evaluation.n,
            failures=0 if evaluation.succeeded else 1,
            status=evaluation.status, fake=evaluation.fake,
        )
        feedback["checkpoint_id"] = checkpoint_id
        feedback["evaluation_id"] = evaluation.evaluation_id
        feedback["inspect_log_sha256"] = evaluation.inspect_log_sha256
        self.record.dev_feedback.append(dict(feedback))
        if evaluation.succeeded and evaluation.score is not None:
            self.dev_scores[checkpoint_id] = float(evaluation.score)
        return feedback

    def _submit(self, request: Any) -> dict[str, Any]:
        checkpoint_id = request.payload["checkpoint_id"]
        if checkpoint_id not in self.checkpoints:
            raise NightshiftError(f"unknown checkpoint for this trial: {checkpoint_id}")
        self.submissions.append({
            "researcher": request.researcher_id, "checkpoint_id": checkpoint_id,
            "dev_score": self.dev_scores.get(checkpoint_id),
        })
        self.runner._event("checkpoint_submitted", {
            "trial_id": self.record.trial_id, "checkpoint_hash": checkpoint_id,
        }, actor=request.researcher_id)
        return {"submitted_checkpoint_id": checkpoint_id}

    def choose_submission(self) -> None:
        if not self.submissions:
            return
        best = max(
            self.submissions,
            key=lambda s: (s["dev_score"] is not None, s["dev_score"] or 0.0),
        )
        self.record.submitted_checkpoint_hash = str(best["checkpoint_id"])
        self.record.submitted_checkpoint_path = str(
            self.checkpoints[best["checkpoint_id"]]
        )


def _eval_model_string(config: NightshiftConfig) -> str | None:
    model = config.provider.options.get("eval_model")
    if model:
        return str(model)
    eval_section = (config.raw or {}).get("evaluation") or {}
    if isinstance(eval_section, dict) and eval_section.get("model"):
        return str(eval_section["model"])
    return None


_PRIVATE_KEY_WORDS = {"selection", "test", "per_item", "per-item"}


def _sanitize_public(value: Any, markers: set[str]) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _PRIVATE_KEY_WORDS:
                continue
            if "selection" in lowered or lowered.endswith("_path") or lowered == "path":
                continue
            if lowered.startswith("test_") or lowered.endswith("_test"):
                continue
            clean[str(key)] = _sanitize_public(item, markers)
        return clean
    if isinstance(value, list):
        return [_sanitize_public(item, markers) for item in value]
    if isinstance(value, str):
        if any(marker and marker in value for marker in markers):
            return "[removed: private material]"
        return value
    return value


def build_reviser_input(
    pack_dir: str | Path,
    pack_version: str,
    trials: list[Any],
    registry: DatasetRegistry,
) -> dict[str, Any]:
    """Permitted public material only (spec 3.10 step 2, A10)."""
    pack_path = Path(pack_dir)
    pack_files: dict[str, str] = {}
    for path in sorted(pack_path.rglob("*")):
        if path.is_file() and not path.is_symlink():
            try:
                pack_files[path.relative_to(pack_path).as_posix()] = (
                    path.read_text(encoding="utf-8")[:20000]
                )
            except UnicodeDecodeError:
                continue
    markers = private_markers(registry)
    trial_records: list[dict[str, Any]] = []
    for trial in trials:
        raw = trial.to_record() if isinstance(trial, TrialRecord) else dict(trial)
        public_trial = {
            "trial_id": raw.get("trial_id"),
            "pack_version": raw.get("pack_version"),
            "seed": raw.get("seed"),
            "status": raw.get("status"),
            "samples_generated": raw.get("samples_generated"),
            "samples_accepted": raw.get("samples_accepted"),
            "history": raw.get("history", []),
            "dev_feedback": raw.get("dev_feedback", []),
            "shared_notes": raw.get("shared_notes", ""),
            "fake": bool(raw.get("fake")),
        }
        trial_records.append(_sanitize_public(public_trial, markers))
    payload = {
        "pack_version": pack_version,
        "pack_files": pack_files,
        "trials": trial_records,
    }
    payload = _sanitize_public(payload, markers)
    assert_payload_is_public(payload, registry)
    return payload


@dataclass
class ScreeningResult:
    pack_version: str
    trials: list[TrialRecord] = field(default_factory=list)
    evaluations: list[EvaluationRecord] = field(default_factory=list)

    @property
    def failures(self) -> int:
        trial_failures = sum(1 for t in self.trials if not t.succeeded)
        eval_failures = sum(1 for e in self.evaluations if not e.succeeded)
        return trial_failures + eval_failures

    def scores_by_seed(self) -> dict[int, float]:
        scores: dict[int, float] = {}
        by_trial = {t.trial_id: t for t in self.trials}
        for evaluation in self.evaluations:
            trial = by_trial.get(evaluation.trial_id)
            if trial is not None and evaluation.succeeded and evaluation.score is not None:
                scores[trial.seed] = float(evaluation.score)
        return scores

    def to_record(self) -> dict[str, Any]:
        mean = mean_score(self.evaluations)
        record: dict[str, Any] = {
            "pack_version": self.pack_version,
            "trials": len(self.trials),
            "failures": self.failures,
            "fake": any(t.fake for t in self.trials),
        }
        if mean is not None:
            record["selection_mean"] = mean
        return record


def summarize_comparison(
    pairs: list[dict[str, Any]],
    margin: float = 0.0,
    bootstrap_samples: int = 1000,
    seed: int = 0,
    fake: bool = False,
) -> dict[str, Any]:
    """Paired comparison math; failed trials stay in the denominator (A17)."""
    diffs: list[float] = []
    original_failures = final_failures = 0
    original_compute = final_compute = 0.0
    for pair in pairs:
        original = pair.get("original") or {}
        final = pair.get("final") or {}
        original_compute += float(original.get("compute_seconds", 0.0))
        final_compute += float(final.get("compute_seconds", 0.0))
        original_ok = original.get("status") == "succeeded" and isinstance(
            original.get("score"), (int, float))
        final_ok = final.get("status") == "succeeded" and isinstance(
            final.get("score"), (int, float))
        if not original_ok:
            original_failures += 1
        if not final_ok:
            final_failures += 1
        if original_ok and final_ok:
            difference = float(final["score"]) - float(original["score"])
            pair["difference"] = difference
            diffs.append(difference)
        else:
            pair.pop("difference", None)
    mean_diff = sum(diffs) / len(diffs) if diffs else None
    ci: list[float | None] = [None, None]
    if diffs:
        rng = random.Random(int(seed))
        means = sorted(
            sum(rng.choice(diffs) for _ in diffs) / len(diffs)
            for _ in range(max(1, int(bootstrap_samples)))
        )
        ci = [means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]]
    evidence = bool(
        not fake
        and mean_diff is not None
        and mean_diff >= float(margin)
        and ci[0] is not None
        and ci[0] > 0
        and original_failures == 0
        and final_failures == 0
    )
    result: dict[str, Any] = {
        "pairs": pairs,
        "pair_denominator": len(pairs),
        "scored_pairs": len(diffs),
        "original_failures": original_failures,
        "final_failures": final_failures,
        "original_compute_seconds": original_compute,
        "final_compute_seconds": final_compute,
        "bootstrap_ci": ci,
        "evidence_of_improvement": evidence,
        "fake": bool(fake),
    }
    if mean_diff is not None:
        result["mean_difference"] = mean_diff
    result["conclusion"] = (
        "evidence of improvement" if evidence else "no evidence of improvement"
    )
    return result


class NightshiftRun:
    """Whole pipeline: outer rounds, then the final matched comparison."""

    def __init__(self, config: NightshiftConfig, progress: ProgressFn = _noop):
        self.config = config
        self.progress = progress
        self.run_dir = Path(config.output_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = EventLedger(self.run_dir / "ledger.jsonl")
        self.registry = DatasetRegistry.from_config(config)
        self.provider = make_provider(config.provider.name, config.provider.options)
        self.resources = ResourceManager(
            budget_store=BudgetStore(
                config.budgets.budget_file,
                ceiling_usd=config.budgets.aggregate_ceiling_usd,
            ),
            lease_store=LeaseStore(self.run_dir / "leases.json"),
            provider=self.provider,
            per_trial_gpu_seconds=config.budgets.per_trial_gpu_seconds,
            max_rental_lifetime_seconds=config.budgets.max_rental_lifetime_seconds,
            max_concurrent_rentals=config.budgets.max_concurrent_rentals,
            run_id=config.run_name,
            ledger=self.ledger,
        )
        self.evaluator = Evaluator.from_config(
            config, self.registry, self.run_dir, ledger=self.ledger
        )
        self.generator = make_generator(config, self.resources)
        self.trainer = FakeTrainer() if config.fake else RealTrainer(self.resources, config)
        self.reviser = make_reviser(config)
        self.trial_runner = TrialRunner(
            config, self.run_dir, self.ledger, self.resources,
            self.registry, self.evaluator, self.generator, self.trainer,
            progress=self._progress,
        )
        self.pack_paths: dict[str, Path] = {}
        self.screening_cache: dict[str, ScreeningResult] = {}
        self.all_trials: list[TrialRecord] = []
        self.phase = "setup"

    def _progress(self, message: str) -> None:
        spent = self.resources.budget.committed_and_reserved_usd()
        self.progress(
            f"{message} spend=${spent:.4f}/${self.config.budgets.aggregate_ceiling_usd:.2f} "
            f"last_event=seq {self.ledger.next_seq - 1}"
            + (" [fake]" if self.config.fake else "")
        )

    def _event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = dict(payload)
        record.setdefault("fake", bool(self.config.fake))
        self.ledger.append(event_type, "controller", record)

    def _source_pack(self) -> Path:
        if self.config.pack_dir is not None:
            pack_dir = Path(self.config.pack_dir)
            if not pack_dir.is_dir():
                create_pack(pack_dir)
            return pack_dir
        fallback = self.run_dir / "source_pack"
        if not fallback.is_dir():
            create_pack(fallback)
        return fallback

    def _screen(self, pack_version: str, round_label: str) -> ScreeningResult:
        if pack_version in self.screening_cache:
            return self.screening_cache[pack_version]
        pack_dir = self.pack_paths[pack_version]
        seeds = [
            self.config.seed + index
            for index in range(max(1, self.config.outer.screen_trials))
        ]
        result = ScreeningResult(pack_version=pack_version)
        for index, seed in enumerate(seeds):
            trial_id = f"screen-{round_label}-{pack_version[:8]}-s{seed}-i{index}"
            trial = self.trial_runner.run_trial(
                pack_dir, pack_version, trial_id, seed, phase=f"screening-{round_label}"
            )
            result.trials.append(trial)
            self.all_trials.append(trial)
            if trial.succeeded:
                evaluation = self.evaluator.evaluate_checkpoint(
                    trial.submitted_checkpoint_path, split="selection",
                    trial_id=trial.trial_id, pack_version=pack_version,
                    purpose="selection_evaluation",
                    model=_eval_model_string(self.config), fake=self.config.fake,
                )
                result.evaluations.append(evaluation)
        self.screening_cache[pack_version] = result
        return result

    def _outer_round(
        self, round_no: int, incumbent_version: str
    ) -> tuple[str, dict[str, Any]]:
        config = self.config
        self.phase = f"outer-round-{round_no}"
        incumbent_dir = self.pack_paths[incumbent_version]
        incumbent_screen = self._screen(incumbent_version, f"r{round_no}-incumbent")
        round_dir = self.run_dir / "outer" / f"round-{round_no}"
        round_dir.mkdir(parents=True, exist_ok=True)
        reviser_input = build_reviser_input(
            incumbent_dir, incumbent_version, list(incumbent_screen.trials), self.registry
        )
        (round_dir / "reviser_input.json").write_text(
            json.dumps(reviser_input, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._event("reviser_started", {
            "round": round_no, "incumbent_pack": incumbent_version,
            "requested_model": config.models.reviser,
        })
        scratch = make_scratch_copy(incumbent_dir, round_dir / "candidate_pack")
        reviser_result = self.reviser.revise(
            scratch, public_input=reviser_input, round_no=round_no
        )
        self._event("reviser_finished", {
            "round": round_no, **reviser_result.to_record(),
        }, )
        round_record: dict[str, Any] = {
            "round": round_no,
            "incumbent_version": incumbent_version,
            "candidate_version": "",
            "candidate_frozen": False,
            "accepted": False,
            "reasons": [],
            "fake": bool(config.fake),
            "incumbent_screening": incumbent_screen.to_record(),
        }
        try:
            changed = check_pack_policy(scratch, incumbent_dir)
        except PolicyError as exc:
            round_record["reasons"] = [f"pack policy rejected the candidate: {exc}"]
            self._event("acceptance_decision", dict(round_record))
            return incumbent_version, round_record
        if not changed:
            round_record["reasons"] = ["candidate made no pack change"]
            self._event("acceptance_decision", dict(round_record))
            return incumbent_version, round_record
        candidate_version, candidate_dir = freeze_pack(
            scratch, self.run_dir / "packs"
        )
        self.pack_paths[candidate_version] = candidate_dir
        round_record.update({
            "candidate_version": candidate_version, "candidate_frozen": True,
            "changed_files": changed,
        })
        self._event("pack_frozen", {
            "round": round_no, "pack_version": candidate_version, "changed": changed,
        })
        candidate_screen = self._screen(candidate_version, f"r{round_no}-candidate")
        round_record["candidate_screening"] = candidate_screen.to_record()
        incumbent_scores = incumbent_screen.scores_by_seed()
        candidate_scores = candidate_screen.scores_by_seed()
        seeds = sorted(set(incumbent_scores) & set(candidate_scores))
        differences = [candidate_scores[s] - incumbent_scores[s] for s in seeds]
        mean_difference = (
            sum(differences) / len(differences) if differences else None
        )
        reasons: list[str] = []
        accepted = True
        if incumbent_screen.failures or candidate_screen.failures:
            accepted = False
            reasons.append(
                "unresolved screening failure(s): incumbent "
                f"{incumbent_screen.failures}, candidate {candidate_screen.failures}"
            )
        if mean_difference is None:
            accepted = False
            reasons.append("no paired selection scores")
        elif mean_difference < float(config.outer.acceptance_margin):
            accepted = False
            reasons.append(
                f"mean paired difference {mean_difference:.4f} below margin "
                f"{config.outer.acceptance_margin}"
            )
        if accepted:
            reasons.append(
                f"mean paired difference {mean_difference:.4f} meets margin "
                f"{config.outer.acceptance_margin} with no unresolved failures"
            )
        round_record.update({
            "accepted": bool(accepted), "reasons": reasons,
            "paired_seeds": seeds,
        })
        if mean_difference is not None:
            round_record["mean_paired_difference"] = mean_difference
        self._event("acceptance_decision", {
            "round": round_no, "candidate_version": candidate_version,
            "accepted": bool(accepted), "reasons": reasons,
            "mean_paired_difference": mean_difference,
        })
        self._progress(
            f"phase=outer round={round_no} candidate={candidate_version[:12]} "
            f"accepted={bool(accepted)}"
        )
        return (candidate_version if accepted else incumbent_version), round_record

    def _final_comparison(self, original_version: str, final_version: str) -> dict[str, Any]:
        self.phase = "final-comparison"
        comparison_id = "comparison-" + hash_object({
            "run": self.config.run_name, "original": original_version,
            "final": final_version, "seed": self.config.seed,
        })[:12]
        # The one and only test claim for this comparison id (A16).
        test_rows = self.evaluator.claim_test_rows(comparison_id)
        self._event("comparison_started", {
            "comparison_id": comparison_id, "original": original_version,
            "final": final_version,
            "trials_per_arm": self.config.comparison.comparison_trials,
        })
        pairs: list[dict[str, Any]] = []
        arms = [("original", original_version), ("final", final_version)]
        for index in range(max(1, self.config.comparison.comparison_trials)):
            seed = self.config.seed + 1000 + index
            pair: dict[str, Any] = {"seed": seed, "fake": bool(self.config.fake)}
            for arm_name, version in arms:
                trial_id = f"compare-{arm_name}-{version[:8]}-s{seed}-i{index}"
                trial = self.trial_runner.run_trial(
                    self.pack_paths[version], version, trial_id, seed,
                    phase="final-comparison",
                )
                self.all_trials.append(trial)
                arm: dict[str, Any] = {
                    "trial_id": trial.trial_id,
                    "pack_version": version,
                    "compute_seconds": trial.trial_gpu_seconds,
                    "fake": trial.fake,
                }
                if trial.succeeded:
                    evaluation = self.evaluator.evaluate_rows(
                        trial.submitted_checkpoint_path, split="test",
                        rows=test_rows, trial_id=trial.trial_id,
                        pack_version=version, purpose="final_comparison",
                        comparison_id=comparison_id,
                        model=_eval_model_string(self.config),
                        fake=self.config.fake,
                    )
                    arm["evaluation_id"] = evaluation.evaluation_id
                    if evaluation.succeeded and evaluation.score is not None:
                        arm.update({"status": "succeeded", "score": float(evaluation.score)})
                    else:
                        arm.update({"status": "failed", "error": evaluation.error})
                else:
                    arm.update({"status": "failed", "error": trial.end_reason})
                pair[arm_name] = arm
            pairs.append(pair)
        comparison = summarize_comparison(
            pairs,
            margin=self.config.comparison.margin,
            bootstrap_samples=self.config.comparison.bootstrap_samples,
            seed=self.config.seed,
            fake=self.config.fake,
        )
        comparison["comparison_id"] = comparison_id
        (self.run_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._event("comparison_result", {
            "comparison_id": comparison_id,
            "pair_denominator": comparison["pair_denominator"],
            "scored_pairs": comparison["scored_pairs"],
            "mean_difference": comparison.get("mean_difference"),
            "bootstrap_ci": comparison["bootstrap_ci"],
            "original_failures": comparison["original_failures"],
            "final_failures": comparison["final_failures"],
            "conclusion": comparison["conclusion"],
        })
        return comparison

    def run(self) -> dict[str, Any]:
        config = self.config
        if isinstance(self.generator, FakeGenerator):
            pass  # fake components are labelled everywhere below
        self.registry.verify_all()
        self._event("run_started", {
            "run_name": config.run_name, "seed": config.seed,
            "researcher_model": config.models.researcher,
            "reviser_model": config.models.reviser,
            "base_model": config.models.base_model,
            "base_revision": config.models.base_revision,
            "provider": config.provider.name,
            "ceiling_usd": config.budgets.aggregate_ceiling_usd,
        })
        record_failure_policy(
            self.ledger, FailurePolicy.from_config(config), fake=config.fake
        )
        original_version, original_dir = freeze_pack(
            self._source_pack(), self.run_dir / "packs"
        )
        self.pack_paths[original_version] = original_dir
        self._event("pack_frozen", {
            "pack_version": original_version, "role": "original",
        })
        self._progress(
            f"phase=start original_pack={original_version[:12]} "
            f"rounds={config.outer.max_rounds}"
        )
        incumbent = original_version
        rounds: list[dict[str, Any]] = []
        for round_no in range(1, int(config.outer.max_rounds) + 1):
            if self.resources.budget.remaining_usd() <= 1e-9:
                self._event("run_budget_exhausted", {"round": round_no})
                break
            incumbent, round_record = self._outer_round(round_no, incumbent)
            rounds.append(round_record)
        comparison = self._final_comparison(original_version, incumbent)
        self.phase = "report"
        summary = self.summary(original_version, incumbent, rounds, comparison)
        write_report(self.run_dir, summary)
        self._event("report_written", {"report": "report.md"})
        self._event("run_finished", {
            "status": "succeeded", "final_pack": incumbent,
            "conclusion": comparison["conclusion"],
        })
        # Keep run.json/.md in step with the terminal ledger state.
        summary = self.summary(original_version, incumbent, rounds, comparison)
        write_report(self.run_dir, summary)
        self._progress(
            f"phase=done report={self.run_dir / 'report.md'} "
            f"conclusion={comparison['conclusion']}"
        )
        return summary

    def summary(
        self,
        original_version: str,
        final_version: str,
        rounds: list[dict[str, Any]],
        comparison: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "run_name": self.config.run_name,
            "status": "succeeded",
            "phase": self.phase,
            "fake": bool(self.config.fake),
            "original_pack_version": original_version,
            "final_pack_version": final_version,
            "budgets": {
                "ceiling_usd": self.config.budgets.aggregate_ceiling_usd,
                "committed_and_reserved_usd": (
                    self.resources.budget.committed_and_reserved_usd()
                ),
                "fake": bool(getattr(self.provider, "fake", False)),
            },
            "rounds": rounds,
            "comparison": comparison,
            "trials": [trial.to_record(public=True) for trial in self.all_trials],
        }


def run_pipeline(config: NightshiftConfig, progress: ProgressFn | None = None) -> dict[str, Any]:
    return NightshiftRun(config, progress or _noop).run()
