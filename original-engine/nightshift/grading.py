"""The evaluation pipeline, strict trusted-grader parsing, and the outcome decision (docs/ARCHITECTURE.md 4.4, 6.3, 8).

Guarantees
----------
* ``evaluate`` runs the untrusted code only on a throwaway copy whose non-scratch files are hash-checked before and after,
  hands the grader only sanitised staged outputs, and hashes ``trusted/`` before and after every grader call.
* ``run_trusted_grader`` accepts only exit code 0 and a last stdout line that is strict JSON (no NaN/Infinity) with an
  echoed split, a boolean ``valid`` and finite numbers; anything else raises ``GraderError`` (a ``ValueError``).
* ``decide_outcome`` is the one place the improvement rule lives. Sealed numbers are never copied into error strings.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from . import sandbox
from .sandbox import RunResult, Workspace
from .task import conditions_hash, manifest_hash, trusted_manifest
from .types import GradeResult, OutcomeDecision, ProtectedPathViolation, Purpose, TaskSpec

GRADER_STDOUT_CAP = 64 * 1024
FAILURE_PRECEDENCE = ("infra", "timeout", "crash", "invalid")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


# --------------------------------------------------------------------------------------------------
# critical values
# --------------------------------------------------------------------------------------------------
class _TTable(dict):
    """Critical values keyed by degrees of freedom (n - 1), non-increasing in the degrees of freedom.

    A degree of freedom that is not tabulated takes the value of the largest tabulated one below it (a slightly wider margin
    than the exact quantile, never a narrower one); beyond the last tabulated one the normal quantile ``far`` applies. The
    normal quantile is below every ``t`` value, so the step down at the end of the table is still monotone. ``df < 1``
    uses ``df == 1``."""

    def __init__(self, data: Mapping[int, float], far: float):
        super().__init__(data)
        self.far = far

    def __missing__(self, df: int) -> float:
        if df < 1:
            return self[1]
        below = [k for k in self if k <= df]
        return self[max(below)] if df <= max(self) else self.far


# Exact Student-t quantiles, 4 decimals (regularised incomplete beta + bisection, see tests/test_grading.py ``_t_quantile``).
# df 1..10 keep the three-decimal values the contract has always shipped (so no existing result changes); df 11..30 are new.
#   df   11      12      13      14      15      16      17      18      19      20      21      22      23      24      25
# T95    1.7959  1.7823  1.7709  1.7613  1.7531  1.7459  1.7396  1.7341  1.7291  1.7247  1.7207  1.7171  1.7139  1.7109  1.7081
# T975   2.2010  2.1788  2.1604  2.1448  2.1314  2.1199  2.1098  2.1009  2.0930  2.0860  2.0796  2.0739  2.0687  2.0639  2.0595
#   df   26      27      28      29      30      40      60      120
# T95    1.7056  1.7033  1.7011  1.6991  1.6973  1.6839  1.6706  1.6577
# T975   2.0555  2.0518  2.0484  2.0452  2.0423  2.0211  2.0003  1.9799
T95 = _TTable({1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
               11: 1.7959, 12: 1.7823, 13: 1.7709, 14: 1.7613, 15: 1.7531, 16: 1.7459, 17: 1.7396, 18: 1.7341, 19: 1.7291,
               20: 1.7247, 21: 1.7207, 22: 1.7171, 23: 1.7139, 24: 1.7109, 25: 1.7081, 26: 1.7056, 27: 1.7033, 28: 1.7011,
               29: 1.6991, 30: 1.6973, 40: 1.6839, 60: 1.6706, 120: 1.6577}, 1.645)
T975 = _TTable({1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314, 16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930,
                20: 2.0860, 21: 2.0796, 22: 2.0739, 23: 2.0687, 24: 2.0639, 25: 2.0595, 26: 2.0555, 27: 2.0518, 28: 2.0484,
                29: 2.0452, 30: 2.0423, 40: 2.0211, 60: 2.0003, 120: 1.9799}, 1.96)


# --------------------------------------------------------------------------------------------------
# trusted grader
# --------------------------------------------------------------------------------------------------
class GraderError(ValueError):
    """The trusted grader failed or broke the output contract. ``kind``: ``invalid`` | ``timeout`` | ``infra``.

    The message never contains grader output; ``stderr_tail`` (for operators only) does."""

    def __init__(self, message: str, *, kind: str = "invalid", stderr_tail: str = ""):
        super().__init__(message)
        self.kind, self.stderr_tail = kind, stderr_tail


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name}")


def _finite(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraderError(f"{what} must be a number")
    try:
        f = float(value)
    except OverflowError:
        raise GraderError(f"{what} is out of range") from None
    if not math.isfinite(f):
        raise GraderError(f"{what} must be finite")
    return f


def _parse_grader_output(stdout: bytes, split: str) -> dict:
    try:
        text = stdout.decode("utf-8")
    except UnicodeDecodeError:
        raise GraderError("grader stdout is not UTF-8") from None
    # Split on "\n" only: str.splitlines() also splits on U+2028, \x1c, \x85 ... which json.dumps(ensure_ascii=False) may emit raw.
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        raise GraderError("grader printed nothing")
    try:
        obj = json.loads(lines[-1], parse_constant=_reject_constant)
    except ValueError as exc:
        raise GraderError(f"last grader line is not strict JSON: {str(exc)[:80]}") from None
    if not isinstance(obj, dict):
        raise GraderError("grader output must be a JSON object")
    if obj.get("split") != split:
        raise GraderError("grader did not echo the requested split")
    valid = obj.get("valid")
    if not isinstance(valid, bool):
        raise GraderError("valid must be a boolean")
    score_raw = obj.get("score")
    if valid or score_raw is not None:
        score: Optional[float] = _finite(score_raw, "score")
    else:
        score = None
    metrics_raw = obj.get("metrics", {})
    if not isinstance(metrics_raw, dict) or not all(isinstance(k, str) for k in metrics_raw):
        raise GraderError("metrics must be an object with string keys")
    metrics = {k: _finite(v, f"metric {k}") for k, v in metrics_raw.items()}
    n_eval = obj.get("n_eval", 0)
    if isinstance(n_eval, bool) or not isinstance(n_eval, int) or n_eval < 0:
        raise GraderError("n_eval must be a non-negative integer")
    notes = obj.get("notes", "")
    if not isinstance(notes, str):
        raise GraderError("notes must be a string")
    return {"valid": valid, "score": score, "split": split, "n_eval": n_eval, "metrics": metrics, "notes": notes}


def _grade_argv(task: TaskSpec, staged: Path, seed: int, split: str) -> list[str]:
    repl = {"{task_dir}": str(Path(task.task_dir).resolve()), "{workspace}": str(Path(staged).resolve()),
            "{seed}": str(int(seed)), "{split}": split}
    out = []
    for token in task.grade_cmd:
        for key, value in repl.items():
            token = token.replace(key, value)
        out.append(token)
    return out


def run_trusted_grader(task: TaskSpec, staged: Path, seed: int, split: str = "dev") -> dict:
    """Run the grader on ``staged`` and return ``{valid, score, split, n_eval, metrics, notes}``.

    The grader runs with ``cwd = task_dir``, a scrubbed environment (``PYTHONDONTWRITEBYTECODE=1``), rlimits, in its own
    process group that is killed afterwards. ``GraderError`` (a ``ValueError``) is raised unless the exit code is 0 and the
    last stdout line (stdout is capped at 64 KiB) satisfies the strict contract of docs/ARCHITECTURE.md 4.2.
    ``valid == False`` is returned, not raised: it is a model-side result.
    """
    if split not in ("dev", "sealed"):
        raise ValueError(f"unknown split {split!r}")
    res = sandbox.run_trusted(task, _grade_argv(task, staged, seed, split), cwd=task.task_dir,
                              env=sandbox.trusted_env(task, str(Path(staged).resolve())),
                              timeout_sec=task.grade_timeout_sec, stdout_cap=GRADER_STDOUT_CAP)
    tail = res.stderr.decode("utf-8", errors="replace")
    if res.spawn_error:
        raise GraderError(f"cannot start the grader: {res.spawn_error}", kind="infra")
    if res.timed_out:
        raise GraderError(f"grader timed out after {task.grade_timeout_sec:g}s", kind="timeout", stderr_tail=tail)
    if res.overflow:
        raise GraderError("grader stdout exceeds 64 KiB", stderr_tail=tail)
    if res.exit_code != 0:
        kind = "infra" if sandbox.classify_failure(task, res.exit_code, tail, False) == "infra" else "invalid"
        raise GraderError(f"grader exited with code {res.exit_code}", kind=kind, stderr_tail=tail)
    try:
        return _parse_grader_output(res.stdout, split)
    except GraderError as exc:
        exc.stderr_tail = tail
        raise


# --------------------------------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------------------------------
@dataclass
class EvalHooks:
    """Callbacks the engine uses to write ledger events. Exceptions they raise propagate (after the process is killed)."""

    on_run_start: Optional[Callable[[int, int, Optional[int]], None]] = None      # (run_idx, seed, pid)
    on_heartbeat: Optional[Callable[[int, int, Optional[float]], None]] = None    # (run_idx, output_bytes, cpu_sec)
    on_run_end: Optional[Callable[[int, int, RunResult], None]] = None            # (run_idx, seed, result)
    on_stalled: Optional[Callable[[int, float], None]] = None                     # (run_idx, idle_sec)


@dataclass
class EvalResult:
    grade: GradeResult
    runs: list[RunResult]
    failure_class: Optional[str]
    conditions_hash: str
    outputs_hashes: list[str] = field(default_factory=list)


def _clean(text: str, limit: int = 200) -> str:
    return _CONTROL.sub("", text)[:limit]


def _mask_paths(run_dir: Path, exp_id: str) -> list[Path]:
    """What the namespace hides from the run: the ledger files and every other experiment's directory."""
    masks = [run_dir / "ledger.jsonl", run_dir / "ledger.jsonl.lock"]
    try:
        siblings = sorted((run_dir / "exp").iterdir())
    except OSError:
        siblings = []
    masks.extend(p for p in siblings if p.name != exp_id and p.is_dir() and not p.is_symlink())
    return masks


def _trusted_changed(task: TaskSpec, baseline: Mapping[str, str]) -> None:
    now = trusted_manifest(task)
    if now != dict(baseline):
        names = sorted(set(baseline) ^ set(now) | {k for k in baseline if k in now and baseline[k] != now[k]})
        raise ProtectedPathViolation([f"trusted/{sandbox.display_path(n)}" for n in names] or ["trusted"], "trusted")


def _stray_bytecode(task: TaskSpec) -> list[str]:
    """Source-less ``*.pyc`` files in ``trusted/`` (invisible to the content hash, importable by the grader)."""
    return sandbox.stray_bytecode(sandbox.trusted_stat(task.trusted_dir))


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def evaluate(task: TaskSpec, ws: Workspace, seeds: Sequence[int], *, purpose: str, run_dir: Path, expected_tree_hash: str,
             hooks: Optional[EvalHooks] = None, run_idx_start: int = 0, attempt: int = 1,
             timeout_sec: Optional[float] = None, isolation: str = "auto", heartbeat_sec: float = 2.0,
             expected_grader_hash: Optional[str] = None) -> EvalResult:
    """Run and grade ``ws`` once per seed (docs/ARCHITECTURE.md 4.4); the split comes from ``Purpose.split(purpose)``.

    Per seed ``i`` (run index ``run_idx_start + i``): copy the workspace to ``exp/<id>/runs/<idx>/ws`` file by file,
    require its non-scratch tree hash to equal ``expected_tree_hash``, run the untrusted command, kill the group, require
    that no non-scratch file changed and no special file appeared (``ProtectedPathViolation(kind="runtime")``), stage the
    outputs, check ``trusted/`` (``kind="trusted"``) before and after the grader. An existing ``runs/<idx>`` directory is
    replaced. Every seed is attempted; a failed seed makes ``grade.valid`` False and ``failure_class`` follows
    infra > timeout > crash > invalid. ``expected_grader_hash`` (optional, e.g. ``run.started.grader_hash``) makes
    ``evaluate`` refuse to start (``kind="trusted"``) when ``trusted/`` no longer hashes to it, so a change made between two
    evaluations is not silently accepted as the new baseline.
    """
    split = Purpose.split(purpose)
    if split == "sealed" and not task.sealed:
        raise ValueError(f"task {task.name!r} has no sealed split")
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("evaluate needs at least one seed")
    hooks = hooks or EvalHooks()
    run_dir = Path(run_dir)
    exp_dir = run_dir / "exp" / ws.exp_id
    trusted_before = trusted_manifest(task)
    grader_hash = manifest_hash(trusted_before)
    if expected_grader_hash is not None and grader_hash != expected_grader_hash:
        # trusted/ changed since the caller last looked (e.g. recorded in run.started): never re-baseline on tampered files
        raise ProtectedPathViolation(["trusted"], "trusted")
    stray = _stray_bytecode(task)
    if stray:                    # a source-less module planted next to the grader: never start on top of it
        raise ProtectedPathViolation(stray, "trusted")
    cond = conditions_hash(task, seeds, split, grader_hash=grader_hash)
    masks = _mask_paths(run_dir, ws.exp_id)

    runs: list[RunResult] = []
    outputs_hashes: list[str] = []
    seed_results: list[dict] = []
    failures: list[tuple[str, str]] = []

    for i, seed in enumerate(seeds):
        idx = run_idx_start + i
        seed_dir = exp_dir / "runs" / str(idx)
        if os.path.lexists(seed_dir):
            shutil.rmtree(seed_dir)
        run_ws = seed_dir / "ws"

        pre = sandbox.copy_regular_hashes(ws.path, run_ws, task, kind="runtime")
        if sandbox.hashes_digest(pre) != expected_tree_hash:
            raise ProtectedPathViolation(["tree_hash_mismatch"], "runtime")

        def started(pid: Optional[int], _idx: int = idx, _seed: int = seed) -> None:
            if hooks.on_run_start:
                hooks.on_run_start(_idx, _seed, pid)

        def beat(output_bytes: int, cpu: Optional[float], _idx: int = idx) -> None:
            if hooks.on_heartbeat:
                hooks.on_heartbeat(_idx, output_bytes, cpu)

        def stalled(idle: float, _idx: int = idx) -> None:
            if hooks.on_stalled:
                hooks.on_stalled(_idx, idle)

        # One watch spans the untrusted run, the staging and the grader: a stat snapshot of trusted/ is taken right before the
        # process is spawned and compared after it has been reaped (inside run_untrusted) and again after grading. The content
        # hash alone cannot see a file that was changed and put back before the process exited. An attempt after the first
        # runs alone, so a difference that was ambiguous (another lane's doing, perhaps) becomes decisive on the retry.
        with sandbox.TrustedWatch(task.trusted_dir, solo=attempt > 1) as watch:
            run = sandbox.run_untrusted(task, run_ws, seed, purpose=purpose, attempt=attempt, on_start=started,
                                        on_heartbeat=beat, on_stalled=stalled, heartbeat_sec=heartbeat_sec,
                                        timeout_sec=timeout_sec, isolation=isolation, log_dir=exp_dir / "logs",
                                        log_prefix=f"run_{idx}", mask_paths=masks, watch=watch)
            runs.append(run)
            violation: Optional[ProtectedPathViolation] = None
            staged_hash, skipped = "", []
            try:
                post, specials = sandbox.snapshot_hashes(run_ws, task)
                bad = specials + [p for p in sandbox.changed_paths(pre, post) if f"special:{p}" not in specials]
                if bad:
                    violation = ProtectedPathViolation(bad, "runtime")
                elif run.classification == "ok":
                    staged_hash, skipped = sandbox.stage_outputs_report(run_ws, seed_dir / "staged" / "outputs", task)
            finally:
                if hooks.on_run_end:
                    hooks.on_run_end(idx, seed, run)
            if violation is not None:
                raise violation
            # Whatever the run did (also when it crashed or timed out), the trusted tree must be exactly what it was before it.
            _trusted_changed(task, trusted_before)
            if run.trusted_changed and not run.trusted_contested:
                raise ProtectedPathViolation(run.trusted_changed, "trusted")
            outputs_hashes.append(staged_hash)
            if run.classification != "ok":
                detail = f": {run.note}" if run.note else ""
                failures.append((run.classification, f"seed {seed}: run {run.classification}{detail}"))
                continue

            _trusted_changed(task, trusted_before)
            before_grader = _stray_bytecode(task)       # planted after the run was swept (a survivor, another lane)
            if before_grader:
                if not watch.contested:
                    raise ProtectedPathViolation(before_grader, "trusted")
                failures.append(("infra", f"seed {seed}: bytecode appeared in the trusted tree "
                                          f"({', '.join(before_grader[:3])}); it may come from a concurrent lane"))
                continue
            try:
                raw: Optional[dict] = run_trusted_grader(task, seed_dir / "staged", seed, split)
                gerr: Optional[GraderError] = None
            except GraderError as exc:
                raw, gerr = None, exc
            _trusted_changed(task, trusted_before)
            late = watch.check(ignore_bytecode=True)    # a grader's own bytecode is tolerated, as the content hash does
            late += [p for p in _stray_bytecode(task) if p not in late]       # but a source-less module is not bytecode
            if late:
                if not watch.contested:
                    raise ProtectedPathViolation(late, "trusted")
                failures.append(("infra", f"seed {seed}: trusted tree changed while grading "
                                          f"({', '.join(late[:3])}); it may come from a concurrent lane"))
                continue
        if gerr is not None:
            failures.append((gerr.kind, f"seed {seed}: grader {gerr.kind}: {_clean(str(gerr))}"))
        elif not raw["valid"]:
            note = _clean(raw["notes"]) if split == "dev" and raw["notes"] else ""
            failures.append(("invalid", f"seed {seed}: grader reported the output invalid" + (f" ({note})" if note else "")))
        else:
            if skipped:
                raw["notes"] = _clean((raw["notes"] + " " if raw["notes"] else "") + "skipped outputs: " + ",".join(skipped[:5]))
            seed_results.append(raw)

    grade: Optional[GradeResult] = None
    if not failures:
        per_seed = [r["score"] for r in seed_results]
        per_seed_metrics = []
        for r in seed_results:
            m = dict(r["metrics"])
            m.setdefault(task.primary_metric, r["score"])
            per_seed_metrics.append(m)
        common = [k for k in per_seed_metrics[0] if all(k in m for m in per_seed_metrics)]
        notes = next((r["notes"] for r in seed_results if r["notes"]), "")
        try:
            grade = GradeResult(valid=True, score=_mean(per_seed),
                                std=statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0,
                                per_seed=per_seed, metrics={k: _mean([m[k] for m in per_seed_metrics]) for k in common},
                                per_seed_metrics=per_seed_metrics, grader_hash=grader_hash,
                                notes=_clean(notes, 500) if split == "dev" else "", error=None, split=split,
                                n_eval=min(r["n_eval"] for r in seed_results), seeds=seeds)
        except (OverflowError, ValueError, ArithmeticError):      # finite scores whose mean overflows (fsum) are not a result
            failures.append(("invalid", "grader scores overflow when averaged"))
    failure_class = next((c for c in FAILURE_PRECEDENCE if any(f[0] == c for f in failures)), None)
    if grade is None:
        grade = GradeResult(valid=False, score=None, std=0.0, per_seed=[], metrics={}, per_seed_metrics=[],
                            grader_hash=grader_hash, notes="", error=_clean("; ".join(t for _, t in failures), 500),
                            split=split, n_eval=0, seeds=seeds)
    return EvalResult(grade=grade, runs=runs, failure_class=failure_class, conditions_hash=cond,
                      outputs_hashes=outputs_hashes)


# --------------------------------------------------------------------------------------------------
# statistics (contract section 8)
# --------------------------------------------------------------------------------------------------
@dataclass
class Decision(OutcomeDecision):
    """``OutcomeDecision`` plus a human-readable reason for an ``invalid`` verdict (``reason`` stays ``not_comparable``)."""

    detail: str = ""


def _critical(k_sigma: Any, n: int) -> float:
    df = n - 1
    if k_sigma is None:
        t = T95[df]
    elif isinstance(k_sigma, Mapping):
        try:
            t = k_sigma[df]
        except KeyError:
            raise ValueError(f"k_sigma table has no entry for {df} degrees of freedom") from None
    elif isinstance(k_sigma, (int, float)) and not isinstance(k_sigma, bool):
        t = k_sigma
    else:
        raise TypeError("k_sigma must be None, a number, or a table keyed by degrees of freedom")
    t = float(t)
    if not math.isfinite(t) or t <= 0:
        raise ValueError("k_sigma must be a positive finite number")
    return t


class _NonFinite(ArithmeticError):
    pass


def _paired(task: TaskSpec, incumbent: Sequence[float], candidate: Sequence[float], *, t: float, sigma_prior: float):
    sign = 1.0 if task.minimize else -1.0
    d = [sign * (a - b) for a, b in zip(incumbent, candidate)]
    n = len(d)
    # Checked here, before fsum/stdev: statistics.stdev raises AttributeError (not ValueError) on inf/nan on Python 3.11 and
    # 3.12 and ValueError on 3.13, so relying on the exception type would make the verdict depend on the interpreter.
    if not all(math.isfinite(x) for x in d):                      # e.g. 1e308 - (-1e308) overflows to inf
        raise _NonFinite("a paired difference overflowed")
    try:
        delta = math.fsum(d) / n
        s = statistics.stdev(d) if n >= 2 else 0.0
    except (OverflowError, ValueError, ZeroDivisionError):       # finite terms whose sum or spread still overflows
        raise _NonFinite("a paired difference overflowed") from None
    s_eff = max(s, sigma_prior, task.sigma_floor)
    margin = max(task.noise_margin, t * s_eff / math.sqrt(n))
    if not all(math.isfinite(v) for v in (delta, s, s_eff, margin)):
        raise _NonFinite("a paired difference overflowed")
    return delta, margin, s_eff


def _not_comparable(detail: str) -> Decision:
    return Decision(outcome="invalid", delta=None, margin=0.0, train_delta=None, overfit_suspect=False,
                    reason="not_comparable", detail=detail)


def _all_finite(values: Sequence[Any]) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


def decide_outcome(task: TaskSpec, candidate: GradeResult, incumbent: GradeResult, *, k_sigma: Any = None,
                   sigma_prior: float = 0.0, eps: float = 1e-12) -> Decision:
    """Compare ``candidate`` with ``incumbent`` exactly as docs/ARCHITECTURE.md section 8.

    Paired by seed index, ``d_i = sign * (incumbent_i - candidate_i)`` (positive is better for either objective);
    ``margin = max(noise_margin, t * max(s, sigma_prior, sigma_floor) / sqrt(n))`` with ``t = T95[n-1]`` or ``k_sigma``
    (a number, or a table such as ``T975`` for the two-sided sealed comparison). ``invalid`` with
    ``reason == "not_comparable"`` when a grade is invalid, anything is non-finite, or grader hash / split / seeds / n differ.
    """
    cand, inc = candidate, incumbent
    if not (cand.valid and inc.valid):
        return _not_comparable("a grade is invalid")
    if cand.grader_hash != inc.grader_hash:
        return _not_comparable("grader_hash differs")
    if cand.split != inc.split:
        return _not_comparable("split differs")
    if list(cand.seeds) != list(inc.seeds):
        return _not_comparable("seeds differ")
    n = len(cand.per_seed)
    if n < 1 or len(inc.per_seed) != n:
        return _not_comparable("per-seed lengths differ or are empty")
    if cand.n_eval and inc.n_eval and cand.n_eval != inc.n_eval:
        return _not_comparable("n_eval differs")
    if (not _all_finite([cand.score, inc.score, cand.std, inc.std, sigma_prior, *cand.per_seed, *inc.per_seed])
            or not math.isfinite(task.noise_margin)):
        return _not_comparable("a score, std or sigma_prior is missing or not finite")
    t = _critical(k_sigma, n)
    sigma_prior = max(0.0, float(sigma_prior))

    try:
        delta, margin, s_eff = _paired(task, inc.per_seed, cand.per_seed, t=t, sigma_prior=sigma_prior)
    except _NonFinite as exc:
        return _not_comparable(str(exc))
    outcome = "improved" if delta > margin + eps else "worse" if delta < -(margin + eps) else "no_gain"

    train_delta = train_margin = None
    overfit = False
    key = task.train_metric
    if key and len(cand.per_seed_metrics) == n and len(inc.per_seed_metrics) == n \
            and all(key in m for m in cand.per_seed_metrics) and all(key in m for m in inc.per_seed_metrics):
        tc, ti = [m[key] for m in cand.per_seed_metrics], [m[key] for m in inc.per_seed_metrics]
        if not _all_finite([*tc, *ti]):
            return _not_comparable("a train metric is not finite")
        try:
            train_delta, train_margin, _ = _paired(task, ti, tc, t=t, sigma_prior=0.0)
        except _NonFinite as exc:
            return _not_comparable(str(exc))
        overfit = train_delta > train_margin + eps and outcome != "improved"
    return Decision(outcome=outcome, delta=delta, margin=margin, train_delta=train_delta, overfit_suspect=overfit,
                    train_margin=train_margin, n=n, s_eff=s_eff, t=t)
