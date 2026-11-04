"""Fold ledger events into a ``LedgerView``. Pure function, tolerant of missing fields, ignores unknown kinds."""
from __future__ import annotations

from typing import Iterable

from .types import (Event, ExpStatus, ExperimentView, Finding, GradeRecord, GradeResult, Kind, LedgerView,
                    Outcome, ReviewRecord, RunRecord)

_ORDER = {ExpStatus.PROPOSED: 0, ExpStatus.WORKSPACE: 1, ExpStatus.IMPLEMENTED: 2, ExpStatus.RUNNING: 3,
          ExpStatus.RAN: 4, ExpStatus.GRADED: 5, ExpStatus.REVIEWED: 6, ExpStatus.RECORDED: 7}


def _advance(exp: ExperimentView, status: str) -> None:
    """Move status forward only. Terminal states (blocked, violation, recorded) are sticky."""
    if exp.status in (ExpStatus.BLOCKED, ExpStatus.VIOLATION, ExpStatus.RECORDED):
        return
    if _ORDER.get(status, -1) >= _ORDER.get(exp.status, -1):
        exp.status = status


def _ensure(view: LedgerView, exp_id: str, ev: Event) -> ExperimentView:
    exp = view.experiments.get(exp_id)
    if exp is None:
        exp = ExperimentView(exp_id=exp_id, proposed_seq=ev.seq, proposed_ts=ev.ts)
        view.experiments[exp_id] = exp
        view.order.append(exp_id)
    return exp


def _run_for(exp: ExperimentView, run_idx: int | None) -> RunRecord | None:
    for r in reversed(exp.runs):
        if run_idx is None or r.run_idx == run_idx:
            return r
    return None


def _apply(v: LedgerView, ev: Event) -> None:
    d = ev.data if isinstance(ev.data, dict) else {}
    k = ev.kind

    if k == Kind.RUN_STARTED:
        v.run = {**d, "run_id": ev.run_id, "ts": ev.ts}
    elif k == Kind.RUN_RESUMED:
        v.run.setdefault("resumes", []).append({**d, "seq": ev.seq, "ts": ev.ts})
        v.finished = None
    elif k == Kind.RUN_FINISHED:
        v.finished = {**d, "seq": ev.seq, "ts": ev.ts}
    elif k == Kind.SUP_TURN:
        v.supervisor_turns += 1
        v.turns_since_experiment += 1
        v.turns.append({**d, "seq": ev.seq, "ts": ev.ts})
    elif k == Kind.SUP_NUDGE:
        v.nudges.append({**d, "seq": ev.seq, "ts": ev.ts})
    elif k == Kind.SUP_FINISH_REQUESTED:
        v.finish_requests.append({**d, "seq": ev.seq, "ts": ev.ts})
    elif k == Kind.SUP_FINISH_REFUSED:
        v.finish_refusals.append({**d, "seq": ev.seq, "ts": ev.ts})
    elif k == Kind.CLAIM:
        cid = d.get("claim_id") or f"claim-{ev.seq}"
        v.claims[cid] = {**d, "claim_id": cid, "seq": ev.seq, "ts": ev.ts, "actor": ev.actor}
        if d.get("exp_id") and d["exp_id"] in v.experiments:
            v.experiments[d["exp_id"]].claim_ids.append(cid)

    elif k == Kind.EXP_PROPOSED:
        exp = _ensure(v, d["exp_id"], ev)
        exp.lane = d.get("lane", "")
        exp.hypothesis = d.get("hypothesis", "")
        exp.plan = d.get("plan", "")
        exp.parent_exp_id = d.get("parent_exp_id")
        exp.base = d.get("base", "template")
        exp.replicate_of = d.get("replicate_of")
        exp.hyp_fingerprint = d.get("hyp_fingerprint", "")
        exp.scripted_id = d.get("scripted_id")
        exp.planted_fault = d.get("planted_fault")
        exp.proposed_seq, exp.proposed_ts = ev.seq, ev.ts
        if exp.lane != "baseline":
            v.turns_since_experiment = 0
    elif k == Kind.EXP_BLOCKED:
        v.blocks.append({**d, "seq": ev.seq, "ts": ev.ts})
        if d.get("exp_id"):
            exp = _ensure(v, d["exp_id"], ev)
            exp.blocked = {**d, "seq": ev.seq, "ts": ev.ts}
            if exp.status != ExpStatus.RECORDED:
                exp.status = ExpStatus.BLOCKED
    elif k == Kind.EXP_WORKSPACE:
        exp = _ensure(v, d["exp_id"], ev)
        exp.workspace_path = d.get("path", "")
        _advance(exp, ExpStatus.WORKSPACE)
    elif k == Kind.EXP_IMPLEMENTED:
        exp = _ensure(v, d["exp_id"], ev)
        exp.worker = d.get("worker", "")
        exp.diff_hash = d.get("diff_hash", "")
        exp.idem_key = d.get("idem_key", "")
        exp.files_changed = list(d.get("files_changed", []))
        exp.diff_summary = d.get("diff_summary", "")
        exp.tree_hash = d.get("tree_hash", exp.tree_hash)
        exp.attempts += 1
        _advance(exp, ExpStatus.IMPLEMENTED)
    elif k == Kind.EXP_PROTECTED:
        exp = _ensure(v, d["exp_id"], ev)
        exp.protected_violations.append({**d, "seq": ev.seq, "ts": ev.ts})
        if exp.status != ExpStatus.RECORDED:
            exp.status = ExpStatus.VIOLATION
    elif k == Kind.EXP_RUN_STARTED:
        exp = _ensure(v, d["exp_id"], ev)
        exp.runs.append(RunRecord(run_idx=int(d.get("run_idx", len(exp.runs))), purpose=d.get("purpose", "primary"),
                                  seed=d.get("seed"), started_seq=ev.seq, started_ts=ev.ts, pid=d.get("pid"),
                                  isolation=d.get("isolation", "")))
        _advance(exp, ExpStatus.RUNNING)
    elif k == Kind.EXP_HEARTBEAT:
        exp = v.experiments.get(d.get("exp_id", ""))
        run = _run_for(exp, d.get("run_idx")) if exp else None
        if run:
            run.last_heartbeat_ts = ev.ts
            run.last_output_bytes = d.get("output_bytes")
    elif k == Kind.EXP_RUN_FINISHED:
        exp = _ensure(v, d["exp_id"], ev)
        run = _run_for(exp, d.get("run_idx"))
        if run:
            run.finished_seq, run.finished_ts = ev.seq, ev.ts
            run.exit_code = d.get("exit_code")
            run.duration_sec = d.get("duration_sec")
            run.classification = d.get("classification")
            run.stalled = bool(d.get("stalled", False))
        if all(r.finished_seq is not None for r in exp.runs):
            _advance(exp, ExpStatus.RAN)
    elif k == Kind.EXP_GRADED:
        exp = _ensure(v, d["exp_id"], ev)
        exp.grades.append(GradeRecord(purpose=d.get("purpose", "primary"), seq=ev.seq, ts=ev.ts,
                                      grade=GradeResult.from_dict(d.get("grade", {"valid": False, "score": None})),
                                      seeds=list(d.get("seeds", [])), conditions_hash=d.get("conditions_hash", ""),
                                      grader_hash=d.get("grader_hash", "")))
        _advance(exp, ExpStatus.GRADED)
    elif k == Kind.EXP_REVIEWED:
        exp = _ensure(v, d["exp_id"], ev)
        exp.reviews.append(ReviewRecord(round=int(d.get("review_round", len(exp.reviews))), seq=ev.seq,
                                        reviewer=d.get("reviewer", ""), verdict=d.get("verdict", ""),
                                        findings=list(d.get("findings", [])), independent=bool(d.get("independent", False))))
        _advance(exp, ExpStatus.REVIEWED)
    elif k == Kind.EXP_REPAIRED:
        exp = _ensure(v, d["exp_id"], ev)
        exp.repairs.append({**d, "seq": ev.seq, "ts": ev.ts})
        if d.get("diff_hash"):
            exp.diff_hash = d["diff_hash"]
    elif k == Kind.EXP_RECORDED:
        exp = _ensure(v, d["exp_id"], ev)
        if d.get("phase", "lane") == "replication":
            # the sealed-split result: final outcome changes, the dev numbers (delta, margin) stay untouched
            exp.outcome = d.get("outcome", exp.outcome)
            exp.sealed_outcome = d.get("outcome")
            exp.sealed_delta = d.get("sealed_delta")
            exp.sealed_margin = d.get("sealed_margin")
            exp.replicated = bool(d.get("replicated", False))
        else:
            exp.outcome = d.get("outcome")
            exp.dev_outcome = d.get("outcome")
            exp.delta = d.get("delta")
            exp.vs_exp_id = d.get("vs_exp_id")
            exp.noise_margin = d.get("noise_margin")
            exp.train_delta = d.get("train_delta")
            exp.train_margin = d.get("train_margin")
            exp.overfit_suspect = bool(d.get("overfit_suspect", False))
            exp.replicated = bool(d.get("replicated", exp.replicated))
            exp.recorded_seq, exp.recorded_ts = ev.seq, ev.ts
            exp.status = ExpStatus.RECORDED
            if exp.outcome == Outcome.BASELINE:
                v.baseline_id = exp.exp_id
                if v.incumbent_id is None:
                    v.incumbent_id = exp.exp_id
    elif k == Kind.INCUMBENT_PROMOTED:
        v.promotions.append({**d, "seq": ev.seq, "ts": ev.ts})
        promoted = d.get("exp_id")
        if isinstance(promoted, str):
            v.incumbent_id = promoted
        exp = v.experiments.get(promoted) if isinstance(promoted, str) else None
        if exp is not None:
            exp.replicated = bool(d.get("replicated", False))

    elif k == Kind.LESSON:
        v.lessons.append({**d, "seq": ev.seq, "ts": ev.ts, "actor": ev.actor})
    elif k == Kind.REFLECTION:
        v.reflections.append({**d, "seq": ev.seq, "ts": ev.ts})
    elif k == Kind.BUDGET_CHARGED:
        kind = d.get("kind")
        amount = float(d.get("amount", 0) or 0)
        key = {"llm": "usd", "experiment": "experiments", "replication": "replications",
               "wall": "wall_sec"}.get(kind, str(kind))
        v.budget_used[key] = v.budget_used.get(key, 0.0) + amount
        if kind == "llm":
            v.budget_used["llm_calls"] = v.budget_used.get("llm_calls", 0.0) + 1
            if d.get("call_id"):
                v.charged_call_ids.add(d["call_id"])
    elif k == Kind.AUDIT_WINDOW:
        v.audit_windows.append({**d, "seq": ev.seq, "ts": ev.ts})
    elif k == Kind.AUDIT_FINDING:
        raw = d.get("finding", {})
        f = Finding.from_dict(raw)
        if not f.created_seq:
            f.created_seq = ev.seq
        if f.finding_id not in v.findings:
            f.status = "open"
            v.findings[f.finding_id] = f
    elif k == Kind.AUDIT_ACK:
        v.acks.append({**d, "seq": ev.seq, "ts": ev.ts, "actor": ev.actor})
        # Only the auditor (after its own containment check) or a human can close a finding. A supervisor
        # asking for it is not an ack; Auditor.acknowledge validates the request and then writes this event.
        f = v.findings.get(d.get("finding_id", ""))
        if f is not None and d.get("action") in ("addressed", "dismissed") and ev.actor in ("auditor", "human"):
            f.status = d["action"]
    elif k == Kind.WATCHDOG_STALLED:
        v.stalls.append({**d, "seq": ev.seq, "ts": ev.ts})


def build_view(events: Iterable[Event]) -> LedgerView:
    """Fold events into a LedgerView. A malformed payload never aborts the fold: the event is skipped and listed in
    ``view.malformed`` so one bad line cannot take down the audit, the dashboard or a resume."""
    v = LedgerView()
    for ev in events:
        v.event_count += 1
        v.last_seq = ev.seq
        v.last_ts = ev.ts
        if not v.first_ts:
            v.first_ts = ev.ts
        try:
            _apply(v, ev)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            v.malformed.append({"seq": ev.seq, "kind": ev.kind, "error": type(exc).__name__})
    return v
