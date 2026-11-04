"""The Auditor: runs the deterministic checks and the LLM questions over the ledger and writes what it finds back to it.

Guarantees
----------
* Findings never act. The Auditor only appends ``audit.finding``, ``audit.ack`` and ``audit.window`` events (actor
  ``auditor``); it never spends money, kills a job or edits policy.
* Every public method holds the Auditor lock, so windows, sweeps and acknowledgements are serialised. Each pass works on
  one consistent snapshot of the ledger (``Ledger.events()``) and rebuilds the view from it, so concurrent appends by
  other threads cannot tear a pass.
* Idempotent: an ``audit.finding`` is written only for a finding id the ledger does not already hold.
* Silence is not a clean audit. Without a judge (or with a ``NullJudge``) the LLM questions are skipped and
  ``audit.window.skipped_reason`` says so; a judge that fails with ``ModelError`` yields one ``insufficient_evidence``
  finding of severity ``info`` per invariant per window. Any other exception (``BudgetExceeded`` included) propagates.
* Only the Auditor closes findings. ``acknowledge`` validates the request and then writes ``audit.ack`` itself
  (docs/ARCHITECTURE.md section 9). It never raises on bad input; it returns ``{"error": ...}``.
* Containment auto-acks happen one window after a finding was written, never in the window that created it.
* An answered LLM finding stays answered. LLM findings are keyed by their evidence (the experiment id when all the cited events
  belong to one experiment, else a short hash of the cited seqs), so the same evidence always yields the same finding id. A
  ``violated`` verdict whose evidence an earlier finding of the same invariant already covers is not written again, and events that an
  acknowledged finding covers are not put to the judge at all. New, different evidence still raises a new finding.
* The judge questions of one pass are independent, so they are asked concurrently (at most ``MAX_JUDGE_WORKERS`` at a time): a pass
  costs about the slowest question, not the sum. Findings are appended in the fixed question order whatever the completion order.
  If a judge raises anything but ``ModelError`` the pool is drained first, then the deterministic findings (and the LLM findings
  already obtained) are written, no ``audit.window`` is written, and the first error in question order propagates.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

from ..types import Clock, Finding, Judge, JudgePacket, JudgeResult, Kind, ModelError, Verdict, short_hash
from ..view import build_view
from .invariants import AuditConfig, AuditContext, BY_ID, CHECKS, CONTAINMENT_DETAIL, REGISTRY, llm_evidence_covered
from .judge import NullJudge, build_packet, normalise_result, sanitize

WINDOW_QUESTIONS = ("I13", "I07", "I08")            # semantic questions asked in every window
FINAL_QUESTIONS = WINDOW_QUESTIONS + ("I14",)       # the final sweep adds the overclaim question
MIN_DISMISS_CHARS = 20
MAX_JUDGE_WORKERS = 4                               # concurrent judge calls within one pass
_NOT_CONFIGURED = "judge not configured"


def _last_window_to(view) -> int:
    for w in reversed(view.audit_windows):
        try:
            return int(w.get("to_seq", 0) or 0)
        except (TypeError, ValueError):
            continue
    return 0


@dataclass
class _Judged:
    """The outcome of one judge question: a validated result, or the exception the judge raised."""

    inv_id: str
    packet: JudgePacket
    result: Optional[JudgeResult] = None
    error: Optional[BaseException] = None


class Auditor:
    def __init__(self, ledger, task, *, judge: Optional[Judge] = None, config: Optional[AuditConfig] = None,
                 clock: Clock = time.time, final_judge: Optional[Judge] = None):
        self.ledger = ledger
        self.task = task
        self.judge = judge
        self.final_judge = final_judge
        self.config = config or AuditConfig()
        self._clock = clock
        self._lock = threading.RLock()

    # ---- public API ----------------------------------------------------------------------------
    def audit_window(self) -> list[Finding]:
        """Audit the events since the last ``audit.window``. Returns the findings written by this call."""
        with self._lock:
            return self._pass(final=False)

    def final_sweep(self) -> list[Finding]:
        """Audit the whole run, add the overclaim question (I14) and use ``final_judge`` when one was given."""
        with self._lock:
            return self._pass(final=True)

    def acknowledge(self, finding_id: Any, action: Any, reason: Any, *, by: Any = "supervisor") -> dict:
        """Close a finding, or say why it cannot be closed. Returns ``{"ok": True, ...}`` or ``{"error": ...}``; bad input never raises."""
        with self._lock:
            try:
                return self._acknowledge(finding_id, action, reason, by)
            except (TypeError, ValueError, AttributeError, KeyError) as exc:
                return {"error": "invalid_request", "detail": type(exc).__name__}

    def coverage(self) -> dict:
        """invariant id -> ``checked`` | ``llm_not_run`` | ``never``, from the audit windows recorded in the ledger."""
        with self._lock:
            windows = build_view(self.ledger.events()).audit_windows
        judged = {i for w in windows for i in (w.get("judged") or [])}
        # ``invariants`` lists the deterministic ids that ran plus the judged ids, so a check that crashed (``check_errors``) is not
        # "checked" just because its LLM half answered in the same window.
        det_ran = {i for w in windows for i in (w.get("invariants") or []) if i not in (w.get("check_errors") or {})}
        out = {}
        for inv in REGISTRY:
            if inv.kind == "deterministic":
                out[inv.id] = "checked" if inv.id in det_ran else "never"
            elif inv.kind == "llm":
                out[inv.id] = "checked" if inv.id in judged else ("llm_not_run" if windows else "never")
            else:
                out[inv.id] = "never" if inv.id not in det_ran else ("checked" if inv.id in judged else "llm_not_run")
        return out

    # ---- one pass ------------------------------------------------------------------------------
    def _pass(self, final: bool) -> list[Finding]:
        events = self.ledger.events()
        if not events:
            return []
        view = build_view(events)
        to_seq = events[-1].seq
        from_seq = 1 if final else max(1, min(_last_window_to(view) + 1, to_seq))
        ctx = AuditContext(view, events, self.task, float(self._clock()), from_seq, to_seq, self.config)

        found, checked, errors = self._run_checks(ctx)
        llm_found, judged, verdicts, no_evidence, skipped, already, fatal = self._run_judge(ctx, final)

        written: list[Finding] = []
        seen = set(view.findings)
        for f in found + llm_found:
            if f.finding_id in seen:
                continue
            seen.add(f.finding_id)
            f.created_seq = self.ledger.append(Kind.AUDIT_FINDING, "auditor", {"finding": f.to_dict()}).seq
            written.append(f)
        if fatal is not None:
            # a judge failed with something other than ModelError: what was found is on record, the pass is not complete (no window)
            raise fatal

        self._auto_ack(ctx)
        payload: dict = {"from_seq": from_seq, "to_seq": to_seq, "invariants": sorted(set(checked) | set(judged)),
                         "judged": judged, "skipped_reason": skipped, "final": final, "new_findings": len(written)}
        if verdicts:
            payload["verdicts"] = verdicts
        if no_evidence:
            payload["no_evidence"] = no_evidence
        if already:
            payload["already_reported"] = already
        if errors:
            payload["check_errors"] = errors
        self.ledger.append(Kind.AUDIT_WINDOW, "auditor", payload)
        return written

    def _run_checks(self, ctx: AuditContext) -> tuple[list[Finding], list[str], dict]:
        found: list[Finding] = []
        checked: list[str] = []
        errors: dict[str, str] = {}
        for inv in REGISTRY:
            check = CHECKS.get(inv.id)
            if check is None:
                continue
            try:
                found.extend(check(ctx))
            except Exception as exc:  # a buggy check must not take the run down; it is reported as not checked
                errors[inv.id] = type(exc).__name__
                continue
            checked.append(inv.id)
        return found, checked, errors

    def _run_judge(self, ctx: AuditContext, final: bool):
        """Ask the judge questions of this pass. Returns ``(findings, judged, verdicts, no_evidence, skipped_reason, already_reported,
        fatal)``; ``fatal`` is the first exception (in question order) that was not a ``ModelError``, raised by the caller once the
        findings are written."""
        judge = self.final_judge if (final and self.final_judge is not None) else self.judge
        if judge is None or isinstance(judge, NullJudge):
            return [], [], {}, [], _NOT_CONFIGURED, [], None
        found: list[Finding] = []
        judged: list[str] = []
        verdicts: dict[str, str] = {}
        no_evidence: list[str] = []
        already: list[str] = []
        attempted = 0
        # 1. packets, in question order (cheap, no model call). Events an acknowledged finding of the same invariant covers are not offered.
        slots: list[tuple[str, Optional[Finding], Optional[JudgePacket]]] = []   # (invariant id, outage finding, packet to ask)
        for inv_id in (FINAL_QUESTIONS if final else WINDOW_QUESTIONS):
            inv = BY_ID[inv_id]
            try:
                packet = build_packet(inv, ctx, exclude=llm_evidence_covered(ctx, inv_id))
            except Exception as exc:  # a packet that cannot be built is a judge outage for this question, never a crash of the run
                attempted += 1
                slots.append((inv_id, self._finding(ctx, inv_id, Verdict.INSUFFICIENT, "info", f"judge_unavailable: packet {type(exc).__name__}"), None))
                continue
            if not packet.evidence:
                no_evidence.append(inv_id)
                continue
            attempted += 1
            slots.append((inv_id, None, packet))
        # 2. the model calls, concurrently
        answers = self._ask(judge, [(i, p) for i, _, p in slots if p is not None])
        # 3. results, in question order
        fatal: Optional[BaseException] = None
        for inv_id, outage, packet in slots:
            inv = BY_ID[inv_id]
            if packet is None:
                if outage is not None:
                    found.append(outage)
                continue
            answer = answers[inv_id]
            if answer.error is not None:
                if isinstance(answer.error, ModelError):
                    found.append(self._finding(ctx, inv_id, Verdict.INSUFFICIENT, "info", f"judge_unavailable: {type(answer.error).__name__}"))
                elif fatal is None:
                    fatal = answer.error
                continue
            result = answer.result
            judged.append(inv_id)
            verdicts[inv_id] = result.verdict
            if result.verdict == Verdict.VIOLATED:
                finding = self._llm_violation(ctx, inv_id, result, inv.severity)
                if finding is None:
                    already.append(inv_id)
                else:
                    found.append(finding)
            elif result.verdict == Verdict.INSUFFICIENT:
                found.append(self._llm_finding(ctx, inv_id, result, "info"))
        skipped = None
        if attempted and not judged:
            skipped = "judge_unavailable"
        elif not attempted:
            skipped = "no judgeable evidence in the window"
        return found, judged, verdicts, no_evidence, skipped, already, fatal

    @staticmethod
    def _ask(judge: Judge, jobs: list[tuple[str, JudgePacket]]) -> dict[str, _Judged]:
        """Put every ``(invariant id, packet)`` to ``judge`` with a pool of at most ``MAX_JUDGE_WORKERS`` threads and wait for all of
        them. Each answer is validated (``normalise_result``) or carries the exception the judge raised; nothing is swallowed."""
        def ask(inv_id: str, packet: JudgePacket) -> JudgeResult:
            return normalise_result(judge.judge(packet), packet)

        out: dict[str, _Judged] = {}
        if len(jobs) <= 1:
            for inv_id, packet in jobs:
                try:
                    out[inv_id] = _Judged(inv_id, packet, result=ask(inv_id, packet))
                except Exception as exc:
                    out[inv_id] = _Judged(inv_id, packet, error=exc)
            return out
        with ThreadPoolExecutor(max_workers=min(MAX_JUDGE_WORKERS, len(jobs)), thread_name_prefix="nightshift-judge") as pool:
            futures = [(inv_id, packet, pool.submit(ask, inv_id, packet)) for inv_id, packet in jobs]
        # leaving the ``with`` block waited for every call: the pool is drained before any error is looked at
        for inv_id, packet, future in futures:
            error = future.exception()
            out[inv_id] = _Judged(inv_id, packet, error=error) if error is not None else _Judged(inv_id, packet, result=future.result())
        return out

    @staticmethod
    def _finding(ctx: AuditContext, inv_id: str, verdict: str, severity: str, rationale: str, *, evidence: Optional[list[int]] = None,
                 confidence: Optional[float] = None, model: Optional[str] = None, exp_id: Optional[str] = None,
                 subject: Optional[str] = None) -> Finding:
        subject = subject or f"w{ctx.window_to}"  # default: one LLM verdict per invariant per window (outages, inconclusive answers)
        return Finding(finding_id=f"{inv_id}@{subject}", invariant_id=inv_id, verdict=verdict, severity=severity, source="llm",
                       rationale=rationale, evidence_seqs=list(evidence or []), confidence=confidence, subject=subject,
                       window_from=ctx.window_from, window_to=ctx.window_to, judge_model=model or None, exp_id=exp_id)

    @staticmethod
    def _attributed_exp(ctx: AuditContext, evidence: list[int]) -> Optional[str]:
        """The experiment all the cited events belong to, or None when they name none, several, or an unknown one."""
        exps = set()
        for seq in evidence:
            ev = ctx.event(seq)
            data = ev.data if ev is not None and isinstance(ev.data, dict) else {}
            if isinstance(data.get("exp_id"), str) and data["exp_id"]:
                exps.add(data["exp_id"])
        only = next(iter(exps)) if len(exps) == 1 else None
        return only if only in ctx.view.experiments else None

    def _llm_finding(self, ctx: AuditContext, inv_id: str, result: JudgeResult, severity: str, *, subject: Optional[str] = None) -> Finding:
        exp_id = self._attributed_exp(ctx, result.evidence_seqs)
        return self._finding(ctx, inv_id, result.verdict, severity, result.rationale, evidence=result.evidence_seqs,
                             confidence=result.confidence, model=result.model, exp_id=exp_id, subject=subject)

    def _llm_violation(self, ctx: AuditContext, inv_id: str, result: JudgeResult, severity: str) -> Optional[Finding]:
        """The finding for a ``violated`` verdict, or None when it only repeats what an earlier finding of the invariant reported.

        The subject is the evidence, not the window: the experiment id when the cited events belong to one experiment, else a short
        hash of the sorted seqs, so asking again about the same evidence gives the same finding id (and ``_pass`` writes it once).
        A verdict whose evidence an earlier finding (open, addressed or dismissed) already covers yields None. Evidence that is new
        but about an experiment that already has a finding gets its own id (``<exp>.n<hash>``), so it is never swallowed.
        """
        evidence = sorted({int(s) for s in result.evidence_seqs})
        if evidence and set(evidence) <= llm_evidence_covered(ctx, inv_id, statuses=None):
            return None
        exp_id = self._attributed_exp(ctx, evidence)
        digest = short_hash(",".join(str(s) for s in evidence), 8)
        subject = exp_id or f"ev-{digest}"
        if f"{inv_id}@{subject}" in ctx.view.findings:
            subject = f"{subject}.n{digest[:6]}"
        return self._llm_finding(ctx, inv_id, result, severity, subject=subject)

    def _auto_ack(self, ctx: AuditContext) -> None:
        """Close deterministic findings from earlier windows whose containment check now passes.

        A finding written by a pass that raised (so it has no ``audit.window`` yet) is not from an earlier window: it waits for the
        pass after the next complete one, like every other finding, so the supervisor is told about it first."""
        last_window_seq = max([int(w.get("seq", 0) or 0) for w in ctx.view.audit_windows] or [0])
        for f in list(ctx.view.findings.values()):
            if f.status != "open" or f.verdict != Verdict.VIOLATED or f.source != "deterministic":
                continue
            if f.created_seq > last_window_seq:
                continue
            decide = CONTAINMENT_DETAIL.get(f.invariant_id)
            if decide is None:
                continue
            try:
                ok, text = decide(ctx, f)
            except Exception:  # an unprovable containment leaves the finding open
                continue
            if ok:
                self.ledger.append(Kind.AUDIT_ACK, "auditor", {
                    "finding_id": f.finding_id, "action": "addressed", "reason": "containment check passed at audit window",
                    "by": "auditor", "containment": text, "followup_seqs": []})

    # ---- acknowledgement -------------------------------------------------------------------------
    def _acknowledge(self, finding_id: Any, action: Any, reason: Any, by: Any) -> dict:
        if not isinstance(finding_id, str) or not finding_id.strip():
            return {"error": "unknown_finding", "finding_id": sanitize(str(finding_id), 80)}
        if action not in ("addressed", "dismissed"):
            return {"error": "invalid_action", "allowed": ["addressed", "dismissed"]}
        text = sanitize(reason, 1000) if isinstance(reason, str) else ""
        requester = sanitize(by, 40) if isinstance(by, str) and by.strip() else "supervisor"
        events = self.ledger.events()
        view = build_view(events)
        finding = view.findings.get(finding_id.strip())
        if finding is None:
            return {"error": "unknown_finding", "finding_id": sanitize(finding_id, 80)}
        if finding.status != "open":
            return {"ok": True, "finding_id": finding.finding_id, "action": finding.status, "already_closed": True}
        if action == "dismissed":
            if finding.source == "deterministic" and finding.severity == "block":
                return {"error": "deterministic_block_not_dismissible"}
            if len(text.strip()) < MIN_DISMISS_CHARS:
                return {"error": "reason_too_short", "min_chars": MIN_DISMISS_CHARS}
            containment = f"dismissed by {requester}; no containment check applies to {finding.severity} {finding.source} findings"
        else:
            if finding.source != "deterministic":
                return {"error": "not_contained",
                        "need": f"a {finding.source} finding cannot be verified in code; dismiss it with a reason of at least "
                                f"{MIN_DISMISS_CHARS} characters instead"}
            decide = CONTAINMENT_DETAIL.get(finding.invariant_id)
            if decide is None:
                return {"error": "not_contained", "need": f"no containment check exists for {finding.invariant_id}"}
            ctx = AuditContext(view, events, self.task, float(self._clock()), _last_window_to(view) + 1, events[-1].seq, self.config)
            try:
                ok, containment = decide(ctx, finding)
            except Exception as exc:  # cannot prove containment, so the finding stays open
                return {"error": "not_contained", "need": f"the containment check failed ({type(exc).__name__})"}
            if not ok:
                return {"error": "not_contained", "need": containment}
        ev = self.ledger.append(Kind.AUDIT_ACK, "auditor", {
            "finding_id": finding.finding_id, "action": action, "reason": text, "by": requester, "containment": containment,
            "followup_seqs": []})
        return {"ok": True, "finding_id": finding.finding_id, "action": action, "containment": containment, "ack_seq": ev.seq}
