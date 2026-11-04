"""Build realistic ledgers for tests without running any experiment.

    clock = FakeClock()
    led = Ledger.create(tmpdir, clock=clock, fsync=False)
    rb = RunBuilder(led)
    rb.start(task="law-discovery")
    rb.baseline(score=1.0, train=0.9)
    e1 = rb.experiment("A", "try quadratic", score=0.8, train=0.7)      # recorded, improved
    rb.claim(e1, "improved", "held-out RMSE dropped")
    rb.promote(e1)
"""
from __future__ import annotations

from typing import Optional

from nightshift.ledger import Ledger
from nightshift.types import (BudgetLimits, GradeResult, Kind, Outcome, canonical_json, sha256_hex)


class FakeClock:
    def __init__(self, start: float = 1_000_000.0, step: float = 1.0):
        self.t, self.step = start, step

    def __call__(self) -> float:
        self.t += self.step
        return self.t

    def advance(self, secs: float) -> None:
        self.t += secs


def grade_dict(score: float, train: Optional[float] = None, std: float = 0.0, valid: bool = True,
               grader_hash: str = "g" * 12, split: str = "dev", seeds: Optional[list[int]] = None) -> dict:
    metrics = {"heldout": score}
    if train is not None:
        metrics["train"] = train
    seeds = list(seeds or [0, 1, 2])
    return GradeResult(valid=valid, score=score, std=std, per_seed=[score] * len(seeds), metrics=metrics,
                       grader_hash=grader_hash, split=split, n_eval=100, seeds=seeds).to_dict()


class RunBuilder:
    """Writes the same event sequences the orchestrator writes. Keep in sync with docs/ARCHITECTURE.md section 4."""

    def __init__(self, ledger: Ledger, seeds: Optional[list[int]] = None, replication_seeds: Optional[list[int]] = None):
        self.led = ledger
        self.seeds = seeds or [0, 1, 2]
        self.rep_seeds = replication_seeds or [100, 101, 102]
        self.cond = sha256_hex("task|grader|dev|seeds")[:12]
        self.cond_sealed = sha256_hex("task|grader|sealed|seeds")[:12]
        self._claim_n = 0
        self._finding_n = 0

    # ---- run -------------------------------------------------------------------------------------
    def start(self, task: str = "demo", backend: str = "scripted", limits: Optional[BudgetLimits] = None) -> None:
        self.led.append(Kind.RUN_STARTED, "engine", {
            "task": task, "backend": backend, "models": {"supervisor": backend}, "limits": (limits or BudgetLimits()).to_dict(),
            "task_hash": "t" * 12, "grader_hash": "g" * 12, "objective": "minimize", "primary_metric": "heldout",
            "config": {}})

    def finish(self, reason: str = "supervisor_finish", summary: str = "done") -> None:
        self.led.append(Kind.RUN_FINISHED, "engine", {"reason": reason, "summary": summary, "final_claims": []})

    def turn(self, n: int = 1) -> None:
        for _ in range(n):
            self.led.append(Kind.SUP_TURN, "supervisor", {"turn": self.led.view().supervisor_turns + 1, "model": "scripted",
                                                          "action_summary": "turn"})

    # ---- experiments -----------------------------------------------------------------------------
    def baseline(self, score: float = 1.0, train: Optional[float] = None) -> str:
        eid = self.led.next_exp_id()
        self.led.append(Kind.EXP_PROPOSED, "engine", {"exp_id": eid, "lane": "baseline", "hypothesis": "unmodified template",
                                                     "plan": "", "parent_exp_id": None, "base": "template",
                                                     "hyp_fingerprint": "baseline", "replicate_of": None})
        self.led.append(Kind.EXP_WORKSPACE, "engine", {"exp_id": eid, "path": f"/tmp/{eid}", "base_hash": "b" * 12})
        self._run(eid, 0, "baseline", self.seeds[0])
        self.led.append(Kind.EXP_GRADED, "engine", {"exp_id": eid, "purpose": "baseline", "grade": grade_dict(score, train),
                                                    "grader_hash": "g" * 12, "seeds": self.seeds, "conditions_hash": self.cond})
        self.led.append(Kind.EXP_RECORDED, "engine", {"exp_id": eid, "phase": "lane", "outcome": Outcome.BASELINE, "delta": None,
                                                      "vs_exp_id": None, "noise_margin": 0.0, "train_delta": None, "replicated": False})
        return eid

    def _run(self, eid: str, idx: int, purpose: str, seed: int, classification: str = "ok", exit_code: int = 0,
             finish: bool = True) -> None:
        self.led.append(Kind.EXP_RUN_STARTED, "engine", {"exp_id": eid, "run_idx": idx, "purpose": purpose, "seed": seed,
                                                         "cmd": ["python", "run.py"], "pid": 1234, "attempt": 1})
        if finish:
            self.led.append(Kind.EXP_RUN_FINISHED, "engine", {"exp_id": eid, "run_idx": idx, "purpose": purpose, "seed": seed,
                                                              "exit_code": exit_code, "duration_sec": 1.0,
                                                              "classification": classification, "stdout_tail": "", "stderr_tail": ""})

    def experiment(self, lane: str = "A", hypothesis: str = "h", *, score: Optional[float] = 0.9, train: Optional[float] = None,
                   std: float = 0.0, outcome: Optional[str] = None, delta: Optional[float] = None, vs: Optional[str] = None,
                   review: Optional[str] = "accept", independent: bool = True, classification: str = "ok",
                   implemented: bool = True, ran: bool = True, graded: bool = True, recorded: bool = True,
                   diff_hash: Optional[str] = None, idem_key: Optional[str] = None, files: Optional[list[str]] = None,
                   conditions_hash: Optional[str] = None, grader_hash: str = "g" * 12, heartbeat: bool = False,
                   protected: Optional[list[str]] = None, noise_margin: float = 0.0, train_delta: Optional[float] = None,
                   replicated: bool = False, parent: Optional[str] = None, fingerprint: Optional[str] = None,
                   scripted_id: Optional[str] = None, planted: Optional[str] = None, overfit_suspect: bool = False,
                   train_margin: Optional[float] = None, reviewer_identity: str = "reviewer-y",
                   implementer_identity: str = "worker-x") -> str:
        eid = self.led.next_exp_id()
        self.led.append(Kind.EXP_PROPOSED, f"designer:{lane}", {
            "exp_id": eid, "lane": lane, "hypothesis": hypothesis, "plan": "plan", "parent_exp_id": parent, "base": "incumbent",
            "hyp_fingerprint": fingerprint or sha256_hex(hypothesis)[:12], "replicate_of": None,
            "scripted_id": scripted_id, "planted_fault": planted})
        self.led.append(Kind.EXP_WORKSPACE, "engine", {"exp_id": eid, "path": f"/tmp/{eid}", "base_hash": "b" * 12})
        if implemented:
            dh = diff_hash or sha256_hex(f"{eid}-{hypothesis}")[:12]
            self.led.append(Kind.EXP_IMPLEMENTED, f"implementer:{lane}", {
                "exp_id": eid, "worker": implementer_identity, "diff_hash": dh, "files_changed": files or ["model.py"],
                "diff_summary": "edit", "idem_key": idem_key or sha256_hex(dh)[:16], "attempt": 1, "steps": 3,
                "tree_hash": sha256_hex("tree" + dh)[:12]})
        if protected:
            self.led.append(Kind.EXP_PROTECTED, "engine", {"exp_id": eid, "paths": protected, "kind": "workspace"})
            return eid
        if ran:
            self._run(eid, 0, "primary", self.seeds[0], classification=classification,
                      exit_code=0 if classification == "ok" else 1, finish=True)
            if heartbeat:
                self.led.append(Kind.EXP_HEARTBEAT, "engine", {"exp_id": eid, "run_idx": 0, "output_bytes": 10})
        if graded and ran and classification == "ok" and score is not None:
            self.led.append(Kind.EXP_GRADED, "engine", {
                "exp_id": eid, "purpose": "primary", "grade": grade_dict(score, train, std, grader_hash=grader_hash),
                "grader_hash": grader_hash, "seeds": self.seeds, "conditions_hash": conditions_hash or self.cond})
        if review:
            self.led.append(Kind.EXP_REVIEWED, f"reviewer:{lane}", {
                "exp_id": eid, "review_round": 0, "reviewer": reviewer_identity if independent else implementer_identity,
                "verdict": review, "findings": [], "independent": independent})
        if recorded:
            if outcome is None:
                outcome = Outcome.IMPROVED if (delta or 0) > noise_margin else Outcome.NO_GAIN
            self.led.append(Kind.EXP_RECORDED, "engine", {
                "exp_id": eid, "phase": "lane", "outcome": outcome, "delta": delta, "vs_exp_id": vs, "noise_margin": noise_margin,
                "train_delta": train_delta, "train_margin": train_margin, "overfit_suspect": overfit_suspect,
                "replicated": replicated})
        return eid

    def replication_baseline(self, base_id: str, score: float, train: Optional[float] = None) -> None:
        """Sealed grade of the baseline experiment, written under the baseline's own exp id."""
        idx = len(self.led.view().exp(base_id).runs)
        self._run(base_id, idx, "replication_baseline", self.rep_seeds[0])
        self.led.append(Kind.EXP_GRADED, "engine", {
            "exp_id": base_id, "purpose": "replication_baseline", "grade": grade_dict(score, train, split="sealed", seeds=self.rep_seeds),
            "grader_hash": "g" * 12, "seeds": self.rep_seeds, "conditions_hash": self.cond_sealed, "split": "sealed"})

    def replicate(self, eid: str, score: float, train: Optional[float] = None, *, success: bool = True,
                  sealed_delta: float = 0.1, sealed_margin: float = 0.01, grader_hash: str = "g" * 12) -> None:
        """Sealed re-grade of a candidate plus the replication-phase record. Does not promote; call ``promote``."""
        idx = len(self.led.view().exp(eid).runs)
        self._run(eid, idx, "replication", self.rep_seeds[0])
        self.led.append(Kind.EXP_GRADED, "engine", {
            "exp_id": eid, "purpose": "replication", "grade": grade_dict(score, train, split="sealed", seeds=self.rep_seeds,
                                                                         grader_hash=grader_hash),
            "grader_hash": grader_hash, "seeds": self.rep_seeds, "conditions_hash": self.cond_sealed, "split": "sealed"})
        self.led.append(Kind.EXP_RECORDED, "engine", {
            "exp_id": eid, "phase": "replication", "outcome": Outcome.IMPROVED if success else Outcome.NO_GAIN,
            "sealed_delta": sealed_delta if success else -abs(sealed_delta) / 10, "sealed_margin": sealed_margin,
            "replicated": success, "note": "" if success else "failed replication"})

    def promote(self, eid: str, replicated: bool = True, score: float = 0.0, prev: Optional[str] = None) -> None:
        self.led.append(Kind.INCUMBENT_PROMOTED, "engine", {
            "exp_id": eid, "score": score, "prev_exp_id": prev, "replicated": replicated,
            "replication_exp_ids": [eid] if replicated else [], "tree_hash": sha256_hex("tree" + eid)[:12],
            "sealed_score": score, "sealed_delta": 0.1, "sealed_margin": 0.01})

    # ---- claims, lessons, audit ------------------------------------------------------------------
    def claim(self, exp_id: Optional[str], claim_type: str, text: str, actor: str = "implementer:A",
              cites: Optional[list[int]] = None, alias: Optional[str] = None) -> str:
        self._claim_n += 1
        cid = f"{exp_id or 'run'}.c{self._claim_n}"
        self.led.append(Kind.CLAIM, actor, {"claim_id": cid, "exp_id": exp_id, "claim_type": claim_type, "text": text,
                                            "cites": cites or [], "alias": alias})
        return cid

    def lesson(self, text: str, source_exps: list[str], kind: str = "observation", classification: str = "model") -> str:
        lid = f"L{len(self.led.view().lessons) + 1:03d}"
        self.led.append(Kind.LESSON, "reflector", {"lesson_id": lid, "text": text, "kind": kind, "source_exps": source_exps,
                                                   "classification": classification})
        return lid

    def charge_llm(self, usd: float, call_id: str, model: str = "muse-spark-1.3") -> None:
        self.led.append(Kind.BUDGET_CHARGED, "engine", {"kind": "llm", "amount": usd, "unit": "usd", "call_id": call_id,
                                                        "model": model, "usage": {}})

    def finding(self, invariant_id: str, subject: str, *, severity: str = "block", verdict: str = "violated",
                source: str = "deterministic", evidence: Optional[list[int]] = None, rationale: str = "r",
                exp_id: Optional[str] = None) -> str:
        fid = f"{invariant_id}@{subject}"
        self.led.append(Kind.AUDIT_FINDING, "auditor", {"finding": {
            "finding_id": fid, "invariant_id": invariant_id, "verdict": verdict, "severity": severity, "source": source,
            "rationale": rationale, "evidence_seqs": evidence or [], "subject": subject, "window_from": 0, "window_to": 0,
            "exp_id": exp_id}})
        return fid

    def ack(self, finding_id: str, action: str = "addressed", reason: str = "fixed", actor: str = "auditor") -> None:
        """Acks only take effect when written by the auditor (or a human), as in production."""
        self.led.append(Kind.AUDIT_ACK, actor, {"finding_id": finding_id, "action": action, "reason": reason,
                                                "by": "supervisor", "followup_seqs": []})
