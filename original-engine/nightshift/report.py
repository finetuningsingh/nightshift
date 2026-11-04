"""Static run report: Markdown and a self-contained HTML page, both rendered from the ledger alone.

Guarantees
----------
* The HTML contains no ``<script``, no inline event handler and no external resource. Every dynamic string goes through
  ``html.escape(quote=True)``; text and attributes inside the SVG figures go through ``xml.sax.saxutils``.
* Markdown escapes ``& < > | [ ] ` `` and newlines in every dynamic string so ledger text cannot inject markup.
* The headline number is the SEALED delta of the final incumbent AGAINST THE BASELINE (the comparison the finish gate
  uses, ``workflows.gates.sealed_decision``), named as such, with the threshold an improvement had to exceed (not a
  confidence half-width). After a promotion chain each step was compared with the previous incumbent; those step-wise
  numbers are listed on a chain line and are never presented as the improvement over the baseline. The dev delta is always
  labelled "selected, optimistic".
* ``n_tested`` is "hypotheses that reached evaluation": experiments with a primary-grade record, even a failed one; an
  experiment that was invalid and never graded is not counted. Expected false promotions assume independent 5% dev draws;
  the report also prints the adaptive worst case ``1 - 0.975^n_sealed_queries`` for the sealed stage alone.
* Chain status reads "chain consistent" (hashes and links verify) and never "secure": a truncated or fully rewritten file
  is detectable only against a head anchor recorded elsewhere, and the report says so.
* A meta run with red-team lanes (``nightshift run --plant``) opens with the RED-TEAM paragraph (how many lanes are scripted
  attackers, which behaviors) before any SIMULATED paragraph: the scripted lanes are never passed off as model work.
* The opening says plainly where the words came from: a scripted run carries the SCRIPTED BACKEND paragraph; a run
  against a local test server carries the SIMULATED MODEL paragraph (it exercised the live code path but is not Meta
  model output); a live run names its endpoint kind and hosts as the ledger recorded them.
* Model names come in two clearly separate kinds. "Configured" labels are what the run asked for (``run.started.models``).
  "Models that answered" are counted from ``budget.charged`` (the model each reply named), so a role whose configured model
  never answered (a fallback model did) is not credited to the configured one: its label is left out of the configured line
  and the report says so. Nothing is called Muse Wonder (or anything else) unless a charged call shows it.
* The fault scoreboard separates detected, missed, "not evaluated" (an LLM-only invariant that no audit window judged) and
  "not run" (a planted red-team lane that a gate stopped before it ran: blocked as a duplicate, blocked by the budget, never
  launched). Recall excludes not-evaluated faults, not-run faults and negative controls; the not-run faults are counted on
  their own ("Planted faults that never ran: N") and the red-team paragraph says how many planted lanes actually ran.
* A report about a run against the local simulator (``run.origin == "simulated"``) never lets a stand-in read as a model:
  the models named by the charged calls are "the models the simulator reported answering (a stand-in, not Meta)", every
  cost figure is a "simulated list-price estimate (nothing was billed)", the LLM half of an invariant is "checked by a
  simulated judge" and the final summary is "written by the simulated supervisor (a scripted stand-in), not a model". Live
  and scripted reports are worded exactly as before.
* Isolation ``none`` (or not recorded) and ``unshare-net`` both carry a warning in the Integrity section: neither keeps the
  labels or the ledger out of the experiment code's reach (a read leaves no trace) and, run as root, ``unshare-net`` does not
  hide the API key either; what they do is DETECT a later change to the trusted directory. Only ``userns`` withholds the labels.
* When the hash chain does not verify, the FIRST lines of both formats say so (``LEDGER CHAIN BROKEN at seq N``) and the
  verdict line reads "unverified"; the report is still written, because what the ledger holds is worth reading, never because
  it can be trusted. The chain status is computed inside ``_collect`` (``build_state`` verifies the events it is given), so
  ``render_markdown`` and ``render_html`` need no extra argument.
* The output is a pure function of the ledger (time comes from the ledger, not the clock) and, only for the
  baseline-referenced sealed number after a promotion chain, of the task's noise parameters: taken from the ``task``
  argument, else from ``run.started`` when it records ``noise_margin`` and ``sigma_floor``, else from the task directory
  ``<repo>/tasks/<name>`` when its task hash equals the one the ledger recorded. When none of these is available the report
  falls back to the baseline record's ``noise_margin`` and no sigma floor, and says that the threshold may be understated.
"""
from __future__ import annotations

import html
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from xml.sax.saxutils import escape as xesc

from . import secrets
from .dashboard.state_api import (RED_TEAM_NOTICE, SIMULATED_BANNER, SIMULATED_COST_LABEL, SIMULATED_JUDGE_LABEL,
                                  SIMULATED_MODELS_LABEL, SIMULATED_RED_TEAM_NOTE, SIMULATED_SUPERVISOR_LABEL, build_state)
from .ledger import LEDGER_NAME, Ledger
from .types import BudgetLimits, Kind, LedgerView
from .view import build_view
from .workflows import gates

DEV_NULL_RATE = 0.05                   # one-sided 5% dev gate
SEALED_NULL_RATE = 0.025               # sealed gate: two-sided 5%, so 2.5% in the improving direction
FALSE_PROMOTION_RATE = DEV_NULL_RATE * SEALED_NULL_RATE

VERDICT_SENTENCES = {
    "success": "An improvement was found, replicated on the sealed split and accepted.",
    "inconclusive": "No improvement survived replication. The honest finding is that nothing beat the baseline.",
    "unresolved_defects": "The run ended with unresolved defects. Do not rely on any result until they are closed.",
    "missing_authority": "The run stopped because it ran out of authority (budget or a halt). The result is partial.",
}
NO_VERDICT = "The run has not recorded a verdict (it is still running, or it stopped without finishing)."

SCRIPTED_PARAGRAPH = ("SCRIPTED BACKEND. No language model produced this run. The supervisor, designer, implementer, reviewer "
                      "and reflector are scripted. Experiments were really executed and really graded, but the faults in them "
                      "were planted by the script. This is a fault-injection harness that demonstrates the audit machinery, "
                      "not an AI result.")

SIMULATED_PARAGRAPH = SIMULATED_BANNER

UNVERIFIED_PARAGRAPH = ("UNVERIFIED SOURCE: the ledger says the meta backend ran but records a scripted endpoint, which "
                        "contradicts it. Where the text in this run came from is not established, so it is not presented as a "
                        "Meta model result.")

RESULT_LABELS = {"detected": "detected", "missed": "MISSED", "not_evaluated": "not evaluated", "not_run": "not run"}

STATE_LABELS = {"ok": "checked, no open finding", "violated": "VIOLATED", "unchecked": "not checked",
                "insufficient": "insufficient evidence", "llm_not_run": "LLM not run (not shown as passing)",
                "ok_simulated": SIMULATED_JUDGE_LABEL + " (a scripted stand-in, not a model), no open finding"}

CHAIN_BROKEN = ("LEDGER CHAIN BROKEN at seq {seq}: this report was written from a ledger that fails verification; "
                "do not trust it")

CHAIN_NOTE = ("Chain consistent means every hash and link in the ledger verifies. A truncated or fully rewritten file is "
              "detectable only against a head anchor you recorded elsewhere (nightshift verify --expect-head).")


# --------------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------------

def _money(headline: dict, simulated: bool = False) -> str:
    """Cost for display: 'n/a' when no model call was charged (a scripted run), never a misleading $0.0000. Against the
    simulator the figure is named for what it is: a list-price estimate for a stand-in, nothing that was billed."""
    if headline.get("cost_known") is False or not headline.get("llm_calls"):
        return "n/a (no model calls)"
    if simulated:
        return f"{SIMULATED_COST_LABEL}: ${headline['cost_usd']:.4f}"
    return f"${headline['cost_usd']:.4f}"

def _f(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf") else None


def _num(x: Any, digits: int = 4, signed: bool = False) -> str:
    v = _f(x)
    if v is None:
        return "n/a"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def _when(ts: Any) -> str:
    v = _f(ts)
    if not v:
        return "n/a"
    try:
        return datetime.fromtimestamp(v, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (OverflowError, OSError, ValueError):     # a timestamp no calendar can show must not abort the report
        return "n/a"


def _max_replications(run: dict) -> int:
    """The run's sealed-query cap from ``run.started.limits``; the default limit when it is missing or malformed."""
    limits = run.get("limits") if isinstance(run, dict) else None
    value = limits.get("max_replications") if isinstance(limits, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return BudgetLimits().max_replications
    return value


def _replication_vs(events: list, exp_id: str) -> Optional[str]:
    """``vs_exp_id`` of the LAST replication record of ``exp_id``: the experiment whose sealed grade it was compared with."""
    vs: Optional[str] = None
    for ev in events:
        d = ev.data if isinstance(ev.data, dict) else {}
        if ev.kind == Kind.EXP_RECORDED and d.get("exp_id") == exp_id and d.get("phase") == "replication":
            vs = d.get("vs_exp_id") if isinstance(d.get("vs_exp_id"), str) else None
    return vs


def _gate_task(run_dir: Any, view: LedgerView, task: Any = None) -> tuple[Any, str]:
    """The task parameters ``gates.sealed_decision`` needs, and where they came from.

    Order: the ``task`` argument; ``run.started`` when it records both ``noise_margin`` and ``sigma_floor``; the task
    directory ``<repo>/tasks/<name>`` when its task hash equals the hash the ledger recorded; else ``partial``: the baseline
    record's ``noise_margin`` and no sigma floor (the threshold may then be understated)."""
    if task is not None:
        return task, "given"
    run = view.run if isinstance(view.run, dict) else {}
    minimize = run.get("objective", "minimize") != "maximize"
    nm, sf = run.get("noise_margin"), run.get("sigma_floor")
    if all(isinstance(x, (int, float)) and not isinstance(x, bool) and x == x for x in (nm, sf)):
        return SimpleNamespace(minimize=minimize, noise_margin=float(nm), sigma_floor=float(sf), train_metric=None), "ledger"
    name = run.get("task")
    if isinstance(name, str) and name and name == os.path.basename(name):
        candidate = Path(__file__).resolve().parent.parent / "tasks" / name
        if (candidate / "task.toml").is_file():
            try:
                from . import task as task_mod        # lazy: the report must load without the task machinery
                loaded = task_mod.load_task(candidate)
                if task_mod.task_hash(loaded) == run.get("task_hash"):
                    return loaded, "task_dir"
            except (ValueError, OSError):
                pass
    base = view.baseline
    base_nm = _f(base.noise_margin) if base is not None else None
    return SimpleNamespace(minimize=minimize, noise_margin=base_nm or 0.0, sigma_floor=0.0, train_metric=None), "partial"


def _sealed_result(run_dir: Any, view: LedgerView, events: list, task: Any = None) -> Optional[dict]:
    """The final incumbent's sealed result AGAINST THE BASELINE, plus its promotion chain; None when nothing was promoted.

    ``delta``/``margin`` are the baseline-referenced numbers the finish gate decides on. They are the recorded sealed numbers
    when the incumbent's replication was compared with the baseline itself (a single promotion); after a chain they come
    from ``gates.sealed_decision(task, view, incumbent, baseline)`` (None when that cannot be computed). The recorded
    numbers of the last step compare the incumbent with the PREVIOUS incumbent and are kept apart in ``step_*``."""
    inc = view.incumbent
    if inc is None or inc.exp_id == view.baseline_id:
        return None
    promos = [p for p in view.promotions if isinstance(p, dict)]
    promo = next((p for p in reversed(promos) if p.get("exp_id") == inc.exp_id), {})
    sd = inc.sealed_delta if inc.sealed_delta is not None else promo.get("sealed_delta")
    sm = inc.sealed_margin if inc.sealed_margin is not None else promo.get("sealed_margin")
    if _f(sd) is None:
        return None
    vs = _replication_vs(events, inc.exp_id)
    direct = vs == view.baseline_id or (vs is None and len(promos) <= 1)
    delta, margin, source, params = _f(sd), _f(sm), "recorded", "recorded"
    if not direct:
        delta = margin = None
        source = "unavailable"
        gate_task, params = _gate_task(run_dir, view, task)
        decision = gates.sealed_decision(gate_task, view, inc.exp_id, view.baseline_id) if view.baseline_id else None
        if decision is not None and decision.outcome != "invalid" and decision.delta is not None:
            delta, margin, source = _f(decision.delta), _f(decision.margin), "computed"
    chain = []
    for p in promos:
        pid = p.get("exp_id")
        exp = view.experiments.get(pid) if isinstance(pid, str) else None
        step_vs = _replication_vs(events, pid) if isinstance(pid, str) else None
        chain.append({"exp_id": pid, "vs": step_vs or p.get("prev_exp_id") or view.baseline_id,
                      "delta": _f(exp.sealed_delta if exp is not None and exp.sealed_delta is not None else p.get("sealed_delta")),
                      "margin": _f(exp.sealed_margin if exp is not None and exp.sealed_margin is not None else p.get("sealed_margin"))})
    return {"exp_id": inc.exp_id, "baseline_id": view.baseline_id, "delta": delta, "margin": margin, "source": source,
            "params": params, "replicated": bool(inc.replicated), "dev_delta": _f(inc.delta), "dev_vs": inc.vs_exp_id,
            "step_vs": vs, "step_delta": _f(sd), "step_margin": _f(sm), "chain": chain if len(chain) > 1 else []}


def _collect(run_dir: Any, task: Any = None) -> dict:
    """Everything both renderers need, derived from the ledger (``task``: see the module docstring)."""
    path = Path(run_dir) / LEDGER_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no ledger at {path}")
    events = Ledger.read_only(run_dir).events()
    view: LedgerView = build_view(events)
    now = float((view.finished or {}).get("ts") or view.last_ts or 0.0)
    st = build_state(view, None, events=events, now=now)
    run = st["run"]
    tested = [e for e in view.experiments.values() if e.exp_id != view.baseline_id and e.lane != "baseline"
              and e.grade_for("primary") is not None]
    sealed_queries = int(view.budget_used.get("replications", 0) or 0) or \
        sum(1 for e in view.experiments.values() if e.grade_for("replication") is not None)
    cap = _max_replications(view.run)
    sealed = _sealed_result(run_dir, view, events, task)
    acks: dict[str, dict] = {}
    for a in view.acks:
        if a.get("actor") in ("auditor", "human"):
            acks[str(a.get("finding_id"))] = a
    rc = st["fault_injection"]["recall"]
    verdict = run["verdict_class"]
    answers = _answers(events)
    n_tested = len(tested)
    n_queries = min(sealed_queries, cap)
    return {
        "state": st, "run": run, "verdict": verdict, "verdict_sentence": VERDICT_SENTENCES.get(verdict or "", NO_VERDICT),
        "scripted": run["backend"] == "scripted", "simulated": run["origin"] == "simulated", "sealed": sealed, "acks": acks,
        "stats": {"n_tested": n_tested, "n_dev_improved": sum(1 for e in tested if e.dev_outcome == "improved"),
                  "n_sealed_queries": sealed_queries, "max_replications": cap,
                  "expected_false": n_tested * FALSE_PROMOTION_RATE,
                  "adaptive_worst_case": 1.0 - (1.0 - SEALED_NULL_RATE) ** n_queries},
        "recall": rc["value"], "n_scored": rc["scored"], "n_caught": rc["caught"],
        "answers": answers,
        "summary": str((view.finished or {}).get("summary") or ""),
        "first_ts": view.first_ts, "last_ts": view.last_ts,
    }


def _answers(events: list) -> dict:
    """Which models answered, from ``budget.charged``: ``{"by_model": {model: calls}, "by_role": {role: {model: calls}}}``.

    A charge without a model name counts under ``"(unnamed)"`` rather than being dropped. The role is the actor up to the
    first ``:`` (``designer:A`` is the designer)."""
    by_model: dict[str, int] = {}
    by_role: dict[str, dict[str, int]] = {}
    for ev in events:
        d = ev.data if isinstance(ev.data, dict) else {}
        if ev.kind != Kind.BUDGET_CHARGED or d.get("kind") != "llm":
            continue
        model = str(d.get("model") or "(unnamed)")
        role = str(ev.actor or "unknown").split(":")[0] or "unknown"
        by_model[model] = by_model.get(model, 0) + 1
        by_role.setdefault(role, {})
        by_role[role][model] = by_role[role].get(model, 0) + 1
    return {"by_model": by_model, "by_role": by_role}


def _same_model(answered: str, configured: str) -> bool:
    """An answer may carry a version suffix (``muse-spark-1.3-0615`` answers ``muse-spark-1.3``)."""
    return answered == configured or answered.startswith(configured + "-") or configured.startswith(answered + "-")


MAX_ANSWERING_MODELS = 12
MAX_MODEL_NAME = 80


def _short(name: str) -> str:
    return name if len(name) <= MAX_MODEL_NAME else name[:MAX_MODEL_NAME - 3] + "..."


def _model_text(run: dict, answers: dict) -> str:
    """Two sentences: the CONFIGURED labels (minus any role whose configured model never answered) and the models that
    ANSWERED with their call counts."""
    configured = {str(k): str(v) for k, v in sorted((run.get("models") or {}).items())}
    kept, replaced = [], []
    for role, model in configured.items():
        seen = answers["by_role"].get(role)
        if seen and not any(_same_model(m, model) for m in seen):
            replaced.append(role)
        else:
            kept.append(f"{role}={model}")
    text = ("Configured model labels (what the run asked for, not proof of which model answered): "
            f"{', '.join(kept) or 'none recorded'}.")
    if replaced:
        text += (f" The configured model of {', '.join(replaced)} is not listed: it never answered a charged call "
                 "(another model did, see below).")
    if answers["by_model"]:
        # The names come from the server's replies: show a bounded, shortened list so a host that answers under a new name every
        # time cannot swamp the report. Every charge is still in the ledger and in the models usage table.
        ranked = sorted(answers["by_model"].items())
        shown, hidden = ranked[:MAX_ANSWERING_MODELS], ranked[MAX_ANSWERING_MODELS:]
        lead = (f" {SIMULATED_MODELS_LABEL}, counted from the charged calls: " if run.get("origin") == "simulated"
                else " Models that answered (from the charged calls, with call counts): ")
        text += lead + ", ".join(f"{_short(m)} ({n} call{'s' if n != 1 else ''})" for m, n in shown)
        if hidden:
            extra = sum(n for _, n in hidden)
            text += f", and {len(hidden)} other model name{'s' if len(hidden) != 1 else ''} ({extra} call{'s' if extra != 1 else ''})"
        text += "."
    else:
        text += " No model call was charged, so the ledger shows no model answering."
    return text


def _endpoint_text(run: dict) -> str:
    ep = run.get("endpoint")
    if not ep:
        return "no endpoint record in this ledger"
    hosts = ", ".join(ep["hosts"]) or "no host recorded"
    return f"endpoint kind {ep['kind']}, host {hosts}"


def _backend_statement(run: dict, answers: dict | None = None) -> str:
    answers = answers or {"by_model": {}, "by_role": {}}
    models = ", ".join(f"{k}={v}" for k, v in sorted((run.get("models") or {}).items())) or "none recorded"
    named = _model_text(run, answers)
    if run["backend"] == "scripted":
        return f"Backend that produced this run: scripted (no language model). Recorded model labels: {models}."
    if run["origin"] == "simulated":
        return (f"Backend that produced this run: meta client against a local test server ({_endpoint_text(run)}), a simulation of "
                f"the live code path and not a Meta model result. {named}")
    if run["backend"] == "meta" and run["origin"] == "unknown":
        return (f"Backend recorded for this run: meta, but the ledger's endpoint record ({_endpoint_text(run)}) contradicts it. "
                f"This report does not claim these words came from a Meta model. {named}")
    if run["backend"] == "meta":
        where = _endpoint_text(run)
        custom = " This is not the Meta API host." if (run.get("endpoint") or {}).get("kind") == "custom" else ""
        return f"Backend that produced this run: meta (Meta Model API client; {where}).{custom} {named}"
    return f"Backend that produced this run: {run['backend'] or 'unknown'}. {named}"


def _mode_paragraph(c: dict) -> Optional[str]:
    """The paragraph that opens the report when the run is not a plain live run."""
    if c["scripted"]:
        return SCRIPTED_PARAGRAPH
    if c["simulated"]:
        return SIMULATED_PARAGRAPH
    if c["run"]["backend"] == "meta" and c["run"]["origin"] == "unknown":
        return UNVERIFIED_PARAGRAPH
    return None


def _mode_paragraphs(c: dict) -> list[str]:
    """The opening paragraphs, most important first: the red-team notice (when the operator planted scripted attacker lanes into
    a meta run), then the scripted / simulated / unverified paragraph."""
    out: list[str] = []
    red = c["run"].get("red_team") or []
    if red:
        names = ", ".join(f"{r['behavior']} ({r['scripted_id']})" for r in red)
        ran = c["state"]["fault_injection"]["red_team"]["text"]
        out.append(RED_TEAM_NOTICE.format(n=len(red)) + f" Planted: {names}." + (f" {SIMULATED_RED_TEAM_NOTE}" if c["simulated"] else "") +
                   (f" {ran}" if ran else "") +
                   " Those lanes are labelled as planted faults in the experiments table; everything else here is the model "
                   "roles' work.")
    mode = _mode_paragraph(c)
    if mode:
        out.append(mode)
    return out


def _context_statement(run: dict) -> Optional[str]:
    """One sentence naming the background files the roles were given (the ledger holds only their hashes), or None."""
    files = run.get("context") or []
    if not files:
        return None
    listed = ", ".join(f"{f['name']} ({f['chars'] if f['chars'] is not None else '?'} characters, sha256 {f['sha256'][:12]})"
                       for f in files)
    text = ("Background material given to the supervisor and designer by the operator (the ledger records only name, size and "
            f"hash, not the text): {listed}.")
    if any(f.get("not_reloaded_after_resume") for f in files):
        text += (" The run was resumed after a crash and --context cannot be given to a resume, so the roles that ran after the "
                 "resume worked WITHOUT this background.")
    return text


def _who(c: dict) -> str:
    if c["scripted"]:
        return "written by the script, not a model"
    return SIMULATED_SUPERVISOR_LABEL if c["simulated"] else "written by the supervisor"


def _chain_text(led: dict) -> str:
    if led["chain_ok"]:
        return "chain consistent"
    return f"CHAIN INCONSISTENT at seq {led['bad_seq']}: do not trust this ledger"


def _chain_banner(c: dict) -> Optional[str]:
    """The alarm that opens a report written from a ledger whose hash chain does not verify; None when it verifies."""
    led = c["state"]["ledger"]
    if led["chain_ok"] is True:
        return None
    seq = led["bad_seq"] if led["bad_seq"] is not None else "unknown"
    return CHAIN_BROKEN.format(seq=seq)


def _verdict_title(c: dict) -> str:
    """The verdict as the heading shows it: followed by 'unverified' when the ledger it was read from fails verification."""
    base = c["verdict"] or "no verdict"
    return base + (" (unverified)" if _chain_banner(c) else "")


_SIM_USAGE_NOTE = (f"{SIMULATED_MODELS_LABEL}: the model names below are what the simulator's replies said. Every usd figure is a "
                   f"{SIMULATED_COST_LABEL}.")


def _usd_head(c: dict) -> str:
    return "usd (simulated list-price estimate, nothing was billed)" if c["simulated"] else "usd"


def _isolation_text(run: dict) -> str:
    iso = run["isolation"] or "not recorded"
    if iso in ("none", "not recorded"):
        return (f"{iso}. Without isolation, same-user experiment code could read the trusted directory and the ledger by "
                "path; protection rests on hash checks that detect tampering after the fact.")
    if iso == "unshare-net":
        return ("unshare-net. It blocks network connections, but experiment code can still read the labels (the trusted directory) "
                "and the ledger by path, and a read leaves no trace; when Nightshift runs as root the API key can be readable too "
                "unless the pid-namespace variant was in use. What it does detect is a change to the trusted directory after the "
                "run. Best-effort containment, not a security boundary.")
    return f"{iso}. Best-effort containment, not a security boundary."


# --------------------------------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------------------------------
def _m(x: Any) -> str:
    s = "" if x is None else str(x)
    for a, b in (("&", "&amp;"), ("\\", "&#92;"), ("<", "&lt;"), (">", "&gt;"), ("|", "\\|"), ("[", "\\["), ("]", "\\]"),
                 ("`", "&#96;")):
        s = s.replace(a, b)
    return " ".join(s.split())


def _md_table(head: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["_none_", ""]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(_m(c) for c in r) + " |" for r in rows]
    return out + [""]


def _recall_line(c: dict, fi: dict) -> str:
    recall = "n/a" if c["recall"] is None else f"{c['n_caught']}/{c['n_scored']} = {c['recall']:.0%}"
    rc = fi["recall"]
    extra = f" Not evaluated: {rc['not_evaluated']} (needs an LLM judge; not run offline, excluded from recall)." if rc["not_evaluated"] else ""
    running = "" if (c["run"]["finished"] or not fi["missed"]) else (" The run has not finished, so a fault listed as missed may still be caught by a later "
                                               "audit window (provisional).")
    neg = f" Negative controls passed: {rc['negatives_ok']}/{rc['negatives_total']}." if rc["negatives_total"] else ""
    never = (f" Planted faults that never ran: {rc['not_run']} (a gate stopped them before they ran; they are neither caught nor "
             "missed and are excluded from recall)." if rc.get("not_run") else "")
    return f"Planted faults caught (recall): {recall}.{extra}{never}{neg} False positives: {len(fi['false_positives'])}.{running}"


def _fault_rows(fi: dict) -> list[list[Any]]:
    rows = []
    for p in fi["planted"]:
        expected = ", ".join(p["expected"]) or ("must not fire " + ", ".join(p["must_not_fire"]))
        label = p["exp_id"] or (f"{p['lane']} (no experiment was created)" if p.get("lane") else
                                f"finish request (seq {p.get('finish_seq', '?')})")
        rows.append([label, p["fault"], expected, RESULT_LABELS.get(p["result"], p["result"]), p["reason"]])
    return rows


def _threshold(margin: Any) -> str:
    return f"an improvement had to exceed {_num(margin, 4)}"


def _sealed_line(c: dict) -> str:
    """The headline: the final incumbent's sealed improvement over the BASELINE and the threshold it had to exceed."""
    s = c["sealed"]
    if s is None:
        return "No candidate was promoted, so there is no replicated improvement to report."
    where = f"Sealed delta vs baseline {s['baseline_id']}" if s["baseline_id"] else "Sealed delta vs baseline"
    tail = f"sealed split, positive is better, {'replicated' if s['replicated'] else 'not replicated'}"
    if s["delta"] is None:
        return (f"{where}: not available in this report (the sealed grades or the task's noise parameters needed to "
                f"compare {s['exp_id']} with the baseline are missing); {tail}")
    return f"{where}: {_num(s['delta'], 4, True)} ({_threshold(s['margin'])}; {tail})"


def _result_notes(c: dict) -> list[str]:
    """Sentences under the headline: where the numbers came from, the promotion chain, the dev delta."""
    s = c["sealed"]
    if s is None:
        return []
    out: list[str] = []
    if s["source"] == "computed":
        out.append(f"The headline compares {s['exp_id']} with the baseline {s['baseline_id']} on the sealed split, the comparison the "
                   "finish gate uses; it was recomputed from the sealed grades in the ledger because the promotion recorded only "
                   "a comparison with the previous incumbent.")
        if s["params"] == "partial":
            out.append("The task's sigma floor was not available to the report, so the threshold above may be understated.")
    if len(s["chain"]) > 1:
        steps = "; ".join(f"{x['exp_id']} {_num(x['delta'], 4, True)} vs {x['vs']} ({_threshold(x['margin'])})" for x in s["chain"])
        out.append("Promotion chain: each step below is a sealed comparison with the PREVIOUS incumbent, not with the baseline "
                   f"(the headline above is the comparison with the baseline): {steps}.")
    elif s["source"] == "unavailable" and s["step_delta"] is not None:
        out.append(f"The recorded sealed comparison of {s['exp_id']} is with {s['step_vs'] or 'its comparison base'}, not with the "
                   f"baseline: {_num(s['step_delta'], 4, True)} ({_threshold(s['step_margin'])}).")
    dev_vs = f" vs {s['dev_vs']}" if s.get("dev_vs") else ""
    out.append(f"Dev delta for the same experiment: {_num(s['dev_delta'], 4, True)}{dev_vs} (selected, optimistic).")
    return out


def _stat_lines(stats: dict) -> list[str]:
    """The selection-statistics bullets (shared by Markdown and HTML)."""
    return [
        f"tested: {stats['n_tested']} experiments (n_tested = hypotheses that reached evaluation: an experiment with a "
        "primary-grade record, even a failed one; one that was invalid and never graded is not counted)",
        f"improved on dev: {stats['n_dev_improved']} (n_dev_improved)",
        f"sealed-split queries: {stats['n_sealed_queries']} (n_sealed_queries; capped at max_replications = {stats['max_replications']})",
        f"expected false promotions under the null = n_tested * 0.05 * 0.025 = {stats['expected_false']:.4f} "
        "(assumes every dev draw is independent with a 5% chance of passing; a supervisor that sees dev scores and tunes against "
        "them can beat that)",
        f"adaptive worst case = 1 - 0.975^n_sealed_queries = {stats['adaptive_worst_case']:.4f}: the probability that at least one "
        "null passed the sealed check if every sealed query was a null candidate that got through dev. It bounds the sealed "
        "stage alone and says nothing about how many nulls reached it.",
    ]


def _experiment_rows(st: dict) -> list[list[Any]]:
    return [[e["exp_id"], e["lane"], e["hypothesis"], e["outcome"] or e["status"], _num(e["score"]), _num(e["delta"], 4, True),
             _num(e["sealed_delta"], 4, True), "yes" if e["replicated"] else "no", e["review"] or "", e["planted_fault"] or ""]
            for e in st["experiments"]]


EXP_HEAD = ["exp", "lane", "hypothesis", "outcome", "dev score", "dev delta (selected)", "sealed delta (vs its base)", "replicated",
            "review", "planted fault"]
EXP_NOTE = ("dev delta (selected) is measured on the data the candidates were chosen on, so it is optimistic. The sealed delta of an "
            "experiment is its sealed comparison with the incumbent at that time (the baseline until something was promoted), not "
            "necessarily with the baseline.")


def render_markdown(run_dir: Any, task: Any = None) -> str:
    """The report as Markdown (no figures; see ``render_html``). ``task``: the run's TaskSpec, optional (module docstring)."""
    c = _collect(run_dir, task)
    st, run, h, stats = c["state"], c["run"], c["state"]["headline"], c["stats"]
    L: list[str] = []
    broken = _chain_banner(c)
    if broken:
        L += [f"> **{_m(broken)}**", ""]
    L += [f"# Nightshift report: {_m(run['task'])}", "", f"Run `{_m(run['run_id'])}`, started {_when(run['started_ts'])}.", ""]
    for mode in _mode_paragraphs(c):
        L += [f"> **{_m(mode)}**", ""]
    L += [_m(_backend_statement(run, c["answers"])), ""]
    if _context_statement(run):
        L += [_m(_context_statement(run)), ""]
    L += [f"## Verdict: {_m(_verdict_title(c))}", "", c["verdict_sentence"], ""]
    L += ["## Headline", ""]
    L += _md_table(["claims made", "claims accepted", "claims refused", "unreplicated improvements blocked", "speedup",
                    "simulated cost" if c["simulated"] else "cost"],
                   [[h["claims_made"], h["claims_accepted"], h["claims_refused"], h["unreplicated_blocked"],
                     f"{h['speedup']:.2f}x", _money(h, c["simulated"])]])
    L += ["## Result", "", f"**{_m(_sealed_line(c))}**", ""]
    for note in _result_notes(c):
        L += [_m(note), ""]
    L += [f"* {_m(line)}" for line in _stat_lines(stats)] + [""]
    if c["summary"]:
        L += [f"Final summary ({_who(c)}):", "", f"> {_m(c['summary'])}", ""]
    L += ["## Experiments", ""] + _md_table(EXP_HEAD, _experiment_rows(st)) + [_m(EXP_NOTE), ""]
    L += ["## Findings", ""]
    L += _md_table(["finding", "invariant", "severity", "verdict", "status", "evidence seqs", "rationale"],
                   [[f["finding_id"], f"{f['invariant_id']} {f['title']}", f["severity"], f["verdict"], f["status"],
                     ", ".join(str(s) for s in f["evidence_seqs"]) or "-", f["rationale"]] for f in st["findings"]])
    dismissed = [f for f in st["findings"] if f["status"] == "dismissed"]
    L += ["### Dismissed findings", ""]
    L += _md_table(["finding", "reason", "by"], [[f["finding_id"], c["acks"].get(f["finding_id"], {}).get("reason", ""),
                                                  c["acks"].get(f["finding_id"], {}).get("by", "")] for f in dismissed])
    L += ["## Invariants coverage", ""]
    L += _md_table(["id", "invariant", "kind", "state", "open"],
                   [[i["id"], i["title"], i["kind"], STATE_LABELS.get(i["state"], i["state"]), i["open"]] for i in st["invariants"]])
    fi = st["fault_injection"]
    if c["scripted"] or fi["planted"]:
        L += ["## Fault-injection scoreboard", "", _recall_line(c, fi), ""]
        L += _md_table(["experiment", "fault", "expected", "result", "detail"], _fault_rows(fi))
        if fi["explained"]:
            L += ["Findings explained by a planted premature finish (not counted as false positives):", ""]
            L += _md_table(["finding", "invariant", "exp", "finish seq"], [
                [x["finding_id"], x["invariant_id"], x["exp_id"] or "", x["finish_seq"]] for x in fi["explained"]])
        if fi["false_positives"]:
            L += ["False positives:", ""] + _md_table(["finding", "invariant", "exp"], [
                [f["finding_id"], f["invariant_id"], f["exp_id"] or ""] for f in fi["false_positives"]])
    L += ["## Models usage", ""]
    if st["models_usage"]:
        if c["simulated"]:
            L += [_m(_SIM_USAGE_NOTE), ""]
        L += _md_table(["role", "model", "calls", "prompt tok", "completion tok", "cache", _usd_head(c), "effort", "retries"],
                       [[u["role"], u["model"], u["calls"], u["prompt_tokens"], u["completion_tokens"],
                         u["cache"] if u["cache"] == "n/a" else f"{u['cached_tokens']} cached", f"{u['usd']:.4f}",
                         u["effort"] or "-", u["retries"]] for u in st["models_usage"]])
    else:
        L += ["No model calls were charged." + (" The scripted backend runs no language model." if c["scripted"] else ""), ""]
    led = st["ledger"]
    L += ["## Integrity", "", f"* Ledger: {led['events']} events, {_m(_chain_text(led))}", f"* Head anchor: `{_m(led['head_anchor'])}`",
          f"* Isolation: {_m(_isolation_text(run))}", "", CHAIN_NOTE, ""]
    return secrets.scrub_text("\n".join(L))


# --------------------------------------------------------------------------------------------------
# svg figures
# --------------------------------------------------------------------------------------------------
_W, _H, _ML, _MR, _MT, _MB = 640, 300, 56, 16, 20, 56


def _scale(vals: list[float]) -> tuple[float, float]:
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.12 or abs(hi) * 0.05 or 1.0
    return lo - pad, hi + pad


def _svg_open(uid: str, title: str, desc: str) -> list[str]:
    return [f'<svg viewBox="0 0 {_W} {_H}" width="100%" role="img" aria-labelledby="{uid}t {uid}d" class="fig">',
            f'<title id="{uid}t">{xesc(title)}</title>', f'<desc id="{uid}d">{xesc(desc)}</desc>']


def _axes(lo: float, hi: float, n: int, labels: list[str], ylabel: str) -> tuple[list[str], Any, Any]:
    pw, ph = _W - _ML - _MR, _H - _MT - _MB

    def X(i: float) -> float:
        return _ML + (pw * (i + 0.5) / max(1, n))

    def Y(v: float) -> float:
        return _MT + ph * (1 - (v - lo) / (hi - lo))

    out = [f'<rect x="{_ML}" y="{_MT}" width="{pw}" height="{ph}" class="plot"/>']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        out.append(f'<line x1="{_ML}" x2="{_ML + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>')
        out.append(f'<text x="{_ML - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" class="tick">{xesc(f"{v:.3g}")}</text>')
    step = max(1, n // 12 + (1 if n % 12 else 0))
    for i, lab in enumerate(labels):
        if i % step == 0:
            out.append(f'<text x="{X(i):.1f}" y="{_H - _MB + 16}" text-anchor="middle" class="tick">{xesc(lab)}</text>')
    out.append(f'<text x="12" y="{_MT + ph / 2:.0f}" transform="rotate(-90 12 {_MT + ph / 2:.0f})" text-anchor="middle" '
               f'class="axis">{xesc(ylabel)}</text>')
    return out, X, Y


def _marker(kind: str, x: float, y: float) -> str:
    if kind == "baseline":
        return f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" class="m-base"/>'
    if kind == "promoted":
        return f'<polygon points="{x:.1f},{y - 7:.1f} {x + 7:.1f},{y:.1f} {x:.1f},{y + 7:.1f} {x - 7:.1f},{y:.1f}" class="m-prom"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" class="m-cand"/>'


def score_figure(st: dict) -> str:
    """Dev score per experiment with the baseline line and the incumbent step line."""
    series = st["series"]
    if not series:
        return '<p class="note">No graded experiments yet, so there is no score figure.</p>'
    metric = st["run"]["primary_metric"] or "dev score"
    better = "lower is better" if st["run"]["objective"] == "minimize" else "higher is better"
    lo, hi = _scale([s["score"] for s in series])
    ax, X, Y = _axes(lo, hi, len(series), [s["exp_id"] for s in series], f"{metric} (dev)")
    out = _svg_open("f1", "Dev score over experiments", f"{metric} on the dev split for each experiment, {better}.") + ax
    base = next((s for s in series if s["kind"] == "baseline"), None)
    if base is not None:
        out.append(f'<line x1="{_ML}" x2="{_W - _MR}" y1="{Y(base["score"]):.1f}" y2="{Y(base["score"]):.1f}" class="baseline"/>')
        out.append(f'<text x="{_W - _MR - 4}" y="{Y(base["score"]) - 5:.1f}" text-anchor="end" class="lab">baseline</text>')
    cur = base["score"] if base else series[0]["score"]
    pts = [(X(0), Y(cur))]
    for i, s in enumerate(series):
        if s["kind"] == "promoted":
            pts += [(X(i), Y(cur)), (X(i), Y(s["score"]))]
            cur = s["score"]
    pts.append((_W - _MR, Y(cur)))
    out.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '" class="incumbent"/>')
    for i, s in enumerate(series):
        out.append(_marker(s["kind"], X(i), Y(s["score"])))
    out.append(f'<text x="{_ML}" y="{_H - 6}" class="lab">square baseline, circle candidate, diamond promoted; line = incumbent; {xesc(better)}</text>')
    return "\n".join(out + ["</svg>"])


def gap_figure(st: dict) -> str:
    """Train versus held-out (dev) score per experiment: the vertical bar is the generalisation gap."""
    rows = [s for s in st["series"] if s["train"] is not None]
    if not rows:
        return '<p class="note">No train metric was recorded, so there is no train versus held-out figure.</p>'
    vals = [s["score"] for s in rows] + [s["train"] for s in rows]
    lo, hi = _scale(vals)
    ax, X, Y = _axes(lo, hi, len(rows), [s["exp_id"] for s in rows], "train vs held-out (dev)")
    out = _svg_open("f2", "Train versus held-out score",
                    "For each experiment the train score (hollow) and the held-out dev score (filled), joined by the gap.") + ax
    for i, s in enumerate(rows):
        out.append(f'<line x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{Y(s["train"]):.1f}" y2="{Y(s["score"]):.1f}" class="gap"/>')
        out.append(f'<circle cx="{X(i):.1f}" cy="{Y(s["train"]):.1f}" r="5" class="m-train"/>')
        out.append(f'<circle cx="{X(i):.1f}" cy="{Y(s["score"]):.1f}" r="5" class="m-held"/>')
    out.append(f'<text x="{_ML}" y="{_H - 6}" class="lab">hollow = train, filled = held-out (dev); a long bar means the model fits train better than held-out</text>')
    return "\n".join(out + ["</svg>"])


# --------------------------------------------------------------------------------------------------
# html
# --------------------------------------------------------------------------------------------------
CSS = """
:root{--bg:#fff;--fg:#1c1f26;--muted:#5b6270;--line:#d9dde4;--panel:#f5f6f8;--warn-bg:#fff4d6;--warn-fg:#6b4a00;--warn-line:#e0b000;
--bad:#b3261e;--good:#17663f;--blue:#1f5fbf;--grey:#6b7280}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8ec;--muted:#9aa1ad;--line:#2d323b;--panel:#1c1f26;--warn-bg:#3a2e08;
--warn-fg:#ffe08a;--warn-line:#a07c00;--bad:#ff8a80;--good:#6fd6a0;--blue:#7fb0ff;--grey:#9aa1ad}}
body{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1040px;padding:16px}
h1,h2,h3{line-height:1.25}h2{margin-top:2rem;border-bottom:1px solid var(--line);padding-bottom:.25rem}
table{border-collapse:collapse;width:100%;font-size:.9rem;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:.35rem .5rem;text-align:left;vertical-align:top}th{background:var(--panel)}
.banner{background:var(--warn-bg);color:var(--warn-fg);border:2px solid var(--warn-line);padding:.75rem 1rem;font-weight:700;margin:1rem 0}
.banner.alarm{background:var(--bad);color:var(--bg);border-color:var(--bad);font-size:1.1rem}
.big{font-size:1.6rem;font-weight:700}.muted,.note{color:var(--muted)}code{background:var(--panel);padding:0 .25rem}
.cards{display:flex;flex-wrap:wrap;gap:.75rem}.card{background:var(--panel);border:1px solid var(--line);padding:.5rem .8rem;min-width:8rem}
.card b{display:block;font-size:1.3rem}.bad{color:var(--bad);font-weight:700}.good{color:var(--good);font-weight:700}
.fig{max-width:720px;height:auto}.fig .plot{fill:var(--panel);stroke:var(--line)}.fig .grid{stroke:var(--line)}
.fig .tick,.fig .axis,.fig .lab{fill:var(--muted);font:11px system-ui,sans-serif}
.fig .baseline{stroke:var(--grey);stroke-dasharray:5 4;stroke-width:1.5}.fig .incumbent{fill:none;stroke:var(--good);stroke-width:2}
.fig .m-base{fill:var(--grey)}.fig .m-cand{fill:var(--blue)}.fig .m-prom{fill:var(--good)}
.fig .gap{stroke:var(--bad);stroke-width:2}.fig .m-train{fill:var(--bg);stroke:var(--blue);stroke-width:2}.fig .m-held{fill:var(--blue)}
"""


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _table(head: list[str], rows: list[list[Any]], empty: str = "none") -> str:
    if not rows:
        return f'<p class="note">{_e(empty)}</p>'
    th = "".join(f'<th scope="col">{_e(h)}</th>' for h in head)
    body = "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def render_html(run_dir: Any, task: Any = None) -> str:
    """The report as one self-contained HTML page (inline CSS, server-rendered SVG, no script)."""
    c = _collect(run_dir, task)
    st, run, h, stats, led = c["state"], c["run"], c["state"]["headline"], c["stats"], c["state"]["ledger"]
    P: list[str] = []
    broken = _chain_banner(c)
    if broken:
        P.append(f'<p class="banner alarm" role="alert">{_e(broken)}</p>')
    P.append(f"<h1>Nightshift report: {_e(run['task'])}</h1>")
    P.append(f'<p class="muted">Run <code>{_e(run["run_id"])}</code>, started {_e(_when(run["started_ts"]))}.</p>')
    for mode in _mode_paragraphs(c):
        P.append(f'<p class="banner" role="note">{_e(mode)}</p>')
    P.append(f"<p>{_e(_backend_statement(run, c['answers']))}</p>")
    if _context_statement(run):
        P.append(f"<p>{_e(_context_statement(run))}</p>")
    P.append(f"<h2>Verdict: {_e(_verdict_title(c))}</h2><p>{_e(c['verdict_sentence'])}</p>")
    cards = [("claims made", h["claims_made"]), ("claims accepted", h["claims_accepted"]), ("claims refused", h["claims_refused"]),
             ("unreplicated improvements blocked", h["unreplicated_blocked"]), ("speedup", f"{h['speedup']:.2f}x"),
             ("simulated cost" if c["simulated"] else "cost", _money(h, c["simulated"]))]
    P.append("<h2>Headline</h2><div class=\"cards\">" + "".join(f'<div class="card"><b>{_e(v)}</b>{_e(k)}</div>' for k, v in cards) + "</div>")
    P.append("<h2>Result</h2>")
    P.append(f'<p class="big">{_e(_sealed_line(c))}</p>')
    for note in _result_notes(c):
        P.append(f"<p>{_e(note)}</p>")
    P.append("<ul>" + "".join(f"<li>{_e(t)}</li>" for t in _stat_lines(stats)) + "</ul>")
    if c["summary"]:
        P.append(f"<p>Final summary ({_e(_who(c))}):</p><blockquote>{_e(c['summary'])}</blockquote>")
    P.append("<h2>Figures</h2>" + score_figure(st) + gap_figure(st))
    P.append("<h2>Experiments</h2>" + _table(EXP_HEAD, _experiment_rows(st), "no experiments") + f'<p class="note">{_e(EXP_NOTE)}</p>')
    P.append("<h2>Findings</h2>" + _table(
        ["finding", "invariant", "severity", "verdict", "status", "evidence seqs", "rationale"],
        [[f["finding_id"], f"{f['invariant_id']} {f['title']}", f["severity"], f["verdict"], f["status"],
          ", ".join(str(s) for s in f["evidence_seqs"]) or "-", f["rationale"]] for f in st["findings"]], "no findings"))
    dismissed = [f for f in st["findings"] if f["status"] == "dismissed"]
    P.append("<h3>Dismissed findings</h3>" + _table(["finding", "reason", "by"], [
        [f["finding_id"], c["acks"].get(f["finding_id"], {}).get("reason", ""), c["acks"].get(f["finding_id"], {}).get("by", "")]
        for f in dismissed], "no findings were dismissed"))
    P.append("<h2>Invariants coverage</h2>" + _table(
        ["id", "invariant", "kind", "state", "open"],
        [[i["id"], i["title"], i["kind"], STATE_LABELS.get(i["state"], i["state"]), i["open"]] for i in st["invariants"]],
        "no invariant registry available"))
    fi = st["fault_injection"]
    if c["scripted"] or fi["planted"]:
        P.append(f"<h2>Fault-injection scoreboard</h2><p>{_e(_recall_line(c, fi))}</p>")
        P.append(_table(["experiment", "fault", "expected", "result", "detail"], _fault_rows(fi), "no faults were planted"))
        if fi["explained"]:
            P.append("<h3>Findings explained by a planted premature finish</h3><p class=\"note\">Not counted as false positives.</p>"
                     + _table(["finding", "invariant", "exp", "finish seq"], [
                         [x["finding_id"], x["invariant_id"], x["exp_id"] or "", x["finish_seq"]] for x in fi["explained"]]))
        if fi["false_positives"]:
            P.append("<h3>False positives</h3>" + _table(["finding", "invariant", "exp"], [
                [f["finding_id"], f["invariant_id"], f["exp_id"] or ""] for f in fi["false_positives"]]))
    P.append("<h2>Models usage</h2>" + (f'<p class="note">{_e(_SIM_USAGE_NOTE)}</p>' if c["simulated"] and st["models_usage"] else "") + _table(
        ["role", "model", "calls", "prompt tok", "completion tok", "cache", _usd_head(c), "effort", "retries"],
        [[u["role"], u["model"], u["calls"], u["prompt_tokens"], u["completion_tokens"],
          "n/a" if u["cache"] == "n/a" else f"{u['cached_tokens']} cached", f"{u['usd']:.4f}", u["effort"] or "-", u["retries"]]
         for u in st["models_usage"]], "no model calls were charged"))
    ok = bool(led["chain_ok"])
    P.append("<h2>Integrity</h2><ul>" + f"<li>Ledger: {led['events']} events, "
             f'<span class="{"good" if ok else "bad"}">{_e(_chain_text(led))}</span></li>'
             f"<li>Head anchor: <code>{_e(led['head_anchor'])}</code></li><li>Isolation: {_e(_isolation_text(run))}</li></ul>"
             f'<p class="note">{_e(CHAIN_NOTE)}</p>')
    doc = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
           "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
           f"<title>{_e('Nightshift report ' + str(run['task']))}</title><style>{CSS}</style></head><body>\n"
           + "\n".join(P) + "\n</body></html>\n")
    return secrets.scrub_text(doc)


def write_reports(run_dir: Any, out_dir: Any = None, task: Any = None) -> tuple[Path, Path]:
    """Write ``report.md`` and ``report.html`` (atomically) into ``out_dir`` (default: the run directory)."""
    out = Path(out_dir) if out_dir else Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in (("report.md", render_markdown(run_dir, task)), ("report.html", render_html(run_dir, task))):
        tmp = out / (name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, out / name)
        paths.append(out / name)
    return paths[0], paths[1]
