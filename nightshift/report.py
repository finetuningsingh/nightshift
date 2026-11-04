"""Status and evidence-linked reporting (spec 3.12, A17, A18)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import NightshiftError
from .ledger import read_events


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def collect_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    summary = _load_json(root / "run.json")
    events: list[dict[str, Any]] = []
    ledger_path = root / "ledger.jsonl"
    if ledger_path.is_file():
        try:
            events = read_events(ledger_path)
        except NightshiftError:
            events = []
    evaluations: list[dict[str, Any]] = []
    eval_dir = root / "evaluations"
    if eval_dir.is_dir():
        for path in sorted(eval_dir.glob("*.json")):
            record = _load_json(path)
            if record:
                evaluations.append(record)
    seq_by_eval: dict[str, int] = {}
    fake = bool(summary.get("fake"))
    for event in events:
        payload = event.get("payload") or {}
        if payload.get("fake"):
            fake = True
        if event.get("type") in ("evaluation_finished", "evaluation_failed"):
            if payload.get("evaluation_id"):
                seq_by_eval[str(payload["evaluation_id"])] = int(event.get("seq", 0))
    if any(record.get("fake") for record in evaluations):
        fake = True
    return {
        "run_dir": root,
        "summary": summary,
        "events": events,
        "evaluations": evaluations,
        "seq_by_eval": seq_by_eval,
        "fake": fake,
    }


def status_text(run_dir: str | Path) -> str:
    data = collect_run(run_dir)
    events, summary = data["events"], data["summary"]
    if not events and not summary:
        raise NightshiftError(f"no Nightshift run found in {run_dir}")
    last = events[-1] if events else {}
    budgets = summary.get("budgets") or {}
    label = " [fake]" if data["fake"] else ""
    lines = [
        f"run: {summary.get('run_name', Path(run_dir).name)}{label}",
        f"status: {summary.get('status', 'in progress')}",
        f"phase: {summary.get('phase', last.get('type', 'unknown'))}",
        f"events: {len(events)} (last seq {last.get('seq', 0)}: {last.get('type', 'n/a')})",
        f"original pack: {str(summary.get('original_pack_version', ''))[:16]}",
        f"final pack: {str(summary.get('final_pack_version', ''))[:16]}",
        "spend: ${:.4f} committed+reserved of ${:.4f} ceiling{}".format(
            float(budgets.get("committed_and_reserved_usd", 0.0)),
            float(budgets.get("ceiling_usd", 0.0)),
            label,
        ),
        f"evaluations: {len(data['evaluations'])} "
        f"({sum(1 for e in data['evaluations'] if e.get('status') == 'failed')} failed)",
    ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_report(run_dir: str | Path) -> str:
    root = Path(run_dir)
    data = collect_run(root)
    summary, events = data["summary"], data["events"]
    fake = data["fake"]
    budgets = summary.get("budgets") or {}
    lines: list[str] = ["# Nightshift run report", ""]
    if fake:
        lines += [
            "> **FAKE RUN — fake components were used.** Every record below "
            "carries `fake: true`. This is not live training and cannot "
            "support any claim about real model quality.",
            "",
        ]
    lines += [
        f"- Run: `{summary.get('run_name', root.name)}`"
        + (" [fake]" if fake else ""),
        f"- Status: {summary.get('status', 'unknown')}",
        f"- Original pack: `{summary.get('original_pack_version', '')}`",
        f"- Final pack: `{summary.get('final_pack_version', '')}`",
        "- Spend: ${:.4f} committed+reserved of ${:.4f} ceiling{}".format(
            float(budgets.get("committed_and_reserved_usd", 0.0)),
            float(budgets.get("ceiling_usd", 0.0)),
            " [fake provider]" if fake else "",
        ),
        f"- Ledger: [{root.name}/ledger.jsonl](ledger.jsonl) "
        f"({len(events)} events, last seq {events[-1].get('seq', 0) if events else 0})",
        "",
        "## Outer rounds",
        "",
        "| Round | Candidate pack | Frozen | Accepted | Paired diff (screening) | Fake |",
        "|---|---|---|---|---|---|",
    ]
    rounds = summary.get("rounds") or []
    if not rounds:
        lines.append("| - | (none recorded) | - | - | - | - |")
    for entry in rounds:
        lines.append(
            "| {round} | `{candidate}` | {frozen} | {accepted} | {diff} | {fake} |".format(
                round=entry.get("round", "?"),
                candidate=str(entry.get("candidate_version", ""))[:16],
                frozen="yes" if entry.get("candidate_frozen") else "no",
                accepted="yes" if entry.get("accepted") else "no",
                diff=_fmt(entry.get("mean_paired_difference", "n/a")),
                fake=str(bool(entry.get("fake", fake))).lower(),
            )
        )
        if entry.get("reasons"):
            lines.append(f"| | decision: {'; '.join(str(r) for r in entry['reasons'])} | | | | |")
    lines += [
        "",
        "Screening numbers chose the winner, so they are an adaptive, biased "
        "estimate; only the final comparison estimates the change.",
        "",
        "## Final matched comparison",
        "",
    ]
    comparison = summary.get("comparison") or {}
    pairs = comparison.get("pairs") or []
    if pairs:
        lines += [
            "| Seed | Original | Final | Difference | Fake |",
            "|---|---|---|---|---|",
        ]
        for pair in pairs:
            original = pair.get("original") or {}
            final = pair.get("final") or {}
            lines.append(
                "| {seed} | {orig} | {fin} | {diff} | {fake} |".format(
                    seed=pair.get("seed", "?"),
                    orig=_fmt(original.get("score", original.get("status", "failed"))),
                    fin=_fmt(final.get("score", final.get("status", "failed"))),
                    diff=_fmt(pair.get("difference", "n/a")),
                    fake=str(bool(pair.get("fake", fake))).lower(),
                )
            )
        ci = comparison.get("bootstrap_ci") or [None, None]
        lines += [
            "",
            f"- Pairs in denominator: {comparison.get('pair_denominator', len(pairs))} "
            "(failed trials stay in the denominator)",
            f"- Scored pairs: {comparison.get('scored_pairs', 0)}",
            f"- Mean difference: {_fmt(comparison.get('mean_difference', 'n/a'))}",
            f"- Bootstrap 95% CI: [{_fmt(ci[0])}, {_fmt(ci[1])}]",
            f"- Failures: original {comparison.get('original_failures', 0)}, "
            f"final {comparison.get('final_failures', 0)}",
            f"- Compute per arm (GPU-seconds): original "
            f"{_fmt(comparison.get('original_compute_seconds', 0))}, final "
            f"{_fmt(comparison.get('final_compute_seconds', 0))}",
        ]
    else:
        lines.append("No final comparison recorded.")
    evidence = bool(comparison.get("evidence_of_improvement")) and not fake
    lines += ["", "## Conclusion", ""]
    if fake:
        lines.append(
            "**No evidence of improvement** [fake]: this run used fake "
            "components, so it cannot support an improvement claim regardless "
            "of the numbers above."
        )
    elif evidence:
        lines.append(
            "**Evidence of improvement**: the configured criterion held on "
            "the untouched final split (see table above)."
        )
    else:
        lines.append(
            "**No evidence of improvement**: the configured criterion did "
            "not hold on the final comparison."
        )
    lines += [
        "",
        "## Evaluation evidence",
        "",
        "| Evaluation | Split | Status | Score | Inspect log (sha256) | Ledger seq | Fake |",
        "|---|---|---|---|---|---|---|",
    ]
    if not data["evaluations"]:
        lines.append("| (none) | - | - | - | - | - | - |")
    for record in data["evaluations"]:
        log_path = record.get("inspect_log_path") or ""
        log_link = f"[`{str(record.get('inspect_log_sha256') or '')[:16]}`]"
        try:
            rel = Path(log_path).relative_to(root) if log_path else None
        except ValueError:
            rel = None
        log_link += f"({rel.as_posix()})" if rel is not None else ""
        lines.append(
            "| {eid} | {split} | {status} | {score} | {log} | {seq} | {fake} |".format(
                eid=record.get("evaluation_id", "?"),
                split=record.get("split", "?"),
                status=record.get("status", "?"),
                score=_fmt(record.get("score", "—")),
                log=log_link,
                seq=data["seq_by_eval"].get(str(record.get("evaluation_id")), "-"),
                fake=str(bool(record.get("fake"))).lower(),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_report(run_dir: str | Path, summary: dict[str, Any] | None = None) -> Path:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    if summary is not None:
        (root / "run.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
    target = root / "report.md"
    target.write_text(render_report(root), encoding="utf-8")
    return target
