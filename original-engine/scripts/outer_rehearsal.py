#!/usr/bin/env python3
"""REHEARSAL of one outer step with the bounded challenge (``nightshift.lab.outer_challenge``), against a real Muse Code login.

Not part of any comparison: nothing is written to a lab root, no ledger event, no pack commit, no GPU. Three subcommands, each meant for
its own isolated container:

    check-context  --lab ROOT --prompt candidates/cand-001/prompt.txt --out DIR [--approval APPROVALDIR]
        (no network, on the host that already holds the lab root) Splits the recorded proposer prompt of a real outer step into the
        ProposalContext the outer command built (system prompt + public user prompt), checks the system part is the repo's own
        ``outer_context.system_prompt``, runs ``assert_no_forbidden`` with ``forbidden_values`` of that lab root, and writes
        ``context.json`` (public text only) and ``context-check.json`` (counts only); with ``--approval`` it also approves that prompt.

    rehearse --context DIR/context.json --pack PACKDIR --out DIR --approval APPROVALDIR [--muse PATH]
        (network, Muse login) Drives ``MuseCliProposer`` (muse-wonder-full-1.0) and ``Challenger`` with a second ``MuseCliProposer`` as the
        critic exactly as ``outer`` does: proposal, checker, one critic call, at most one revision, checker again. Writes the proposal, the
        challenge report and artefacts and ``summary.json`` (models, findings, verdict, usage) into DIR only. The full denylist needs the lab
        host's private data, so EVERY prompt is sent only after the lab host approved its exact text (``LabHostApproval``): an unapproved
        prompt is written to ``APPROVALDIR/pending/`` and the run stops with exit 3 before sending it; replies are cached, so the rerun after
        the approval continues without calling Muse again.

    check-prompts --lab ROOT (--approve APPROVALDIR | --dir DIR)
        (no network, on the lab host) ``--approve``: the full denylist check of every pending prompt, BEFORE it is sent (approved/ or
        denied/ by sha256). ``--dir``: the same check afterwards over every text file the rehearsal wrote. Prints counts only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nightshift.lab.contracts import AccessDenied, LabLayout, PackRef, ProposalContext, loads_strict  # noqa: E402
from nightshift.lab.outer_context import assert_no_forbidden, forbidden_values, system_prompt  # noqa: E402
from nightshift.lab.packs import DEFAULT_EDITABLE_PATHS  # noqa: E402

WONDER = "muse-wonder-full-1.0"
SPLIT = "\n\n## Incumbent pack: version "


def _forbidden(lab: Path, real_root: str = "") -> frozenset:
    """``forbidden_values`` of the lab root at ``lab``. ``real_root`` is the lab root's path on its own host (the copy here is mounted
    elsewhere, and the root path is one of the denied values)."""
    from nightshift.lab.fold import FoldReader

    layout = LabLayout(lab)
    cfg = json.loads((lab / "lab.json").read_text(encoding="utf-8"))
    hosts = tuple(h for h in (cfg.get("control_host"), cfg.get("researcher_host")) if isinstance(h, str))
    out = forbidden_values(FoldReader(layout).state(), layout, hosts)
    if real_root:  # deny the real root path instead of the mount point of this copy
        out = (out - {str(layout.root), str(layout.private_dir)}) | {real_root, real_root.rstrip("/") + "/private"}
    return frozenset(out)


def check_context(a: argparse.Namespace) -> int:
    lab, out = Path(a.lab), Path(a.out)
    text = (lab / a.prompt).read_text(encoding="utf-8")
    cut = text.index(SPLIT)
    system, user = text[:cut], text[cut + 2:]
    if system != system_prompt(DEFAULT_EDITABLE_PATHS):
        print("the recorded system prompt is not the repo's outer_context.system_prompt", file=sys.stderr)
        return 1
    forbidden = _forbidden(lab, a.real_root)
    assert_no_forbidden(system + "\n\n" + user, forbidden)
    head = user.split("\n", 1)[0]  # "## Incumbent pack: version v0 (commit 9a153aebc6)"
    version = head.split("version ", 1)[1].split(" ", 1)[0]
    short = head.split("(commit ", 1)[1].rstrip(")")
    cid = user.rsplit("(candidate id ", 1)[1].split(")", 1)[0]
    budget = loads_strict((lab / "frozen" / "budget.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    ctx = {"candidate_id": cid, "incumbent_version": version, "incumbent_commit_short": short, "system_prompt": system, "user_prompt": user,
           "gpu_minutes": float(budget["gpu_minutes"]), "source": a.prompt}
    (out / "context.json").write_text(json.dumps(ctx, indent=1) + "\n", encoding="utf-8")
    report = {"denylist_values": len(forbidden), "assert_no_forbidden": "passed", "prompt_chars": len(text), "system_prompt_matches_repo": True}
    (out / "context-check.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    if a.approval:
        (Path(a.approval) / "approved").mkdir(parents=True, exist_ok=True)
        (Path(a.approval) / "approved" / _sha(system + "\n\n" + user)).write_text("", encoding="utf-8")
    print(json.dumps(report))
    return 0


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AwaitingApproval(Exception):
    """A prompt must be checked against the full denylist on the lab host before it is sent; nothing was sent."""


class LabHostApproval:
    """The rehearsal's ``send_guard``: the full denylist needs the lab host's private data, which never comes here, so every prompt is sent
    only after ``check-prompts --approve`` on the lab host has approved its exact text (by sha256). Unknown text is written to
    ``pending/<sha>.txt`` and the run stops (``AwaitingApproval``) BEFORE the send; a denied text is refused like a denylist hit."""

    def __init__(self, root: Path):
        self.root = root
        for d in ("pending", "approved", "denied"):
            (root / d).mkdir(parents=True, exist_ok=True)

    def __call__(self, prompt: str) -> None:
        h = _sha(prompt)
        if (self.root / "approved" / h).exists():
            return
        if (self.root / "denied" / h).exists():
            raise AccessDenied(f"the lab host's denylist check refused this prompt (sha256 {h[:8]}); nothing was sent")
        (self.root / "pending" / f"{h}.txt").write_text(prompt, encoding="utf-8")
        raise AwaitingApproval(f"pending/{h}.txt")


class _Cached:
    """Forwards ``propose`` / ``critique`` to a ``MuseCliProposer`` once per exact prompt: replies are kept in ``cache/`` so a rerun after an
    approval does not call Muse again. Also keeps each call's Muse usage."""

    def __init__(self, inner, cache: Path, guard):
        self.inner, self.cache, self.guard, self.calls = inner, cache, guard, []
        self.model = inner.model
        cache.mkdir(parents=True, exist_ok=True)

    def _load(self, key: str):
        p = self.cache / f"{key}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def _store(self, key: str, rec: dict) -> None:
        (self.cache / f"{key}.json").write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")

    def _usage(self) -> dict:
        return {"usage": dict(self.inner.last_usage), "workspace_files": list(self.inner.last_run.workspace_files) if self.inner.last_run else []}

    def propose(self, ctx):
        from nightshift.lab.contracts import Proposal, ProposalResult

        text = ctx.system_prompt + "\n\n" + ctx.user_prompt
        self.guard(text)  # before the send: the lab host approved exactly this text
        key = "propose-" + _sha(text)
        rec = self._load(key)
        if rec is None:
            r = self.inner.propose(ctx)
            rec = {"model": r.model, "served_model": r.served_model, "fallback_used": r.fallback_used, "usd": r.usd, "prompt_tokens": r.prompt_tokens,
                   "completion_tokens": r.completion_tokens, "raw_response": r.raw_response, "proposal": r.proposal.to_dict(), **self._usage()}
            self._store(key, rec)
        self.calls.append({"usage": rec["usage"], "workspace_files": rec["workspace_files"]})
        return ProposalResult(Proposal.from_dict(rec["proposal"]), rec["model"], rec["served_model"], rec["fallback_used"],
                              rec["usd"], rec["prompt_tokens"], rec["completion_tokens"], rec["raw_response"])

    def critique(self, prompt: str):
        from nightshift.lab.outer_challenge import CriticReply

        key = "critique-" + _sha(prompt)  # the Challenger already ran the guard on this prompt
        rec = self._load(key)
        if rec is None:
            r = self.inner.critique(prompt)
            rec = {"text": r.text, "model": r.model, "served_model": r.served_model, "fallback_used": r.fallback_used, **self._usage()}
            self._store(key, rec)
        self.calls.append({"usage": rec["usage"], "workspace_files": rec["workspace_files"]})
        return CriticReply(rec["text"], rec["model"], rec["served_model"], rec["fallback_used"])


def rehearse(a: argparse.Namespace) -> int:
    from nightshift.lab.outer_challenge import Challenger, blocking
    from nightshift.lab.proposer_muse_cli import MuseCliProposer

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c = json.loads(Path(a.context).read_text(encoding="utf-8"))
    pack = Path(a.pack)
    parent = {p.relative_to(pack).as_posix(): p.read_text(encoding="utf-8") for p in sorted(pack.rglob("*")) if p.is_file()}
    # the PackRef is a placeholder for the record: the rehearsal commits nothing (the prompt names the real short commit)
    ctx = ProposalContext(c["candidate_id"], PackRef("packs/pack.git", (c["incumbent_commit_short"] + "0" * 40)[:40]), c["incumbent_version"],
                          c["system_prompt"], c["user_prompt"])
    # Here only secrets and addresses can be checked; the FULL denylist is applied before every send by the lab host's approval (guard).
    public = frozenset()
    guard = LabHostApproval(Path(a.approval))
    proposer = _Cached(MuseCliProposer(muse_path=a.muse, model=WONDER, effort=a.effort, precheck_allowance=True), out / "cache", guard)
    critic = _Cached(MuseCliProposer(muse_path=a.muse, model=WONDER, effort=a.effort, precheck_allowance=True), out / "cache", guard)
    (out / "proposer_prompt.txt").write_text(ctx.system_prompt + "\n\n" + ctx.user_prompt, encoding="utf-8")
    try:
        result = proposer.propose(ctx)
        (out / "proposer_response.txt").write_text(result.raw_response, encoding="utf-8")
        (out / "proposal.json").write_text(json.dumps(result.proposal.to_dict(), indent=1) + "\n", encoding="utf-8")
        first_usage = proposer.calls[-1]
        ch = Challenger(critic, send_guard=guard).run(ctx=ctx, result=result, parent_files=parent, gpu_minutes=float(c["gpu_minutes"]), proposer=proposer,
                                                      forbidden=public)
    except AwaitingApproval as exc:
        print(json.dumps({"awaiting_approval": str(exc), "sent": "nothing for this prompt; run check-prompts --approve on the lab host, then rerun"}))
        return 3
    summary = {
        "label": "REHEARSAL - not part of any comparison; no lab ledger, no pack commit, no GPU",
        "proposer": {"requested_model": result.model, "served_model": result.served_model, "fallback_used": result.fallback_used,
                     "lever": result.proposal.lever.value, "change_summary": result.proposal.change_summary,
                     "edited_paths": [e.path for e in result.proposal.edits], "usage": first_usage},
    }
    if ch is None:
        summary["challenge"] = "not run: the proposal does not apply to the incumbent (the outer loop would record it INVALID)"
    else:
        for name, text in ch.artefacts.items():
            (out / name).write_text(text, encoding="utf-8")
        (out / "challenge.json").write_text(json.dumps(ch.report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        (out / "candidate.diff").write_text(ch.diff, encoding="utf-8")
        if ch.revised:
            (out / "proposal_revised.json").write_text(json.dumps(ch.result.proposal.to_dict(), indent=1) + "\n", encoding="utf-8")
        summary["critic"] = {**ch.report.get("critic", {}), "usage": critic.calls[-1] if critic.calls else None}
        summary["revision"] = {**ch.report["revision"], "usage": proposer.calls[1] if len(proposer.calls) > 1 else None}
        if ch.revised:
            summary["revision"].update(lever=ch.result.proposal.lever.value, change_summary=ch.result.proposal.change_summary)
        summary["initial_findings"] = ch.report["initial_findings"]
        summary["final_findings"] = ch.report["final_findings"]
        summary["blocking"] = [f.code for f in blocking(ch.findings)]
        summary["outcome"] = "REFUSED (would be recorded INVALID, never run)" if ch.refused else "PASSES the challenge (would go on to lint and commit)"
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: summary.get(k) for k in ("outcome", "blocking")}))
    return 0


def check_prompts(a: argparse.Namespace) -> int:
    forbidden = _forbidden(Path(a.lab), a.real_root)
    if a.approve:  # decide every pending prompt BEFORE it is sent: approved/<sha> or denied/<sha>; prints counts only
        root = Path(a.approve)
        done = {"approved": 0, "denied": 0}
        for p in sorted((root / "pending").glob("*.txt")):
            h = p.stem
            text = p.read_text(encoding="utf-8")
            if _sha(text) != h:
                continue
            try:
                assert_no_forbidden(text, forbidden)
                kind = "approved"
            except AccessDenied:
                kind = "denied"
            (root / kind).mkdir(parents=True, exist_ok=True)
            (root / kind / h).write_text("", encoding="utf-8")
            done[kind] += 1
        print(json.dumps({"denylist_values": len(forbidden), **done}))
        return 0
    d = Path(a.dir)
    checked, bad = [], []
    for p in sorted(d.glob("*.txt")) + sorted(d.glob("*.json")) + sorted(d.glob("*.diff")):
        try:
            assert_no_forbidden(p.read_text(encoding="utf-8"), forbidden)
            checked.append(p.name)
        except AccessDenied as exc:
            bad.append(f"{p.name}: {str(exc)[:120]}")
    report = {"denylist_values": len(forbidden), "checked": checked, "refused": bad}
    print(json.dumps(report))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check-context")
    s.add_argument("--lab", required=True)
    s.add_argument("--prompt", default="candidates/cand-001/prompt.txt")
    s.add_argument("--out", required=True)
    s.add_argument("--real-root", default="", help="the lab root's path on its own host (also denied)")
    s.add_argument("--approval", default="", help="also approve the checked proposer prompt in this approval folder")
    s = sub.add_parser("rehearse")
    s.add_argument("--context", required=True)
    s.add_argument("--pack", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--muse", default="/opt/muse/muse")
    s.add_argument("--effort", default="high")
    s.add_argument("--approval", required=True, help="the approval folder (pending/, approved/, denied/) shared with the lab host's check")
    s = sub.add_parser("check-prompts")
    s.add_argument("--lab", required=True)
    s.add_argument("--dir", default="", help="post-hoc: check every text file the rehearsal wrote")
    s.add_argument("--approve", default="", help="before send: decide every pending prompt of this approval folder")
    s.add_argument("--real-root", default="")
    a = ap.parse_args(argv)
    return {"check-context": check_context, "rehearse": rehearse, "check-prompts": check_prompts}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
