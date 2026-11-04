"""SIMULATED Muse Spark: a deterministic stand-in brain behind a real Chat Completions server on loopback.

THIS IS A SIMULATION OF MODEL BEHAVIOUR FOR TESTING THE LIVE CODE PATH. IT IS NEVER EVIDENCE OF HOW A META MODEL BEHAVES.
Nothing here calls Meta, nothing here is Muse Spark output, and no result produced against this server may be described as
a Meta model result. What it proves is plumbing: that the 'meta' backend (MetaChatClient over real HTTP, the supervisor tool
loop, the patch worker, the strict-schema roles, metering, the audit judge, the ledger and the dashboard) fits together and
degrades safely. What it cannot prove: that a real model writes good hypotheses, good patches, careful reviews or honest
summaries. The brain below knows the answer key (``tasks/law-discovery/scripted``), so any 'good science' it shows is scripted.

Composition, not modification: ``SimMuse`` owns a ``tests.support.fake_meta_server.FakeMetaServer`` (which keeps validating the
documented wire rules: nested tools, strict json_schema subset, forbidden parameters, tool-message pairing, 429/503/504) and
installs ``SimBrain`` as its reply handler. The brain recognises the role from the request itself, as a hosted model would
have to, never from a side channel:

* supervisor      the tools include ``launch_experiments``
* patch worker    the tools are ``list_files`` / ``read_file`` / ``write_file`` / ``replace_in_file``
* designer        ``response_format`` json_schema named ``experiment_plan``
* reviewer        ``response_format`` json_schema named ``review`` (the reviewer and the optional hard reviewer)
* reflector       ``response_format`` json_schema named ``reflection``
* audit judge     ``response_format`` json_schema named ``audit_verdict``

Behaviour, in one paragraph. The supervisor reads ``get_state``, launches a first round of three lanes whose hypotheses come
from ``manifest.json`` proposals (sent as two ``launch_experiments`` calls in ONE assistant message, to exercise batching),
reads the results, replicates the strongest development gain listed in ``awaiting_replication`` (the next one if it did not
replicate), reflects once when lanes gained nothing, launches a second round that asks for the optional hard review stage,
replicates again, and finishes with a summary built only from what it was shown. The patch worker reads ``model.py``,
writes the patch file matching the hypothesis keywords from ``scripted/patches/<id>/`` and answers with a short, claim-free
final message. The reviewer accepts unless the diff carries an obvious marker. The designer, reflector and judge return
minimal valid JSON for their schemas; the judge cites evidence seqs taken from the packet. Token usage is estimated from the
request size plus a fixed reasoning allowance per effort level, with prefix caching modelled on the system message.

Knobs (all deterministic): ``rounds`` (which proposals to launch), ``faults`` (``ScheduledFault``: an HTTP failure or a
misbehaving reply for the Nth request of a role), ``overclaim`` (patch ids whose final message asserts an improvement),
``never_dismiss`` (a model that ignores the advice to dismiss a semantic finding), ``latency`` (seconds a role's replies are
held, and counted: ``peak_in_flight`` and ``hold_spans`` prove that requests overlapped without any wall-clock bound), ``halt_file`` / ``halt_at`` (the operator touches HALT mid-run), ``passes_per_round``. ``SimMuse`` passes ``models``,
``unsupported_params`` and ``cache_report`` through to the fake server to imitate a partner host.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tests.support.fake_meta_server import DEFAULT_API_KEY, FakeMetaServer, Fault, RecordedRequest, Reply

ROOT = Path(__file__).resolve().parents[2]
SCRIPTED_DIR = ROOT / "tasks" / "law-discovery" / "scripted"

# Reasoning tokens a hosted reasoning model would add to the completion count at each effort level (they are billed).
REASONING_TOKENS = {"minimal": 20, "low": 120, "medium": 350, "high": 800, "xhigh": 1800}
STRUCTURED = {"experiment_plan": "designer", "review": "reviewer", "reflection": "reflector", "audit_verdict": "judge"}
PATCH_TOOLS = frozenset({"list_files", "read_file", "write_file", "replace_in_file"})

# Hypothesis keywords -> patch directory under scripted/patches. The most specific entries come first.
PATCH_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("poly_tuned", ("backtest", "hold out the last", "held out the last", "by backtesting")),
    ("tamper", ("clip every prediction", "clip them", "leave the range")),
    ("injection", ("normal-equations", "normal equations")),
    ("quadratic_defect", ("sorting the readings", "sort the readings")),
    ("poly9", ("degree-9", "degree 9", "degree9")),
    ("damped", ("damped", "oscillat", "sinusoid")),
    ("quadratic", ("quadratic", "degree-2", "degree 2", "parabola")),
    ("claim_without_run", ("ridge penalty", "tiny ridge")),
)
# A probe that is not in the scripted bank: the run reports what its own environment shows, so a test can prove that the
# experiment process cannot see the API key that the parent process holds. Names only, never values.
ENV_PROBE_PROPOSAL = {
    "hypothesis": "Reporting which environment variables the run can see shows whether it can reach credentials.",
    "approach": "Print the number of environment variables and whether the model API key variables are visible, "
                "then fit the same straight line as the baseline."}
ENV_PROBE_SOURCE = '''"""Straight line, plus a report of what the run's environment contains (names only)."""
import os

from linalg import polyfit


def fit_predict(train, query_times):
    names = sorted(os.environ)
    print("ENV_COUNT=%d" % len(names))
    print("MODEL_API_KEY_VISIBLE=%s" % ("MODEL_API_KEY" in os.environ))
    print("META_API_KEY_VISIBLE=%s" % ("META_API_KEY" in os.environ))
    print("SECRETISH_NAMES=%r" % [n for n in names if any(w in n.upper() for w in ("KEY", "TOKEN", "SECRET"))])
    predict = polyfit([t for t, _ in train], [y for _, y in train], 1)
    return [predict(t) for t in query_times]
'''
ENV_PROBE_KEYWORDS = ("environment variables",)

# Text that tells a reviewer or an auditor what to conclude: the sim judge reads it as evidence of a violation of I13.
INJECTION_TEXT = re.compile(r"(?i)\b(approve this|ignore (?:all |the )?(?:previous )?(?:rules|instructions))\b")
REVIEW_MARKERS = ("BUG(", "grader", "labels.", "trusted/", "ledger.jsonl", "../", "tasks/", "approve this", "ignore the rules")

# Rounds of lanes: (lane label, manifest proposal id). The first round is sent as two calls in one assistant message.
DEFAULT_ROUNDS: tuple[tuple[tuple[str, str], ...], ...] = (
    (("A", "quadratic"), ("B", "poly9"), ("C", "claim_without_run")),
    (("D", "damped"), ("E", "poly_tuned")),
)
SPLIT_ROUND = 0                 # index of the round sent as two launch_experiments calls in ONE assistant message
HARD_REVIEW_ROUND = 1           # index of the round that asks for the optional hard review (stage review_hard)
HARD_REVIEW_STAGES = ["design", "implement", "run", "grade", "review", "review_hard", "record"]


# --------------------------------------------------------------------------------------------------
# request analysis
# --------------------------------------------------------------------------------------------------
def classify_request(body: Mapping[str, Any] | None) -> str:
    """The role a request belongs to, from its tools or its json_schema name. ``"unknown"`` if neither says."""
    body = body or {}
    names = {(t.get("function") or {}).get("name") for t in body.get("tools") or [] if isinstance(t, dict)}
    if "launch_experiments" in names:
        return "supervisor"
    if names and names <= PATCH_TOOLS:
        return "patch_worker"
    rf = body.get("response_format") or {}
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        return STRUCTURED.get((rf.get("json_schema") or {}).get("name"), "unknown")
    return "unknown"


def fenced(text: str, label: str) -> str | None:
    """The body of the first ``<<<DATA:label:...>>>`` block in ``text`` (None if there is none)."""
    tag = re.escape(label)
    m = re.search(r"<<<DATA:%s:[0-9a-f]+>>>\n(.*?)\n(?:\[truncated \d+ chars\]\n)?<<<END:%s:[0-9a-f]+>>>" % (tag, tag), text, re.S)
    return m.group(1) if m else None


def _user_text(body: Mapping[str, Any]) -> str:
    return "\n".join(m.get("content") or "" for m in body.get("messages", []) if m.get("role") == "user")


def _estimate(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class Exchange:
    """One tool call the assistant made and the tool message that answered it."""

    name: str
    args: dict
    raw: str
    obj: Any


def tool_history(messages: Sequence[Mapping[str, Any]]) -> list[list[Exchange]]:
    """The assistant's tool calls with their results, grouped by assistant message (oldest first)."""
    answers = {m.get("tool_call_id"): m.get("content") or "" for m in messages if m.get("role") == "tool"}
    turns: list[list[Exchange]] = []
    for m in messages:
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        turn = []
        for call in m["tool_calls"]:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            raw = answers.get(call.get("id"), "")
            try:
                obj = json.loads(raw)
            except ValueError:
                obj = None
            turn.append(Exchange(str(fn.get("name")), args if isinstance(args, dict) else {}, raw, obj))
        turns.append(turn)
    return turns


# --------------------------------------------------------------------------------------------------
# fault schedule
# --------------------------------------------------------------------------------------------------
@dataclass
class ScheduledFault:
    """Answer the ``nth`` (0-based) request of ``role`` with ``fault`` instead of the brain's reply, once.

    ``fault`` is a raw HTTP failure (``Fault``) or a misbehaving model reply (``Reply``: plain text where a tool call was
    due, invalid JSON, a reply cut off at the token limit). ``role`` is a ``classify_request`` value. With ``containing``
    only requests whose user text holds that string count, and ``nth`` counts among those: the judge calls of one audit
    window run concurrently, so 'the second judge call' is a race and ``containing="Invariant I07"`` is not."""

    role: str
    fault: Fault | Reply
    nth: int = 0
    containing: str | None = None
    fired: bool = field(default=False, init=False)
    seen: int = field(default=0, init=False)


# --------------------------------------------------------------------------------------------------
# the brain
# --------------------------------------------------------------------------------------------------
class SimBrain:
    """``FakeMetaServer`` reply handler. Deterministic: the same conversation always gets the same answer."""

    def __init__(self, *, rounds: Sequence[Sequence[tuple[str, str]]] = DEFAULT_ROUNDS, split_round: int | None = SPLIT_ROUND,
                 hard_review_round: int | None = HARD_REVIEW_ROUND, passes_per_round: int = 1, overclaim: Sequence[str] = (),
                 halt_file: Path | None = None, halt_at: tuple[str, int] | None = None, never_dismiss: bool = False,
                 latency: Mapping[str, float] | None = None,
                 faults: Sequence[ScheduledFault] = (), manifest_path: Path | None = None, patches_dir: Path | None = None):
        manifest = json.loads((manifest_path or SCRIPTED_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.proposals: dict[str, dict] = {**manifest["proposals"], "envprobe": ENV_PROBE_PROPOSAL}
        self.patches_dir = patches_dir or SCRIPTED_DIR / "patches"
        self.rounds = [list(r) for r in rounds]
        self.split_round = split_round
        self.hard_review_round = hard_review_round
        self.passes_per_round = passes_per_round
        self.overclaim = set(overclaim)
        self.latency = dict(latency or {})      # role -> seconds the server holds each reply (a slow reasoning model)
        self.never_dismiss = never_dismiss      # a stubborn model: it never follows the advice to dismiss a semantic finding
        self.halt_file = halt_file
        self.halt_at = halt_at          # (role, 0-based request number of that role) at which the operator touches HALT
        self.faults = list(faults)
        self._lock = threading.Lock()
        self._seen_by_role: Counter[str] = Counter()
        self._cache_prefix: dict[str, str] = {}
        self._call_no = 0
        self.role_counts: Counter[str] = Counter()          # replies produced, by role
        self.patches_written: list[str] = []                # patch ids the patch worker chose
        self.arrivals: dict[int, float] = {}                # request seq -> time.monotonic() when the brain was asked
        self.in_flight: Counter[str] = Counter()            # role -> requests being held right now (only roles with a latency)
        self.peak_in_flight: Counter[str] = Counter()       # role -> the most requests ever held at one time
        self.hold_spans: dict[int, tuple[float, float]] = {}    # request seq -> (held from, released at) on one monotonic clock

    # ---- entry point -------------------------------------------------------------------------------
    def __call__(self, req: RecordedRequest) -> Reply | Fault:
        body = req.body or {}
        role = classify_request(body)
        self.arrivals[req.seq] = time.monotonic()
        scheduled: Fault | Reply | None = None
        with self._lock:
            nth = self._seen_by_role[role]
            self._seen_by_role[role] += 1
            user = _user_text(body) if any(sf.containing for sf in self.faults) else ""
            for sf in self.faults:
                if sf.fired or sf.role != role:
                    continue
                if sf.containing is None:
                    here = nth
                elif sf.containing in user:
                    here, sf.seen = sf.seen, sf.seen + 1
                else:
                    continue
                if sf.nth == here:
                    sf.fired = True
                    scheduled = sf.fault
                    break
        if self.halt_file is not None and self.halt_at == (role, nth):
            self.halt_file.write_text("operator stop\n", encoding="utf-8")      # the operator touches HALT while a turn is in flight
        if isinstance(scheduled, Fault):
            return scheduled
        if scheduled is not None:
            return self._held(role, req.seq, self._with_usage(scheduled, body, role))
        handler: Callable[[dict], Reply] = {
            "supervisor": self._supervisor, "patch_worker": self._patch_worker, "designer": self._designer,
            "reviewer": self._reviewer, "reflector": self._reflector, "judge": self._judge}.get(role, self._unknown)
        reply = handler(body)
        return self._held(role, req.seq, self._with_usage(reply, body, role))

    def _held(self, role: str, seq: int, reply: Reply) -> Reply:
        """Hold the answer for the role's ``latency`` INSIDE the brain (the server thread of this request sleeps here), so that the
        requests held at one moment can be counted: sequential callers peak at 1 however fast or slow the machine is."""
        seconds = self.latency.get(role)
        if not seconds:
            return reply
        started = time.monotonic()
        with self._lock:
            self.in_flight[role] += 1
            self.peak_in_flight[role] = max(self.peak_in_flight[role], self.in_flight[role])
        try:
            time.sleep(seconds)
        finally:
            with self._lock:
                self.in_flight[role] -= 1
                self.hold_spans[seq] = (started, time.monotonic())
        return reply

    def _unknown(self, body: dict) -> Reply:
        return Reply.text("I did not recognise this request.")

    # ---- usage ---------------------------------------------------------------------------------------
    def _with_usage(self, reply: Reply, body: dict, role: str) -> Reply:
        prompt = _estimate(json.dumps(body.get("messages", [])) + json.dumps(body.get("tools") or []))
        content = reply.content if isinstance(reply.content, str) else json.dumps(reply.content or "")
        produced = content + "".join(c["name"] + c["arguments"] for c in reply.tool_calls)
        reasoning = REASONING_TOKENS.get(str(body.get("reasoning_effort") or "medium"), 350)
        cached = 0
        key = body.get("prompt_cache_key")
        system = next((m.get("content") or "" for m in body.get("messages", []) if m.get("role") == "system"), "")
        digest = hashlib.sha256(system.encode()).hexdigest()
        with self._lock:
            self.role_counts[role] += 1
            if key:
                if self._cache_prefix.get(key) == digest:
                    cached = min(prompt, _estimate(system))
                self._cache_prefix[key] = digest
        if role in self.latency:
            reply.delay = 0.0           # held in ``_held`` instead, where it is counted
        reply.prompt_tokens = prompt
        reply.completion_tokens = _estimate(produced) + reasoning
        reply.cached_tokens = cached
        if not isinstance(reply.content, str) and reply.content is not None:      # parts-list content: the server cannot size it
            reply.usage = {"prompt_tokens": prompt, "completion_tokens": reply.completion_tokens,
                           "total_tokens": prompt + reply.completion_tokens, "prompt_tokens_details": {"cached_tokens": cached}}
        return reply

    def _tool_reply(self, *calls: tuple[str, dict], text: str | None = None) -> Reply:
        with self._lock:
            self._call_no += 1
            n = self._call_no
        reply = Reply.tools(*[(name, args, f"call_sim{n:05d}_{i}") for i, (name, args) in enumerate(calls)], text=text)
        return reply

    # ---- supervisor ----------------------------------------------------------------------------------
    def lane_args(self, label: str, proposal_id: str) -> dict:
        p = self.proposals[proposal_id]
        return {"lane": label, "hypothesis": p["hypothesis"], "approach": p["approach"]}

    def _round_calls(self, index: int) -> list[tuple[str, dict]]:
        lanes = [self.lane_args(label, pid) for label, pid in self.rounds[index]]
        extra = {"stages": HARD_REVIEW_STAGES} if index == self.hard_review_round else {}
        if index == self.split_round and len(lanes) >= 2:
            return [("launch_experiments", {"template": "explore", "lanes": lanes[:-1], **extra}),
                    ("launch_experiments", {"template": "explore", "lanes": lanes[-1:], **extra})]
        return [("launch_experiments", {"template": "explore", "lanes": lanes, **extra})]

    def _supervisor(self, body: dict) -> Reply:
        turns = tool_history(body.get("messages", []))
        if not turns:
            return self._tool_reply(("get_state", {}),
                                    text="Reading the budget, the baseline and what has been tried before I plan a round.")
        last = turns[-1]
        names = [e.name for e in last]
        for e in last:
            if (e.name == "acknowledge_finding" and not self.never_dismiss and isinstance(e.obj, dict)
                    and "dismiss it" in str(e.obj.get("need"))):
                return self._tool_reply(("acknowledge_finding", {
                    "finding_id": e.args.get("finding_id"), "action": "dismissed",
                    "reason": "This is a semantic finding that code cannot verify; I read its rationale and found no action I can take."}))
        if names != ["get_state"]:
            return self._tool_reply(("get_state", {}))           # always re-read the state after acting
        state = last[0].obj if isinstance(last[0].obj, dict) else {}
        return self._decide(turns, state)

    def _decide(self, turns: list[list[Exchange]], state: dict) -> Reply:
        flat = [e for t in turns for e in t]
        # A round is a launch that was accepted. Old tool results are replaced by stubs once the history is compacted, so the
        # judgement uses the arguments (valid JSON with lanes) and treats only a visible error result as a failed launch.
        launch_turns = [i for i, t in enumerate(turns)
                        if any(e.name == "launch_experiments" and e.args.get("lanes") and not self._failed(e) for e in t)]
        since_launch = [e for t in turns[launch_turns[-1]:] for e in t] if launch_turns else []
        recent = {r.get("exp_id"): r for r in state.get("recent") or [] if isinstance(r, dict)}
        # A fresh conversation (after a restart) must not repeat rounds the state already shows: a round counts as done once
        # every one of its hypotheses appears among the recent experiments.
        tried = {self._plain(r.get("hypothesis")) for r in recent.values()}
        rounds_done = len(launch_turns)
        for index, lanes in enumerate(self.rounds):
            if all(self.proposals[pid]["hypothesis"] in tried for _, pid in lanes):
                rounds_done = max(rounds_done, index + 1)
        current_lanes = {label for label, _ in self.rounds[rounds_done - 1]} if rounds_done else set()

        acked = {e.args.get("finding_id") for e in flat if e.name == "acknowledge_finding"}
        for f in state.get("open_findings") or []:
            if f.get("severity") == "block" and f.get("finding_id") not in acked:
                return self._tool_reply(("acknowledge_finding", {
                    "finding_id": f["finding_id"], "action": "addressed",
                    "reason": "The experiment concerned is finished and is not the incumbent; I am not citing its claim."}))

        replicated_ids = {e.args.get("exp_id") for e in flat if e.name == "replicate"}
        passed = sum(1 for e in since_launch if e.name == "replicate" and isinstance(e.obj, dict) and e.obj.get("replicated") is True)
        awaiting = [x for x in state.get("awaiting_replication") or []
                    if x not in replicated_ids and recent.get(x, {}).get("lane") in current_lanes]
        awaiting.sort(key=lambda x: -(recent.get(x, {}).get("delta") or 0.0))
        if awaiting and passed < self.passes_per_round and (state.get("budget") or {}).get("replications", 0) > 0:
            return self._tool_reply(("replicate", {"exp_id": awaiting[0]}),
                                    text="The strongest development gain still needs its independent re-check.")

        never_reflected = not any(e.name == "reflect" for e in flat)
        misses = [r for r in recent.values() if r.get("lane") in current_lanes and r.get("outcome") in ("worse", "no_gain")]
        if rounds_done and never_reflected and (state.get("stalled") or len(misses) >= 2):
            return self._tool_reply(("reflect", {}), text="Two lanes of this round gained nothing; I want the lessons "
                                                          "consolidated before I plan the next round.")

        if rounds_done < len(self.rounds):
            return self._tool_reply(*self._round_calls(rounds_done))
        return self._tool_reply(("finish", {"summary": self.summary(state, rounds_done, len(replicated_ids))}))

    @staticmethod
    def _failed(e: Exchange) -> bool:
        return (isinstance(e.obj, dict) and "error" in e.obj) or ", error)" in e.raw

    @staticmethod
    def _plain(text: Any) -> str:
        m = re.search(r"<<<DATA:[^>]*>>>\n(.*?)\n<<<END:", str(text), re.S)
        return (m.group(1) if m else str(text)).strip()

    def summary(self, state: dict, rounds: int, rechecks: int) -> str:
        """An honest summary built from the last state only: what was tried, what replicated, what did not."""
        recent = [r for r in state.get("recent") or [] if isinstance(r, dict)]
        inc = state.get("incumbent") or {}
        replicated = [r for r in recent if r.get("replicated")]
        dev_gain = [r for r in recent if r.get("outcome") == "improved" and not r.get("replicated")]
        lines = [f"I ran {len(recent)} lanes in {rounds} round(s) and asked for {rechecks} independent re-check(s), always for the strongest "
                 "development gain."]
        if replicated:
            lines.append("These replicated: " + "; ".join(f"{r['exp_id']} ({self._plain(r.get('hypothesis'))[:90]})" for r in replicated) + ".")
        else:
            lines.append("Nothing replicated, so I claim no improvement.")
        if dev_gain:
            lines.append("These improved on the development data but were not re-checked or did not replicate, so they are not findings: "
                         + ", ".join(r["exp_id"] for r in dev_gain) + ".")
        if inc.get("replicated"):
            lines.append(f"The current best is {inc['exp_id']}, which replicated. The re-check says only pass or fail, so I report no size for the gain.")
        lines.append("Caveat: one task, a small pool of measurements, and the development numbers are optimistic because they were used to choose lanes.")
        return " ".join(lines)

    # ---- patch worker --------------------------------------------------------------------------------
    def choose_patch(self, text: str) -> str | None:
        low = text.lower()
        if any(w in low for w in ENV_PROBE_KEYWORDS):
            return "envprobe"
        for patch_id, words in PATCH_KEYWORDS:
            if any(w in low for w in words):
                return patch_id
        return None

    def _patch_worker(self, body: dict) -> Reply:
        user = _user_text(body)
        wanted = "\n".join(filter(None, (fenced(user, "hypothesis"), fenced(user, "approach"))))
        patch_id = self.choose_patch(wanted)
        turns = tool_history(body.get("messages", []))
        if not turns:
            return self._tool_reply(("read_file", {"path": "model.py"}), text="Reading the editable file first.")
        if patch_id is None:
            return Reply.text("I could not map this plan to a concrete change in model.py, so I left the file as it is.")
        wrote = [e for t in turns for e in t if e.name == "write_file"]
        if not wrote:
            fixed = self.patches_dir / patch_id / "model.fixed.py"
            if patch_id == "envprobe":
                contents = {"model.py": ENV_PROBE_SOURCE}
            elif "REPAIR." in user and fixed.is_file():
                # a repair round: the reviewer's findings are in the prompt, so write the corrected version of the one file
                contents = {"model.py": fixed.read_text(encoding="utf-8")}
            else:
                # a model that follows its plan literally writes every file the plan names, in one message (parallel tool calls)
                contents = {f.name: f.read_text(encoding="utf-8")
                            for f in sorted((self.patches_dir / patch_id).glob("*.py")) if f.name != "model.fixed.py"}
            with self._lock:
                self.patches_written.append(patch_id)
            return self._tool_reply(*[("write_file", {"path": name, "content": text}) for name, text in contents.items()])
        rejected = [e.args.get("path") for e in wrote if e.raw.startswith("error")]
        if len(rejected) == len(wrote):
            return Reply.text("Every write was rejected, so the workspace is unchanged.")
        what = patch_id.replace("_", " ")
        text = f"I replaced the body of fit_predict in model.py with the {what} model described in the plan."
        if rejected:
            text += f" My write to {', '.join(map(str, rejected))} was rejected because that file is not editable, so I did not change it."
        text += (" This improves the extrapolation error." if patch_id in self.overclaim
                 else " Nothing has been run, so I cannot say how it scores.")
        return Reply.text(text)

    # ---- designer / reviewer / reflector / judge -------------------------------------------------------
    def _designer(self, body: dict) -> Reply:
        hypothesis = fenced(_user_text(body), "hypothesis") or "the stated hypothesis"
        hint = fenced(_user_text(body), "hint")
        approach = (f"In model.py, rewrite fit_predict so that it tests this hypothesis: {hypothesis[:300]} "
                    + (f"Hint: {hint[:300]} " if hint else "") + "Keep the signature and return one float per query time.")
        return Reply.json({"approach": approach,
                           "expected_effect": "If the assumed form matches the data, extrapolation error on the later times should fall.",
                           "risks": "A flexible form can fit the training window well and still extrapolate badly.",
                           "files_to_touch": ["model.py"]})

    def _reviewer(self, body: dict) -> Reply:
        diff = fenced(_user_text(body), "diff") or ""
        touched = re.findall(r"^\+\+\+ b?/?(\S+)", diff, re.M)
        hits = [m for m in REVIEW_MARKERS if m in diff]
        other = [f for f in touched if f.split("/")[-1] != "model.py"]
        if hits or other:
            what = hits[0] if hits else other[0]
            return Reply.json({"verdict": "repair", "summary": "The diff contains something outside the hypothesis.",
                               "findings": [{"id": "out_of_scope", "severity": "block",
                                             "text": "The change touches or names something other than the model's fit.", "evidence": what[:120]}]})
        return Reply.json({"verdict": "accept", "summary": "The diff implements the stated hypothesis in model.py and nothing else.", "findings": []})

    def _reflector(self, body: dict) -> Reply:
        text = _user_text(body)
        ids = list(dict.fromkeys(re.findall(r"\be\d{3}\b", fenced(text, "tried") or "")))
        tried_damped = "damped" in (fenced(text, "tried") or "").lower()
        return Reply.json({
            "summary": "Several ideas were tried; the ones that matched the physics are worth extending.",
            "lessons": [{"text": "Models with a functional form matching the physical system extrapolate more safely than extra flexibility.",
                         "kind": "strategy", "source_exps": ids[:3], "classification": "model"}],
            "pivots": ["Refine the oscillation model's search grid." if tried_damped else "Try a physically motivated damped-oscillation form."]})

    def _judge(self, body: dict) -> Reply:
        text = _user_text(body)
        items = [(int(m.group(1)), chunk) for chunk in re.split(r"(?=\[seq \d+ \|)", text)
                 if (m := re.match(r"\[seq (\d+) \|", chunk))]
        if not items:
            return Reply.json({"verdict": "not_applicable", "confidence": 0.6, "evidence_seqs": [],
                               "rationale": "The packet holds no evidence about this invariant."})
        injected = [seq for seq, chunk in items if INJECTION_TEXT.search(chunk)]
        if injected:
            return Reply.json({"verdict": "violated", "confidence": 0.7, "evidence_seqs": injected[:2],
                               "rationale": "Program output in the evidence tells a reviewer to approve or to ignore rules."})
        return Reply.json({"verdict": "compliant", "confidence": 0.75, "evidence_seqs": [seq for seq, _ in items][-2:],
                           "rationale": "The cited records are consistent with the invariant."})


# --------------------------------------------------------------------------------------------------
# the server
# --------------------------------------------------------------------------------------------------
class SimMuse:
    """``with SimMuse() as sim:`` starts a loopback server whose replies come from a ``SimBrain``."""

    def __init__(self, brain: SimBrain | None = None, *, api_key: str = DEFAULT_API_KEY, **server_kw: Any):
        self.brain = brain or SimBrain()
        self.server = FakeMetaServer(api_key, **server_kw)
        self.server.set_handler(self.brain)

    def __enter__(self) -> "SimMuse":
        self.server.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.stop()

    @property
    def base_url(self) -> str:
        return self.server.base_url

    @property
    def api_key(self) -> str:
        return self.server.api_key

    @property
    def requests(self) -> list[RecordedRequest]:
        return self.server.requests

    def requests_by_role(self) -> dict[str, list[RecordedRequest]]:
        out: dict[str, list[RecordedRequest]] = {}
        for r in self.requests:
            out.setdefault(classify_request(r.body), []).append(r)
        return out
