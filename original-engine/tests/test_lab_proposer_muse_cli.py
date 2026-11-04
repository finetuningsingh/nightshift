"""Tests for nightshift.lab.proposer_muse_cli (blocker B3): the CLI-backed proposal source, against a FAKE ``muse`` executable.

The fake is a small python script written into a temp dir. What it does on each call is read from ``scenario.json`` beside it (the child
environment is an allowlist, so the scenario cannot travel in the environment), and what it saw (argv, cwd, environment NAMES, prompt
file, workspace listing) is appended to ``calls.jsonl`` beside it. Test inputs are data: nothing here is handed to a shell, and nothing
real is executed except the fake script this test wrote.

Run: PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_lab_proposer_muse_cli
"""

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nightshift.lab.contracts import BudgetExhausted, EditOp, Lever, PackRef, ProposalContext, ProposalInvalid
from nightshift.lab.proposer import ProposerUnavailable, UnparseableProposal
from nightshift.lab.proposer_muse_cli import (
    ENV_ALLOWLIST,
    RESTRICTIVE_FLAGS,
    MuseAllowanceExhausted,
    MuseAuthMissing,
    MuseCliError,
    MuseCliProposer,
    build_child_env,
    build_prompt,
    parse_events,
)
from tests.test_lab_proposer import good_reply

FAKE = r'''#!__PYTHON__
import json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
scn = json.load(open(os.path.join(here, "scenario.json")))
def opt(name):
    return argv[argv.index(name) + 1] if name in argv else None
if argv[:1] == ["usage"]:
    sys.stdout.write(scn.get("usage_text", "Window: 0% used\nWeekly: 0% used\n"))
    sys.exit(0)
ws = opt("--workspace")
pf = opt("--prompt-file")
calls = os.path.join(here, "calls.jsonl")
n = 0
if os.path.exists(calls):
    n = sum(1 for _ in open(calls))
step = scn["steps"][min(n, len(scn["steps"]) - 1)]
rec = {"argv": argv, "cwd": os.getcwd(), "env_names": sorted(os.environ), "env": dict(os.environ),
       "prompt": open(pf, encoding="utf-8").read() if pf else None, "ws_at_start": sorted(os.listdir(ws)) if ws else None,
       "prompt_in_ws": bool(pf and ws and os.path.realpath(pf).startswith(os.path.realpath(ws) + os.sep))}
with open(calls, "a") as f:
    f.write(json.dumps(rec) + "\n")
if step.get("write_file"):
    open(os.path.join(ws, step["write_file"]), "w").write(step.get("write_text", "x"))
if step.get("sleep"):
    time.sleep(step["sleep"])
if step.get("stderr"):
    sys.stderr.write(step["stderr"])
for line in step.get("stdout_lines", []):
    sys.stdout.write(line + "\n")
sys.exit(step.get("exit", 0))
'''


def ev(kind, payload_type, **fields):
    """One recorded-style event line (the envelope of `muse exec --json`)."""
    payload = {"kind": kind, "command_id": "c-1", "run_stream": {"kind": "run", "id": "c-1"}}
    payload.update(fields)
    return json.dumps({"schema_version": 1, "id": "i", "stream": {"kind": "session", "id": "s"}, "sequence": 1, "record_type": "event",
                       "durability": "durable", "payload_type": payload_type, "payload_schema_version": 1, "payload": payload})


def usage_line(inp=28938, cached=0, out=13, reasoning=3, window=0, weekly=0, model="muse-wonder-full-1.0"):
    return ev("run_usage", "run.usage", models=[{"model": model, "input_tokens": inp, "cached_tokens": cached, "output_tokens": out,
                                                 "reasoning_tokens": reasoning, "calls": 1}],
              calls=1, completed_calls=1, failed_calls=0, cancelled_calls=0, children_included=False,
              allowance={"tier": None, "window": {"used_percent": window, "resets_at_ms": 1793769033000},
                         "weekly": {"used_percent": weekly, "resets_at_ms": 1794182400000}, "as_of_ms": 1793751044660, "source": "stream"})


def ok_stream(text, **usage):
    return [ev("task_lifecycle", "task.lifecycle.started"), ev("run_output_delta", "run.output.delta", text=text), usage_line(**usage),
            ev("run_terminal", "run.terminal.completed", terminal="completed", text=text, reason=None)]


def failed_stream(reason, **usage):
    return [usage_line(**usage), ev("run_terminal", "run.terminal.failed", terminal="failed", text="", reason=reason)]


PLANTED = {"MUSE_API_KEY": "planted-muse-value-1", "BASETEN_API_KEY": "planted-baseten-value-2", "META_API_KEY": "planted-meta-value-3",
           "OTHER_API_KEY": "planted-other-value-4", "ANTHROPIC_AUTH_TOKEN": "planted-token-value-5", "AWS_SECRET_ACCESS_KEY": "planted-aws-value-6"}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "fake"
        self.dir.mkdir()
        self.scratch = Path(self.tmp.name) / "scratch"  # temp_parent: must be empty again after every call
        self.scratch.mkdir()
        self.exe = self.dir / "muse"
        self.exe.write_text(FAKE.replace("__PYTHON__", sys.executable))
        self.exe.chmod(self.exe.stat().st_mode | stat.S_IXUSR)
        self.ctx = ProposalContext("cand-001", PackRef("packs/pack.git", "b" * 40), "v0", "SYSTEM RULES text", "USER FACTS text")

    def scenario(self, steps, **extra):
        (self.dir / "scenario.json").write_text(json.dumps({"steps": steps, **extra}))

    def calls(self):
        p = self.dir / "calls.jsonl"
        return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []

    def proposer(self, **kw):
        kw.setdefault("base_env", {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/home/x", "LANG": "C"})
        return MuseCliProposer(muse_path=str(self.exe), temp_parent=str(self.scratch), **kw)

    def assertScratchEmpty(self):
        self.assertEqual(list(self.scratch.iterdir()), [])


class SuccessTests(Base):
    def test_success_parses_the_proposal_and_books_tokens_with_zero_dollars(self):
        reply = json.dumps(good_reply())
        self.scenario([{"stdout_lines": ok_stream(reply, inp=29000, cached=100, out=500, reasoning=40, window=3, weekly=7)}])
        p = self.proposer()
        res = p.propose(self.ctx)
        self.assertEqual(res.proposal.lever, Lever.CONTEXT_NOTES)
        self.assertEqual([e.op for e in res.proposal.edits], [EditOp.SEARCH_REPLACE, EditOp.REPLACE_FILE])
        self.assertEqual((res.model, res.served_model, res.fallback_used), ("muse-wonder-full-1.0", "muse-wonder-full-1.0", False))
        self.assertEqual((res.usd, res.prompt_tokens, res.completion_tokens), (0.0, 29000, 500))
        self.assertEqual(res.raw_response, reply)
        u = p.last_usage
        self.assertEqual((u["cached_tokens"], u["reasoning_tokens"], u["calls"]), (100, 40, 1))
        self.assertEqual((u["allowance_window_used_percent"], u["allowance_weekly_used_percent"]), (3, 7))
        self.assertEqual(len(self.calls()), 1)
        self.assertScratchEmpty()

    def test_command_line_is_restrictive_and_workspace_is_the_empty_cwd(self):
        self.scenario([{"stdout_lines": ok_stream(json.dumps(good_reply()))}])
        self.proposer(effort="low").propose(self.ctx)
        (c,) = self.calls()
        argv = c["argv"]
        self.assertEqual(argv[:2], ["exec", "--json"])
        self.assertEqual(argv[argv.index("--model") + 1], "muse-wonder-full-1.0")
        self.assertEqual(argv[argv.index("--effort") + 1], "low")
        ws = argv[argv.index("--workspace") + 1]
        self.assertEqual(os.path.realpath(ws), os.path.realpath(c["cwd"]))
        self.assertEqual(c["ws_at_start"], [])  # fresh and empty
        self.assertFalse(c["prompt_in_ws"])
        joined = " ".join(argv)
        for flag in ("--preset empty", "--disable-shell", "--disable-write", "--disable-web-tools", "--approval-mode never",
                     "--contributor-mode off", "--no-session-log", "--no-foreign-personal-context"):
            self.assertIn(flag, joined)
        for forbidden in ("--yolo", "--disable-sandbox", "--disable-approval", "--trust-workspace", "--api-key-stdin", "--base-url"):
            self.assertNotIn(forbidden, argv)
        self.assertEqual(list(RESTRICTIVE_FLAGS[:2]), ["--preset", "empty"])

    def test_default_effort_is_high(self):
        self.scenario([{"stdout_lines": ok_stream(json.dumps(good_reply()))}])
        self.proposer().propose(self.ctx)
        argv = self.calls()[0]["argv"]
        self.assertEqual(argv[argv.index("--effort") + 1], "high")

    def test_final_text_falls_back_to_output_deltas(self):
        reply = json.dumps(good_reply())
        lines = [ev("run_output_delta", "run.output.delta", text=reply[:20]), ev("run_output_delta", "run.output.delta", text=reply[20:]),
                 usage_line(), ev("run_terminal", "run.terminal.completed", terminal="completed", text="", reason=None)]
        self.scenario([{"stdout_lines": lines}])
        self.assertEqual(self.proposer().propose(self.ctx).raw_response, reply)

    def test_reply_in_a_code_fence_is_accepted_like_the_api_path(self):
        self.scenario([{"stdout_lines": ok_stream("```json\n" + json.dumps(good_reply()) + "\n```")}])
        self.assertEqual(self.proposer().propose(self.ctx).proposal.lever, Lever.CONTEXT_NOTES)

    def test_parse_events_skips_garbage_and_takes_the_last_usage(self):
        out = "\n".join(["muse: notice", "{not json", "[1]", json.dumps({"payload": 3}), usage_line(inp=1), usage_line(inp=2, window=9),
                         ev("run_terminal", "run.terminal.completed", terminal="completed", text="hi", reason=None)])
        text, terminal, reason, usage = parse_events(out)
        self.assertEqual((text, terminal, reason, usage["input_tokens"], usage["allowance_window_used_percent"]), ("hi", "completed", "", 2, 9))

    def test_a_file_the_agent_wrote_is_ignored(self):
        # a perfectly valid proposal file in the workspace, but the final text is not a proposal: only the text counts
        self.scenario([{"write_file": "PROPOSAL.json", "write_text": json.dumps(good_reply()), "stdout_lines": ok_stream("I wrote PROPOSAL.json")}] * 2)
        p = self.proposer()
        with self.assertRaises(UnparseableProposal):
            p.propose(self.ctx)
        self.assertEqual(p.last_run.workspace_files, ("PROPOSAL.json",))  # visible to the caller, never read
        self.assertScratchEmpty()


class InvalidReplyTests(Base):
    def test_valid_json_but_invalid_proposal_is_not_asked_again(self):
        bad = good_reply()
        bad["lever"] = "not_a_lever"
        self.scenario([{"stdout_lines": ok_stream(json.dumps(bad), inp=100, out=10)}])
        with self.assertRaises(UnparseableProposal) as cm:
            self.proposer().propose(self.ctx)
        self.assertIsInstance(cm.exception, ProposalInvalid)
        self.assertEqual((cm.exception.usd, cm.exception.prompt_tokens, cm.exception.completion_tokens), (0.0, 100, 10))
        self.assertEqual(cm.exception.raw_response, json.dumps(bad))
        self.assertEqual(len(self.calls()), 1)
        self.assertScratchEmpty()

    def test_extra_key_is_invalid(self):
        bad = good_reply(extra="x")
        self.scenario([{"stdout_lines": ok_stream(json.dumps(bad))}])
        with self.assertRaises(UnparseableProposal) as cm:
            self.proposer().propose(self.ctx)
        self.assertIn("unknown", cm.exception.reason)

    def test_no_json_object_is_retried_once_at_a_lower_effort_and_costs_add_up(self):
        self.scenario([{"stdout_lines": ok_stream("Sorry, I cannot.", inp=1000, out=20)},
                       {"stdout_lines": ok_stream(json.dumps(good_reply()), inp=1000, out=300)}])
        res = self.proposer(effort="high").propose(self.ctx)
        self.assertEqual((res.prompt_tokens, res.completion_tokens), (2000, 320))
        c1, c2 = self.calls()
        self.assertEqual(c1["argv"][c1["argv"].index("--effort") + 1], "high")
        self.assertEqual(c2["argv"][c2["argv"].index("--effort") + 1], "medium")
        self.assertEqual(c1["prompt"], c2["prompt"])

    def test_two_unusable_replies_raise_unparseable_with_both_costs(self):
        self.scenario([{"stdout_lines": ok_stream("no", inp=1000, out=20)}])
        with self.assertRaises(UnparseableProposal) as cm:
            self.proposer().propose(self.ctx)
        self.assertEqual((cm.exception.prompt_tokens, cm.exception.completion_tokens), (2000, 40))
        self.assertEqual(len(self.calls()), 2)
        self.assertScratchEmpty()

    def test_empty_reply_is_retried_then_unparseable(self):
        self.scenario([{"stdout_lines": ok_stream("")}])
        with self.assertRaises(UnparseableProposal):
            self.proposer().propose(self.ctx)
        self.assertEqual(len(self.calls()), 2)


class FailureTests(Base):
    def test_nonzero_exit_is_a_cli_error_and_unavailable(self):
        self.scenario([{"exit": 2, "stderr": "boom: something broke\n"}])
        with self.assertRaises(MuseCliError) as cm:
            self.proposer().propose(self.ctx)
        self.assertIsInstance(cm.exception, ProposerUnavailable)
        self.assertNotIsInstance(cm.exception, (MuseAuthMissing, BudgetExhausted))
        self.assertIn("exit 2", str(cm.exception))
        self.assertEqual(len(self.calls()), 1)  # a CLI failure is not retried
        self.assertScratchEmpty()

    def test_failed_terminal_with_exit_zero_is_still_a_failure(self):
        self.scenario([{"stdout_lines": failed_stream("provider returned 500")}])
        with self.assertRaises(MuseCliError):
            self.proposer().propose(self.ctx)

    def test_auth_missing(self):
        self.scenario([{"exit": 1, "stderr": "missing meta credentials: run `muse login` or set META_API_KEY, or save credentials at /x/auth.json\n"}])
        with self.assertRaises(MuseAuthMissing) as cm:
            self.proposer().propose(self.ctx)
        self.assertIsInstance(cm.exception, ProposerUnavailable)
        self.assertEqual(len(self.calls()), 1)

    def test_allowance_exhausted_text_is_a_budget_stop(self):
        self.scenario([{"exit": 1, "stdout_lines": failed_stream("You have reached your weekly usage limit. Allowance resets in 4d.")}])
        with self.assertRaises(MuseAllowanceExhausted) as cm:
            self.proposer().propose(self.ctx)
        self.assertIsInstance(cm.exception, BudgetExhausted)  # the outer loop already stops on this
        self.assertEqual(cm.exception.axis, "muse_allowance")
        self.assertNotIsInstance(cm.exception, ProposerUnavailable)

    def test_allowance_at_100_percent_on_a_failed_run(self):
        self.scenario([{"exit": 1, "stdout_lines": failed_stream("request rejected", weekly=100)}])
        with self.assertRaises(MuseAllowanceExhausted) as cm:
            self.proposer().propose(self.ctx)
        self.assertEqual(cm.exception.used, 100.0)

    def test_precheck_stops_before_any_model_call(self):
        self.scenario([{"stdout_lines": ok_stream(json.dumps(good_reply()))}], usage_text="Tier: Muse Code Max Usage\nWindow: 100% used, resets in 1h\nWeekly: 40% used\n")
        with self.assertRaises(MuseAllowanceExhausted):
            self.proposer(precheck_allowance=True).propose(self.ctx)
        self.assertEqual(self.calls(), [])
        self.scenario([{"stdout_lines": ok_stream(json.dumps(good_reply()))}], usage_text="Window: 0% used\nWeekly: 40% used\n")
        self.assertEqual(self.proposer(precheck_allowance=True).propose(self.ctx).proposal.lever, Lever.CONTEXT_NOTES)

    def test_timeout_kills_the_run(self):
        self.scenario([{"sleep": 30}])
        with self.assertRaises(MuseCliError) as cm:
            self.proposer(timeout_seconds=0.5).propose(self.ctx)
        self.assertIn("timed out", str(cm.exception))
        self.assertScratchEmpty()

    def test_missing_executable(self):
        p = MuseCliProposer(muse_path=str(self.dir / "does-not-exist"), temp_parent=str(self.scratch))
        with self.assertRaises(MuseCliError):
            p.propose(self.ctx)
        self.assertScratchEmpty()

    def test_error_message_never_carries_environment_values(self):
        self.scenario([{"exit": 1, "stderr": "failed\n"}])
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/home/x", **PLANTED}
        with self.assertRaises(MuseCliError) as cm:
            self.proposer(base_env=env).propose(self.ctx)
        for v in PLANTED.values():
            self.assertNotIn(v, str(cm.exception))


class BillingGuardTests(Base):
    def test_child_env_is_the_allowlist_only(self):
        env = build_child_env({**PLANTED, "PATH": "/p", "HOME": "/h", "LANG": "C", "TMPDIR": "/t", "EXTRA": "e"})
        self.assertEqual(set(env), set(ENV_ALLOWLIST))
        self.assertEqual(ENV_ALLOWLIST, ("PATH", "HOME", "LANG", "TMPDIR"))
        self.assertFalse([k for k in env if k.endswith("_API_KEY")])

    def test_child_env_defaults_a_path(self):
        self.assertEqual(build_child_env({"HOME": "/h"})["PATH"], "/usr/bin:/bin")

    def test_the_running_child_sees_no_api_key_even_when_the_process_has_them(self):
        self.scenario([{"stdout_lines": ok_stream(json.dumps(good_reply()))}])
        with mock.patch.dict(os.environ, PLANTED):
            p = MuseCliProposer(muse_path=str(self.exe), temp_parent=str(self.scratch))  # base_env defaults to os.environ
            p.propose(self.ctx)
        (c,) = self.calls()
        names = {n for n in c["env_names"] if n != "__CF_USER_TEXT_ENCODING"}  # macOS adds this one inside the python fake itself
        self.assertTrue(names <= set(ENV_ALLOWLIST), names)
        self.assertFalse([k for k in c["env_names"] if "API_KEY" in k or "TOKEN" in k or "SECRET" in k])
        self.assertFalse(set(PLANTED.values()) & set(c["env"].values()))
        self.assertNotIn("MUSE_API_KEY", c["env"])
        self.assertNotIn("BASETEN_API_KEY", c["env"])
        # TMPDIR points inside the private temporary root, which is gone afterwards
        self.assertFalse(Path(c["env"]["TMPDIR"]).exists())

    def test_the_precheck_child_env_is_also_clean(self):
        self.scenario([{"stdout_lines": ok_stream(json.dumps(good_reply()))}])
        # the usage subcommand is run through the same builder; the fake does not log it, so assert the builder is what the module uses
        with mock.patch("nightshift.lab.proposer_muse_cli.subprocess.run") as run:
            run.return_value = mock.Mock(stdout="Window: 0% used\n")
            self.proposer(base_env={**PLANTED, "PATH": "/p", "HOME": "/h"}, precheck_allowance=True)._precheck()
        self.assertEqual(set(run.call_args.kwargs["env"]), {"PATH", "HOME"})


class PromptTests(Base):
    def test_prompt_is_exactly_the_context_plus_schema_and_nothing_else(self):
        secret_dir = "/Users/someone/PRIVATE-LAB-ROOT"
        self.scenario([{"stdout_lines": ok_stream(json.dumps(good_reply()))}])
        with mock.patch.dict(os.environ, {**PLANTED, "PRIVATE_NOTE": secret_dir}):
            MuseCliProposer(muse_path=str(self.exe), temp_parent=str(self.scratch)).propose(self.ctx)
        (c,) = self.calls()
        self.assertEqual(c["prompt"], build_prompt(self.ctx, MuseCliProposer().schema))
        self.assertTrue(c["prompt"].startswith("SYSTEM RULES text\n\nUSER FACTS text\n\n"))
        self.assertIn('"additionalProperties": false', c["prompt"])  # the schema the API path sends out of band
        for leaked in [secret_dir, str(self.dir), str(self.scratch), *PLANTED.values(), "packs/pack.git", "b" * 40, "cand-001"]:
            self.assertNotIn(leaked, c["prompt"])

    def test_forbidden_markers_are_only_present_if_the_context_has_them(self):
        # the proposer adds nothing of its own that could carry selection or test data: the prompt minus the context is a fixed text
        ctx2 = ProposalContext("cand-002", self.ctx.incumbent, "v0", "S", "U")
        extra = build_prompt(ctx2, MuseCliProposer().schema)[len("S\n\nU\n\n"):]
        self.assertTrue(extra.startswith("OUTPUT FORMAT"))
        for word in ("selection", "held-out", "heldout", "test split", "score"):
            self.assertNotIn(word, extra.lower())

    def test_the_prompt_file_is_outside_the_workspace_and_removed(self):
        self.scenario([{"stdout_lines": ok_stream(json.dumps(good_reply()))}])
        self.proposer().propose(self.ctx)
        self.assertFalse(self.calls()[0]["prompt_in_ws"])
        self.assertScratchEmpty()


class CleanupTests(Base):
    def test_workspace_and_prompt_removed_after_success_failure_and_unparseable(self):
        for step in ({"stdout_lines": ok_stream(json.dumps(good_reply()))}, {"exit": 3}, {"stdout_lines": ok_stream("nope")}):
            self.scenario([step])
            (self.dir / "calls.jsonl").unlink(missing_ok=True)
            try:
                self.proposer().propose(self.ctx)
            except (MuseCliError, UnparseableProposal):
                pass
            self.assertScratchEmpty()

    def test_each_call_gets_a_fresh_workspace(self):
        self.scenario([{"write_file": "leftover.txt", "stdout_lines": ok_stream(json.dumps(good_reply()))}])
        p = self.proposer()
        p.propose(self.ctx)
        p.propose(self.ctx)
        self.assertEqual([c["ws_at_start"] for c in self.calls()], [[], []])
        self.assertEqual(p.last_run.workspace_files, ("leftover.txt",))


if __name__ == "__main__":
    unittest.main()
