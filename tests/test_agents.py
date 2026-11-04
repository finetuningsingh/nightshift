"""Agents: fake researcher/reviser behaviour and Muse adapter safety."""

import os
import stat

import pytest

from nightshift.agents import (
    AgentError,
    AgentRunResult,
    MuseCodeAdapter,
    FakeReviser,
    parse_muse_jsonl,
)
from nightshift.errors import PolicyError
from nightshift.pack import check_pack_policy, create_pack, make_scratch_copy


def test_fake_reviser_makes_policy_valid_edit(tmp_path):
    base = create_pack(tmp_path / "base")
    scratch = make_scratch_copy(base, tmp_path / "scratch")
    result = FakeReviser().revise(
        scratch,
        public_input={"trials": [{"dev_feedback": [{"split": "dev", "score": 0.5}]}]},
        round_no=1,
    )
    assert result.fake is True
    assert result.validated_live is False
    assert result.to_record()["validated_live"] is False
    assert AgentRunResult().validated_live is False
    changed = check_pack_policy(scratch, base)
    assert "instructions.md" in changed
    assert "two candidates per prompt" in (scratch / "instructions.md").read_text().lower()


def test_fake_reviser_can_produce_a_policy_violating_candidate(tmp_path):
    base = create_pack(tmp_path / "base")
    scratch = make_scratch_copy(base, tmp_path / "scratch")
    FakeReviser(violating=True).revise(scratch)
    with pytest.raises(PolicyError):
        check_pack_policy(scratch, base)


def test_b1_muse_adapter_refuses_without_explicit_unsandboxed_opt_in(tmp_path):
    adapter = MuseCodeAdapter(model="muse-spark-1.3", executable="muse-missing-xyz-123")
    with pytest.raises(AgentError, match="allow_unsandboxed_cli"):
        adapter.run_prompt("hello", tmp_path)


def test_muse_adapter_fails_clearly_when_muse_is_missing(tmp_path):
    adapter = MuseCodeAdapter(
        model="muse-spark-1.3", executable="muse-missing-xyz-123",
        allow_unsandboxed_cli=True,
    )
    with pytest.raises(AgentError, match="not found"):
        adapter.run_prompt("hello", tmp_path)


def test_parse_muse_jsonl_records_requested_and_served_models():
    stdout = "\n".join([
        '{"payload": {"kind": "run_model_configured", "model_id": "muse-spark-1.3"}}',
        '{"payload": {"kind": "run_usage", "models": [{"model": "muse-spark-1.3", "input_tokens": 10, "output_tokens": 3, "calls": 1}]}}',
        '{"payload": {"kind": "run_terminal", "terminal": "completed", "text": "done"}}',
    ])
    result = parse_muse_jsonl(stdout, requested_model="fallback")
    assert result.requested_model == "muse-spark-1.3"
    assert result.served_model == "muse-spark-1.3"
    assert result.usage == {"input_tokens": 10, "output_tokens": 3, "calls": 1}
    assert result.text == "done"


def test_muse_adapter_args_and_environment_are_safe(tmp_path, monkeypatch):
    script = tmp_path / "fake-muse"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\" > args.txt\n"
        "if [ -n \"$META_API_KEY\" ] || [ -n \"$MUSE_API_KEY\" ]; then echo leaked > env.txt; "
        "else echo clean > env.txt; fi\n"
        "printf '%s\\n' "
        "'{\"payload\": {\"kind\": \"run_model_configured\", \"model_id\": \"muse-wonder-full-1.0\"}}' "
        "'{\"payload\": {\"kind\": \"run_usage\", \"models\": [{\"model\": \"muse-wonder-full-1.0\", \"input_tokens\": 1, \"output_tokens\": 1, \"calls\": 1}]}}' "
        "'{\"payload\": {\"kind\": \"run_terminal\", \"terminal\": \"completed\", \"text\": \"ok\"}}'\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("META_API_KEY", "should-not-leak")
    monkeypatch.setenv("MUSE_API_KEY", "should-not-leak")
    adapter = MuseCodeAdapter(
        model="muse-wonder-full-1.0", executable=str(script), max_model_steps=7,
        allow_unsandboxed_cli=True,
    )
    result = adapter.run_prompt("do the thing", workspace)
    assert result.terminal == "completed"
    args = (workspace / "args.txt").read_text(encoding="utf-8").splitlines()
    assert args[0] == "exec" and "--json" in args
    assert args[args.index("--model") + 1] == "muse-wonder-full-1.0"
    assert args[args.index("--approval-mode") + 1] == "never"
    assert "--no-session-log" in args and "--disable-web-tools" in args
    assert "--yolo" not in args and "--disable-sandbox" not in args
    assert (workspace / "env.txt").read_text(encoding="utf-8").strip() == "clean"
    assert result.validated_live is False
    assert os.environ["META_API_KEY"] == "should-not-leak"  # parent env untouched
