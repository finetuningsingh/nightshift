"""API-backed agents: injected-transport tests, no network, no real keys."""

import json
import types

import pytest

from nightshift.agents import context_for, make_researcher, make_reviser
from nightshift.api_agents import (
    ApiAgentError,
    ApiChatClient,
    ApiEndpointConfig,
    ApiResearcher,
    ApiReviser,
    clients_from_config,
)
from nightshift.pack import check_pack_policy, create_pack, make_scratch_copy
from nightshift.workspaces import create_trial_workspace, load_outbox

TEST_KEY_ENV = "NIGHTSHIFT_TEST_API_KEY"


def _client(monkeypatch, reply, model="meta-models/muse-spark-open-1026", served=None):
    monkeypatch.setenv(TEST_KEY_ENV, "test-key-value-not-a-real-credential")
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append({"url": url, "headers": headers, "payload": payload})
        return {
            "model": served if served is not None else model,
            "choices": [{"message": {"content": reply}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }

    endpoint = ApiEndpointConfig(
        base_url="https://inference.baseten.co/v1",
        model=model,
        api_key_env=TEST_KEY_ENV,
    )
    return ApiChatClient(endpoint, transport=transport), calls


def test_api_client_records_requested_and_served_models(monkeypatch):
    client, calls = _client(monkeypatch, "hello")
    result = client.complete("ping")
    assert result["requested_model"] == "meta-models/muse-spark-open-1026"
    assert result["served_model"] == "meta-models/muse-spark-open-1026"
    assert result["usage"] == {"input_tokens": 11, "output_tokens": 7, "calls": 1}
    assert calls[0]["url"] == "https://inference.baseten.co/v1/chat/completions"
    assert calls[0]["payload"]["model"] == "meta-models/muse-spark-open-1026"
    # The credential goes in the header only, never in the result record.
    assert "test-key-value" not in json.dumps(result)


def test_api_client_refuses_served_model_substitution(monkeypatch):
    client, _calls = _client(monkeypatch, "hello", served="some-other-model")
    with pytest.raises(ApiAgentError, match="does not match"):
        client.complete("ping")


def test_api_client_fails_closed_without_credential(monkeypatch):
    monkeypatch.delenv(TEST_KEY_ENV, raising=False)
    called = []

    def transport(url, headers, payload, timeout):
        called.append(url)
        return {}

    endpoint = ApiEndpointConfig(
        base_url="https://api.meta.ai/v1",
        model="muse-wonder-1.0",
        api_key_env=TEST_KEY_ENV,
    )
    client = ApiChatClient(endpoint, transport=transport)
    with pytest.raises(ApiAgentError, match="no request was sent"):
        client.complete("ping")
    assert called == []


def _trial_context(tmp_path):
    pack = create_pack(tmp_path / "pack")
    trial = create_trial_workspace(
        tmp_path / "trials", "trial-api", ["researcher-1"],
        pack_version="test-pack", seed=1,
        initial_notes_dir=pack / "initial_notes",
    )
    rows = [{"id": "train-001", "prompt": "toy prompt", "checks": []}]
    return pack, context_for(trial, "researcher-1", pack, rows, [], "", 1)


def test_api_researcher_writes_validated_request(tmp_path, monkeypatch):
    reply = json.dumps({
        "type": "generate",
        "request_id": "researcher-1-generate-1",
        "prompt_ids": ["train-001"],
        "count": 2,
        "sampling_params": {"temperature": 0.7, "top_p": 1.0, "max_tokens": 96},
    })
    client, _calls = _client(monkeypatch, reply)
    _pack, context = _trial_context(tmp_path)
    agent = ApiResearcher("researcher-1", client)
    request = agent.act(context)
    assert request is not None and request.type == "generate"
    assert agent.last_run.validated_live is False
    assert agent.last_run.fake is False
    valid, errors = load_outbox(context.trial, "researcher-1")
    assert errors == [] and [r.type for r in valid] == ["generate"]


def test_api_researcher_done_and_bad_reply(tmp_path, monkeypatch):
    client, _calls = _client(monkeypatch, "DONE")
    _pack, context = _trial_context(tmp_path)
    agent = ApiResearcher("researcher-1", client)
    assert agent.act(context) is None and agent.done is True

    # Adversarial text is handed to the parser as data, never to a shell.
    bad_client, _calls = _client(monkeypatch, "not-json; rm -rf /")
    bad_agent = ApiResearcher("researcher-1", bad_client)
    with pytest.raises(ApiAgentError, match="not a single JSON"):
        bad_agent.act(context)


def test_api_reviser_applies_pack_edit_and_rejects_escape(tmp_path, monkeypatch):
    base = create_pack(tmp_path / "base")
    original = (base / "instructions.md").read_text(encoding="utf-8")
    edited = original.rstrip() + "\n\n## Reviser strategy, round 1\n\n- Generate two candidates per prompt, then filter hard.\n"
    reply = json.dumps({"files": {"instructions.md": edited}})
    client, _calls = _client(monkeypatch, reply, model="muse-wonder-1.0")
    scratch = make_scratch_copy(base, tmp_path / "scratch")
    result = ApiReviser(client).revise(scratch, public_input={}, round_no=1)
    assert result.fake is False
    assert result.served_model == "muse-wonder-1.0"
    assert "instructions.md" in check_pack_policy(scratch, base)

    escape = json.dumps({"files": {"../outside.md": "nope"}})
    bad_client, _calls = _client(monkeypatch, escape, model="muse-wonder-1.0")
    scratch2 = make_scratch_copy(base, tmp_path / "scratch2")
    with pytest.raises(ApiAgentError, match="escapes the pack"):
        ApiReviser(bad_client).revise(scratch2, public_input={}, round_no=1)
    assert not (tmp_path / "outside.md").exists()


def test_clients_from_config_supports_split_lane():
    config = types.SimpleNamespace(raw={"agents": {"backend": "api", "api": {
        "researcher_base_url": "https://inference.baseten.co/v1",
        "researcher_model": "meta-models/muse-spark-open-1026",
        "researcher_api_key_env": "BASETEN_API_KEY",
        "reviser_base_url": "https://api.meta.ai/v1",
        "reviser_model": "muse-wonder-1.0",
        "reviser_api_key_env": "META_API_KEY",
    }}})
    researcher, reviser = clients_from_config(config)
    assert researcher.endpoint.model == "meta-models/muse-spark-open-1026"
    assert researcher.endpoint.base_url == "https://inference.baseten.co/v1"
    assert reviser.endpoint.model == "muse-wonder-1.0"
    assert reviser.endpoint.base_url == "https://api.meta.ai/v1"


def test_factories_select_api_backend_without_network():
    config = types.SimpleNamespace(
        fake=False,
        models=types.SimpleNamespace(
            researcher="muse-spark-1.3", reviser="muse-wonder-full-1.0"
        ),
        raw={"agents": {"backend": "api", "api": {
            "base_url": "https://inference.baseten.co/v1",
            "model": "meta-models/muse-spark-open-1026",
            "api_key_env": "BASETEN_API_KEY",
        }}},
    )
    assert isinstance(make_researcher(config, "researcher-1", True), ApiResearcher)
    assert isinstance(make_reviser(config), ApiReviser)

    broken = types.SimpleNamespace(
        fake=False,
        models=config.models,
        raw={"agents": {"backend": "carrier-pigeon"}},
    )
    with pytest.raises(Exception, match="unknown agents.backend"):
        make_reviser(broken)
