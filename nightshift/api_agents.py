"""OpenAI-compatible API researcher and reviser backends.

This module lets a run use hosted models as the actual researcher and
reviser agents through any OpenAI-compatible ``/chat/completions``
endpoint, for example the Muse Model API (``https://api.meta.ai/v1``,
model ``muse-wonder-1.0``) for the reviser and Baseten's Muse Spark Open
endpoint (``https://inference.baseten.co/v1``, model
``meta-models/muse-spark-open-1026``) for researchers. The backend is
operator configuration in ``[agents]`` / ``[agents.api]``; see
``docs/configuration.md``.

Boundaries:

- API output is reasoning and code only. A researcher reply becomes one
  strictly validated controller request file; a reviser reply becomes pack
  file edits inside the scratch pack. API text never becomes a supervised
  fine-tuning target -- real targets are always verified local base-model
  generations on approved training prompts.
- Credentials are read from the environment variable named in config and
  are never written to disk, logs, or run records. A missing credential
  fails closed before any request is sent.
- Requested and served model ids are both recorded. If the endpoint
  reports a different served model than the one requested, the call fails
  instead of silently substituting.
- These adapters are NOT YET VALIDATED LIVE in this repository. Tests
  exercise them through an injected transport; no network is used there.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .errors import NightshiftError
from .hashing import canonical_json
from .pack import MAX_PACK_FILE_BYTES
from .workspaces import ResearcherRequest, validate_request, write_request

# A transport takes (url, headers, json_payload, timeout_seconds) and
# returns the parsed JSON response body. Tests inject a fake transport.
Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


class ApiAgentError(NightshiftError):
    """An API-backed agent call failed or was refused."""


_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _require_safe_base_url(base_url: str) -> None:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme == "https" and parsed.hostname:
        return
    if parsed.scheme == "http" and (parsed.hostname or "") in _LOOPBACK_HOSTS:
        return
    raise ApiAgentError(
        f"API base_url {base_url!r} must be https (plain http is "
        "allowed only for loopback development endpoints)"
    )


@dataclass(frozen=True)
class ApiEndpointConfig:
    base_url: str
    model: str
    api_key_env: str
    timeout_seconds: float = 300.0
    max_tokens: int = 4096
    temperature: float = 0.0
    max_calls: int | None = None

    def __post_init__(self) -> None:
        _require_safe_base_url(self.base_url)
        if self.max_calls is not None and int(self.max_calls) <= 0:
            raise ApiAgentError("max_calls must be a positive integer when set")

    def completions_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ApiAgentError(
                f"API agent refused: credential environment variable "
                f"{self.api_key_env!r} is empty (endpoint {self.base_url!r}, "
                f"model {self.model!r}); no request was sent"
            )
        return key


def _urllib_transport(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:
            detail = ""
        raise ApiAgentError(f"API request failed with HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ApiAgentError(f"API request failed: {exc}") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ApiAgentError("API response was not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ApiAgentError("API response JSON was not an object")
    return parsed


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped


class ApiChatClient:
    """Minimal chat-completions client with strict served-model checking."""

    def __init__(
        self, endpoint: ApiEndpointConfig, transport: Transport | None = None
    ):
        self.endpoint = endpoint
        self.transport = transport or _urllib_transport
        self._calls = 0

    def complete(self, prompt: str) -> dict[str, Any]:
        cfg = self.endpoint
        if cfg.max_calls is not None and self._calls >= int(cfg.max_calls):
            raise ApiAgentError(
                f"API call cap reached for endpoint {cfg.base_url!r} "
                f"model {cfg.model!r}: max_calls={int(cfg.max_calls)}; "
                "no request was sent"
            )
        self._calls += 1
        headers = {
            "Authorization": f"Bearer {cfg.api_key()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(cfg.max_tokens),
            "temperature": float(cfg.temperature),
        }
        response = self.transport(
            cfg.completions_url(), headers, payload, float(cfg.timeout_seconds)
        )
        served = str(response.get("model") or "")
        if not served:
            raise ApiAgentError(
                f"API response omitted the served model for requested "
                f"model {cfg.model!r}; refusing to invent it"
            )
        if served != cfg.model:
            raise ApiAgentError(
                f"served model {served!r} does not match requested model "
                f"{cfg.model!r}; refusing to substitute silently"
            )
        choices = response.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise ApiAgentError("API response contained no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ApiAgentError("API response contained no message text")
        usage_raw = response.get("usage") or {}
        usage = {
            "input_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage_raw.get("completion_tokens", 0) or 0),
            "calls": 1,
        }
        return {
            "text": content,
            "requested_model": cfg.model,
            "served_model": served,
            "base_url": cfg.base_url,
            "usage": usage,
        }


def _run_result(completion: dict[str, Any]) -> Any:
    from .agents import AgentRunResult

    return AgentRunResult(
        requested_model=str(completion["requested_model"]),
        served_model=str(completion["served_model"]),
        terminal="completed",
        text=str(completion["text"]),
        usage=dict(completion["usage"]),
        fake=False,
        validated_live=False,
    )


class ApiResearcher:
    """Hosted-model researcher. NOT YET VALIDATED LIVE.

    Each step asks the model for exactly one controller request as JSON
    (or DONE), validates it with the same strict validator used for every
    request file, and writes it to the researcher's outbox. The controller
    executes or rejects it; the model never runs code directly.
    """

    fake = False
    validated_live = False
    name = "api-researcher"

    def __init__(
        self,
        researcher_id: str,
        client: ApiChatClient,
        primary: bool = True,
    ):
        self.researcher_id = researcher_id
        self.client = client
        self.model = client.endpoint.model
        self.primary = bool(primary)
        self.done = False
        self.transcript: list[str] = []
        self.last_run: Any = None

    def act(self, context: Any) -> ResearcherRequest | None:
        if self.done:
            return None
        instructions = ""
        pack_instructions = Path(context.pack_dir) / "instructions.md"
        if pack_instructions.is_file():
            instructions = pack_instructions.read_text(encoding="utf-8")
        prompt = (
            "You are one researcher in a Nightshift trial. Follow these pack "
            "instructions:\n\n" + instructions + "\n\n"
            "Approved training prompts (the only prompt ids you may use):\n"
            + canonical_json(context.train_prompts) + "\n\n"
            "Shared notes from this trial so far:\n"
            + (context.shared_notes_text or "(none)")
            + "\n\nController history (your requests and inbox results):\n"
            + canonical_json(context.history) + "\n\n"
            "Reply with EXACTLY ONE of the following, and nothing else:\n"
            "1. A single JSON object for your next request, with fields "
            "{\"type\": ..., \"request_id\": ...} plus the fields for that "
            "type. Types: generate (prompt_ids, count, sampling_params), "
            "filter (sample_set_id, verifier_ids), train (sample_set_id, "
            "recipe_overrides), evaluate_dev (checkpoint_id), submit "
            "(checkpoint_id), note (text).\n"
            "2. The word DONE if you are finished.\n"
            "You have no access to held-out evaluation material; do not seek it. "
            "Your output is reasoning for the controller only; it is never "
            "training data."
        )
        completion = self.client.complete(prompt)
        self.last_run = _run_result(completion)
        text = _strip_code_fence(str(completion["text"]))
        self.transcript.append(f"step {context.step}: api reply received")
        if text.upper() == "DONE":
            self.done = True
            return None
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ApiAgentError(
                "researcher API reply was not a single JSON request object"
            ) from exc
        request = validate_request(raw, context.allowed_recipe_keys, self.researcher_id)
        write_request(context.trial, self.researcher_id, request)
        self.transcript.append(
            f"step {context.step}: {request.type} [{request.request_id}]"
        )
        if request.type == "submit":
            self.done = True
        return request


class ApiReviser:
    """Hosted-model outer reviser. NOT YET VALIDATED LIVE.

    The model returns whole-file pack edits as JSON. Edits are applied
    inside the scratch pack only; the controller's pack policy check still
    decides whether the candidate may freeze.
    """

    fake = False
    validated_live = False
    name = "api-reviser"

    def __init__(self, client: ApiChatClient):
        self.client = client
        self.model = client.endpoint.model

    def revise(
        self,
        scratch_pack_dir: str | Path,
        public_input: dict[str, Any] | None = None,
        round_no: int = 1,
    ) -> Any:
        pack = Path(scratch_pack_dir)
        prompt = (
            "You improve a Nightshift researcher pack (outer-loop revision: "
            "edit the researchers, not their artefacts). Edit only files "
            "inside the pack. Never edit pack controller fields or "
            "controller-owned recipe keys (base_model, datasets, "
            "output_dir, max_steps, seed), and never reference budgets, "
            "private held-out material, or scorer internals.\n\n"
            "Permitted public material from finished trials (pack files, "
            "transcripts, public dev feedback, shared notes):\n"
            + canonical_json(public_input or {})
            + "\n\nReply with EXACTLY ONE JSON object and nothing else:\n"
            "{\"files\": {\"relative/pack/path\": \"full new file text\"}}\n"
            "Include only files you change. Paths must be relative pack "
            "paths such as instructions.md, recipe.yaml, or "
            "initial_notes/note.md. Make one coherent improvement for the "
            "next round."
        )
        completion = self.client.complete(prompt)
        result = _run_result(completion)
        text = _strip_code_fence(str(completion["text"]))
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ApiAgentError("reviser API reply was not valid JSON") from exc
        files = parsed.get("files") if isinstance(parsed, dict) else None
        if not isinstance(files, dict) or not files:
            raise ApiAgentError(
                "reviser API reply must be an object with a non-empty "
                "'files' mapping of relative path to full file text"
            )
        root = pack.resolve()
        for rel_path, content in files.items():
            if not isinstance(rel_path, str) or not isinstance(content, str):
                raise ApiAgentError("reviser 'files' must map string paths to string text")
            rel = Path(rel_path)
            if rel.is_absolute() or ".." in rel.parts or not rel_path.strip():
                raise ApiAgentError(f"reviser edit escapes the pack: {rel_path!r}")
            if len(content.encode("utf-8")) > MAX_PACK_FILE_BYTES:
                raise ApiAgentError(f"reviser edit too large: {rel_path!r}")
            target = (root / rel).resolve()
            if target != root and root not in target.parents:
                raise ApiAgentError(f"reviser edit escapes the pack: {rel_path!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                raise ApiAgentError(f"reviser edit targets a symlink: {rel_path!r}")
            target.write_text(content, encoding="utf-8")
        result.text = (
            f"api reviser round {round_no}: edited {len(files)} file(s) via "
            f"{completion['base_url']} model {completion['served_model']}"
        )
        return result


def _endpoint_for(api_cfg: dict[str, Any], role: str) -> ApiEndpointConfig:
    def pick(*keys: str, default: Any = "") -> Any:
        for key in keys:
            value = api_cfg.get(key)
            if value not in (None, ""):
                return value
        return default

    base_url = str(pick(f"{role}_base_url", "base_url"))
    model = str(pick(f"{role}_model", "model"))
    api_key_env = str(pick(f"{role}_api_key_env", "api_key_env"))
    missing = [
        name
        for name, value in (
            ("base_url", base_url),
            ("model", model),
            ("api_key_env", api_key_env),
        )
        if not value
    ]
    if missing:
        raise ApiAgentError(
            f"[agents.api] is missing {role} field(s): {', '.join(missing)}; "
            f"set {role}_<field> or the shared <field> for each"
        )
    _require_safe_base_url(base_url)
    max_calls_raw = pick(f"{role}_max_calls", "max_calls", default=None)
    return ApiEndpointConfig(
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        timeout_seconds=float(pick("timeout_seconds", default=300.0)),
        max_tokens=int(pick("max_tokens", default=4096)),
        temperature=float(pick("temperature", default=0.0)),
        max_calls=int(max_calls_raw) if max_calls_raw is not None else None,
    )


def clients_from_config(config: Any) -> tuple[ApiChatClient, ApiChatClient]:
    """Build (researcher, reviser) clients from ``config.raw['agents']``."""
    raw_agents = (getattr(config, "raw", None) or {}).get("agents") or {}
    api_cfg = raw_agents.get("api") if isinstance(raw_agents, dict) else None
    if not isinstance(api_cfg, dict) or not api_cfg:
        raise ApiAgentError(
            "agents.backend = 'api' requires a non-empty [agents.api] table"
        )
    researcher_endpoint = _endpoint_for(api_cfg, "researcher")
    reviser_endpoint = _endpoint_for(api_cfg, "reviser")
    # Preflight: every configured credential must be present before any
    # trial spend, not discovered missing at first use.
    for endpoint in (researcher_endpoint, reviser_endpoint):
        if not os.environ.get(endpoint.api_key_env, ""):
            raise ApiAgentError(
                f"API agent preflight failed: credential environment "
                f"variable {endpoint.api_key_env!r} is empty (endpoint "
                f"{endpoint.base_url!r}, model {endpoint.model!r})"
            )
    return (
        ApiChatClient(researcher_endpoint),
        ApiChatClient(reviser_endpoint),
    )
