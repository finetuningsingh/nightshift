"""Researcher and reviser agents: deterministic fakes and Muse adapters.

Collaboration follows the CORAL pattern (Human-Agent-Society/CORAL,
Apache-2.0, rev 0123dfb939b35228cf2c1fde224cd0561e727408): per-agent
isolated workspaces, shared notes inside one trial, heartbeat-style
reflect/consolidate/pivot notes. The outer method follows AIDE² (WeCo,
arXiv:2609.26457): the outer agent edits the researcher pack, never the
artefact directly. No upstream code is vendored.

The Muse adapters are NOT YET VALIDATED LIVE. They fail clearly when
the `muse` executable is missing, never read or pass API keys (subscription
login is used; key variables are stripped unless the operator explicitly
opts into API-key mode), and record requested and served model ids.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import NightshiftError
from .hashing import canonical_json, hash_object
from .pack import load_pack_metadata
from .workspaces import ResearcherRequest, TrialWorkspace, validate_request, write_request


class AgentError(NightshiftError):
    """An agent run failed or its runtime is unavailable."""


@dataclass
class AgentRunResult:
    requested_model: str = ""
    served_model: str = ""
    terminal: str = "completed"
    text: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    fake: bool = False
    validated_live: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "requested_model": self.requested_model,
            "served_model": self.served_model,
            "terminal": self.terminal,
            "text": self.text[:2000],
            "usage": dict(self.usage),
            "fake": bool(self.fake),
            "validated_live": bool(self.validated_live),
        }


@dataclass
class AgentContext:
    trial: TrialWorkspace
    researcher_id: str
    pack_dir: Path
    train_prompt_ids: list[str]
    allowed_recipe_keys: set[str] = field(default_factory=set)
    history: list[dict[str, Any]] = field(default_factory=list)
    train_prompts: list[dict[str, Any]] = field(default_factory=list)
    shared_notes_text: str = ""
    step: int = 0


def _history_successes(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [h for h in history if (h.get("result") or {}).get("status") == "succeeded"]


def _latest_result(history: list[dict[str, Any]], request_type: str) -> dict[str, Any] | None:
    for entry in reversed(history):
        if (entry.get("request") or {}).get("type") == request_type:
            if (entry.get("result") or {}).get("status") == "succeeded":
                return dict(entry["result"])
    return None


def _generation_count(pack_dir: Path) -> int:
    text = ""
    for name in ("instructions.md", "budget_habits.md"):
        path = Path(pack_dir) / name
        if path.is_file():
            text += path.read_text(encoding="utf-8").lower() + "\n"
    if "three candidates per prompt" in text:
        return 3
    if "two candidates per prompt" in text:
        return 2
    return 1


class FakeResearcher:
    """Deterministic scripted researcher. Every artefact it drives is fake."""

    fake = True
    name = "fake-researcher"

    def __init__(self, researcher_id: str, primary: bool = True, model: str = "muse-spark-1.3"):
        self.researcher_id = researcher_id
        self.primary = bool(primary)
        self.model = model
        self.done = False
        self.transcript: list[str] = []

    def _emit(self, context: AgentContext, raw: dict[str, Any]) -> ResearcherRequest:
        request = validate_request(raw, context.allowed_recipe_keys, self.researcher_id)
        write_request(context.trial, self.researcher_id, request)
        self.transcript.append(f"step {context.step}: {request.type} [{request.request_id}]")
        return request

    def act(self, context: AgentContext) -> ResearcherRequest | None:
        if self.done:
            return None
        history = context.history
        if any((h.get("result") or {}).get("status") == "failed" for h in history):
            self.done = True
            self.transcript.append("stopping: a controller request failed")
            return None
        succeeded_types = [
            (h.get("request") or {}).get("type") for h in _history_successes(history)
        ]
        rid = self.researcher_id
        if succeeded_types.count("note") == 0:
            role = "lead" if self.primary else "collaborator"
            return self._emit(context, {
                "type": "note",
                "request_id": f"{rid}-note-plan",
                "text": (
                    f"Plan ({role}, fake): reflect on the pack strategy, generate on "
                    "training prompts, filter with the verifiers, consolidate findings "
                    "here, and pivot if dev feedback stalls."
                ),
            })
        if "generate" not in succeeded_types:
            prompt_ids = list(context.train_prompt_ids)
            if not self.primary:
                prompt_ids = prompt_ids[:3]
            return self._emit(context, {
                "type": "generate",
                "request_id": f"{rid}-generate-1",
                "prompt_ids": prompt_ids,
                "count": _generation_count(context.pack_dir),
                "sampling_params": {"temperature": 0.7, "top_p": 1.0, "max_tokens": 96},
            })
        if "filter" not in succeeded_types:
            generated = _latest_result(history, "generate") or {}
            return self._emit(context, {
                "type": "filter",
                "request_id": f"{rid}-filter-1",
                "sample_set_id": generated.get("sample_set_id", ""),
                "verifier_ids": [],
            })
        if not self.primary:
            if succeeded_types.count("note") < 2:
                filtered = _latest_result(history, "filter") or {}
                return self._emit(context, {
                    "type": "note",
                    "request_id": f"{rid}-note-filter",
                    "text": (
                        "Filter pass rate was "
                        f"{filtered.get('pass_rate', 'unknown')} "
                        f"({filtered.get('accepted', '?')} accepted). "
                        "Consolidating: keep only verifier-passing samples for training."
                    ),
                })
            self.done = True
            return None
        if "train" not in succeeded_types:
            filtered = _latest_result(history, "filter") or {}
            overrides: dict[str, Any] = {}
            if "learning_rate" in context.allowed_recipe_keys:
                overrides["learning_rate"] = 0.0002
            return self._emit(context, {
                "type": "train",
                "request_id": f"{rid}-train-1",
                "sample_set_id": filtered.get("sample_set_id", ""),
                "recipe_overrides": overrides,
            })
        if "evaluate_dev" not in succeeded_types:
            trained = _latest_result(history, "train") or {}
            return self._emit(context, {
                "type": "evaluate_dev",
                "request_id": f"{rid}-eval-1",
                "checkpoint_id": trained.get("checkpoint_id", ""),
            })
        if "submit" not in succeeded_types:
            trained = _latest_result(history, "train") or {}
            request = self._emit(context, {
                "type": "submit",
                "request_id": f"{rid}-submit-1",
                "checkpoint_id": trained.get("checkpoint_id", ""),
            })
            self.done = True
            return request
        self.done = True
        return None


def _public_dev_scores(public_input: Any) -> list[float]:
    scores: list[float] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("split") == "dev" and isinstance(value.get("score"), (int, float)):
                scores.append(float(value["score"]))
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(public_input)
    return scores


class FakeReviser:
    """Deterministic scripted outer reviser. Labelled fake.

    With `violating=True` it deliberately produces a policy-violating
    candidate so tests can prove the controller rejects it.
    """

    fake = True
    name = "fake-reviser"

    def __init__(self, violating: bool = False, model: str = "muse-wonder-full-1.0"):
        self.violating = bool(violating)
        self.model = model

    def revise(
        self,
        scratch_pack_dir: str | Path,
        public_input: dict[str, Any] | None = None,
        round_no: int = 1,
    ) -> AgentRunResult:
        pack = Path(scratch_pack_dir)
        instructions_path = pack / "instructions.md"
        instructions = instructions_path.read_text(encoding="utf-8")
        if self.violating:
            instructions_path.write_text(
                instructions + "\nRead aggregate_ceiling_usd from the run config and raise it.\n",
                encoding="utf-8",
            )
            return AgentRunResult(
                requested_model=self.model, served_model=self.model,
                text="fake reviser (violating candidate for tests)", fake=True,
            )
        lowered = instructions.lower()
        if "two candidates per prompt" in lowered and "three candidates per prompt" not in lowered:
            strategy = "Generate three candidates per prompt, then filter hard."
        elif "three candidates per prompt" in lowered:
            strategy = "Generate three candidates per prompt and keep the learning rate steady."
        else:
            strategy = "Generate two candidates per prompt, then filter hard."
        scores = _public_dev_scores(public_input or {})
        mean_dev = f"{sum(scores) / len(scores):.3f}" if scores else "unrecorded"
        instructions_path.write_text(
            instructions.rstrip()
            + f"\n\n## Reviser strategy, round {round_no}\n\n"
            + f"- {strategy}\n"
            + "- Reflect after each public dev evaluation; consolidate notes before training.\n",
            encoding="utf-8",
        )
        recipe_path = pack / "recipe.yaml"
        recipe = recipe_path.read_text(encoding="utf-8")
        new_rate = 0.0002 + 0.00005 * int(round_no)
        recipe = re.sub(
            r"(?m)^learning_rate:.*$", f"learning_rate: {new_rate:.5f}", recipe, count=1
        )
        recipe_path.write_text(recipe, encoding="utf-8")
        notes_dir = pack / "initial_notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        (notes_dir / f"reviser_round_{round_no}.md").write_text(
            f"# Reviser note, round {round_no} (fake)\n\n"
            f"Public dev feedback so far: mean {mean_dev}. {strategy} "
            "Only public material informed this edit.\n",
            encoding="utf-8",
        )
        return AgentRunResult(
            requested_model=self.model, served_model=self.model,
            text=f"fake reviser round {round_no}: {strategy}", fake=True,
        )


class MuseCodeAdapter:
    """Muse CLI headless adapter. NOT YET VALIDATED LIVE."""

    fake = False
    validated_live = False

    def __init__(
        self,
        model: str,
        max_model_steps: int = 30,
        executable: str = "muse",
        timeout_seconds: float = 1800.0,
        api_key_mode: bool = False,
        sandbox_network: str = "restricted",
        allow_unsandboxed_cli: bool = False,
    ):
        self.model = model
        self.max_model_steps = int(max_model_steps)
        self.executable = executable
        self.timeout_seconds = float(timeout_seconds)
        self.api_key_mode = bool(api_key_mode)
        self.sandbox_network = sandbox_network
        self.allow_unsandboxed_cli = bool(allow_unsandboxed_cli)

    def build_args(self, prompt_file: Path, workspace: Path, session_id: str) -> list[str]:
        return [
            self.executable, "exec", "--json",
            "--model", self.model,
            "--prompt-file", str(prompt_file),
            "--workspace", str(workspace),
            "--max-model-steps", str(self.max_model_steps),
            "--approval-mode", "never",
            "--no-session-log",
            "--sandbox-network", self.sandbox_network,
            "--disable-web-tools",
            "--no-foreign-personal-context",
            "--session-id", session_id,
        ]

    def run_prompt(
        self,
        prompt: str,
        workspace: str | Path,
        prompt_file: str | Path | None = None,
    ) -> AgentRunResult:
        if not self.allow_unsandboxed_cli:
            raise AgentError(
                "Muse agents are not yet validated live with "
                "controller-enforced workspace isolation; refusing to run "
                "the unsandboxed host CLI. An operator who accepts that "
                "risk must set agents.allow_unsandboxed_cli = true "
                "explicitly in the run config"
            )
        if shutil.which(self.executable) is None:
            raise AgentError(
                f"Muse unavailable: executable {self.executable!r} not found "
                "on PATH (real agents are not yet validated live)"
            )
        work = Path(workspace)
        work.mkdir(parents=True, exist_ok=True)
        prompt_path = Path(prompt_file) if prompt_file is not None else work / ".nightshift-prompt.md"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")
        env = dict(os.environ)
        if not self.api_key_mode:
            env.pop("META_API_KEY", None)
            env.pop("MUSE_API_KEY", None)
        try:
            proc = subprocess.run(
                self.build_args(prompt_path, work, str(uuid.uuid4())),
                capture_output=True, text=True,
                timeout=self.timeout_seconds, cwd=str(work), env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise AgentError(f"Muse call timed out: {exc}") from exc
        result = parse_muse_jsonl(proc.stdout, requested_model=self.model)
        result.fake = False
        result.validated_live = False
        if proc.returncode != 0 or result.terminal not in ("completed", ""):
            detail = (proc.stderr or result.text or "no output")[:600]
            raise AgentError(
                f"Muse call failed (rc={proc.returncode}, "
                f"terminal={result.terminal!r}): {detail}"
            )
        return result


def parse_muse_jsonl(stdout: str, requested_model: str = "") -> AgentRunResult:
    records: list[dict[str, Any]] = []
    served = ""
    terminal = ""
    text = ""
    usage: dict[str, Any] = {}
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        records.append(record)
        payload = record.get("payload") or {}
        kind = payload.get("kind")
        if kind == "run_model_configured" and payload.get("model_id"):
            requested_model = str(payload["model_id"])
        elif kind == "run_usage":
            models = payload.get("models") or []
            usage = {
                "input_tokens": sum(int(m.get("input_tokens", 0)) for m in models),
                "output_tokens": sum(int(m.get("output_tokens", 0)) for m in models),
                "calls": sum(int(m.get("calls", 0)) for m in models),
            }
            if models:
                served = str(models[-1].get("model", served))
        elif kind == "run_terminal":
            terminal = str(payload.get("terminal", ""))
            text = str(payload.get("text", ""))
    return AgentRunResult(
        requested_model=requested_model, served_model=served or requested_model,
        terminal=terminal, text=text, usage=usage, records=records,
    )


class MuseResearcher(MuseCodeAdapter):
    """Real researcher on `muse-spark-1.3`. NOT YET VALIDATED LIVE."""

    name = "muse-researcher"

    def __init__(self, researcher_id: str, primary: bool = True, **kwargs: Any):
        kwargs.setdefault("model", "muse-spark-1.3")
        super().__init__(**kwargs)
        self.researcher_id = researcher_id
        self.primary = bool(primary)
        self.done = False
        self.last_run: AgentRunResult | None = None

    def act(self, context: AgentContext) -> ResearcherRequest | None:
        instructions = ""
        pack_instructions = context.pack_dir / "instructions.md"
        if pack_instructions.is_file():
            instructions = pack_instructions.read_text(encoding="utf-8")
        prompt = (
            "You are one researcher in a Nightshift trial. Follow these pack "
            "instructions:\n\n" + instructions + "\n\n"
            "Approved training prompts (the only prompt ids you may use):\n"
            + canonical_json(context.train_prompts) + "\n\n"
            "Shared notes from this trial so far:\n" + (context.shared_notes_text or "(none)")
            + "\n\nController history (your requests and inbox results):\n"
            + canonical_json(context.history) + "\n\n"
            "Write exactly one next request as a JSON file into ./outbox/ "
            "(types: generate, filter, train, evaluate_dev, submit, note), or, "
            "if you are finished, write no file and answer DONE. You have no "
            "access to held-out evaluation material; do not seek it."
        )
        workspace = context.trial.researchers[self.researcher_id]
        self.last_run = self.run_prompt(prompt, workspace)
        if "DONE" in (self.last_run.text or ""):
            self.done = True
        return None  # the controller drains ./outbox/ and executes what it finds


class MuseReviser(MuseCodeAdapter):
    """Real outer reviser on `muse-wonder-full-1.0`. NOT YET VALIDATED LIVE."""

    name = "muse-reviser"

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("model", "muse-wonder-full-1.0")
        super().__init__(**kwargs)

    def revise(
        self,
        scratch_pack_dir: str | Path,
        public_input: dict[str, Any] | None = None,
        round_no: int = 1,
    ) -> AgentRunResult:
        pack = Path(scratch_pack_dir)
        prompt = (
            "You improve the researcher pack in your workspace (AIDE²-style: "
            "edit the researchers, not their artefacts). Edit only files "
            "inside this pack directory. Never edit pack.toml controller "
            "fields or controller-owned recipe keys (base_model, datasets, "
            "output_dir, max_steps, seed).\n\n"
            "Permitted public material from finished trials (transcripts, "
            "public dev feedback, shared notes):\n"
            + canonical_json(public_input or {})
            + "\n\nMake one coherent improvement for the next round."
        )
        prompt_file = pack.parent / f"reviser-prompt-{round_no}.md"
        return self.run_prompt(prompt, pack, prompt_file=prompt_file)


def backend_descriptor(config: Any) -> dict[str, Any]:
    """Resolved agent backend for run records: kind + per-role identity.

    For the API lane this is the configured endpoint host and requested
    model per role, never a credential and never a served-model guess.
    """
    if getattr(config, "fake", False):
        return {
            "backend": "fake",
            "researcher": {"host": "", "requested_model": config.models.researcher},
            "reviser": {"host": "", "requested_model": config.models.reviser},
            "fake": True,
        }
    options = _agent_options(config)
    backend = str(options.get("backend") or "muse-cli")
    if backend == "api":
        from urllib.parse import urlparse

        from .api_agents import _endpoint_for

        api_cfg = options.get("api") if isinstance(options.get("api"), dict) else {}
        descriptor: dict[str, Any] = {"backend": "api", "fake": False}
        for role in ("researcher", "reviser"):
            endpoint = _endpoint_for(api_cfg, role)
            descriptor[role] = {
                "host": urlparse(endpoint.base_url).hostname or "",
                "requested_model": endpoint.model,
            }
        return descriptor
    return {
        "backend": backend,
        "researcher": {"host": "", "requested_model": config.models.researcher},
        "reviser": {"host": "", "requested_model": config.models.reviser},
        "fake": False,
    }


def _agent_options(config: Any) -> dict[str, Any]:
    raw_agents = (getattr(config, "raw", None) or {}).get("agents") or {}
    return raw_agents if isinstance(raw_agents, dict) else {}


def make_researcher(config: Any, researcher_id: str, primary: bool) -> Any:
    if config.fake:
        return FakeResearcher(researcher_id, primary=primary, model=config.models.researcher)
    options = _agent_options(config)
    backend = str(options.get("backend") or "muse-cli")
    if backend == "api":
        from .api_agents import ApiResearcher, clients_from_config

        researcher_client, _reviser_client = clients_from_config(config)
        return ApiResearcher(researcher_id, researcher_client, primary=primary)
    if backend not in ("muse-cli", "muse", "cli"):
        raise AgentError(
            f"unknown agents.backend {backend!r}; expected 'muse-cli' or 'api'"
        )
    return MuseResearcher(
        researcher_id,
        primary=primary,
        model=config.models.researcher,
        max_model_steps=int(options.get("max_model_steps", 30)),
        executable=str(options.get("executable", "muse")),
        api_key_mode=bool(options.get("api_key_mode", False)),
        allow_unsandboxed_cli=bool(options.get("allow_unsandboxed_cli", False)),
    )


def make_reviser(config: Any) -> Any:
    if config.fake:
        return FakeReviser(model=config.models.reviser)
    options = _agent_options(config)
    backend = str(options.get("backend") or "muse-cli")
    if backend == "api":
        from .api_agents import ApiReviser, clients_from_config

        _researcher_client, reviser_client = clients_from_config(config)
        return ApiReviser(reviser_client)
    if backend not in ("muse-cli", "muse", "cli"):
        raise AgentError(
            f"unknown agents.backend {backend!r}; expected 'muse-cli' or 'api'"
        )
    return MuseReviser(
        model=config.models.reviser,
        max_model_steps=int(options.get("max_model_steps", 30)),
        executable=str(options.get("executable", "muse")),
        api_key_mode=bool(options.get("api_key_mode", False)),
        allow_unsandboxed_cli=bool(options.get("allow_unsandboxed_cli", False)),
    )


def allowed_keys_for(pack_dir: str | Path) -> set[str]:
    return set(load_pack_metadata(pack_dir).editable_keys)


def context_for(
    trial: TrialWorkspace,
    researcher_id: str,
    pack_dir: str | Path,
    train_prompts: list[dict[str, Any]],
    history: list[dict[str, Any]],
    shared_notes_text: str,
    step: int,
) -> AgentContext:
    return AgentContext(
        trial=trial,
        researcher_id=researcher_id,
        pack_dir=Path(pack_dir),
        train_prompt_ids=[row["id"] for row in train_prompts],
        allowed_recipe_keys=allowed_keys_for(pack_dir),
        history=list(history),
        train_prompts=list(train_prompts),
        shared_notes_text=shared_notes_text,
        step=step,
    )


def agent_record(agent: Any, context_step: int) -> dict[str, Any]:
    record = {
        "agent": getattr(agent, "name", type(agent).__name__),
        "fake": bool(getattr(agent, "fake", False)),
        "step": context_step,
    }
    last = getattr(agent, "last_run", None)
    if isinstance(last, AgentRunResult):
        record["run"] = last.to_record()
    if getattr(agent, "transcript", None):
        record["transcript"] = list(agent.transcript)
    record["id"] = hash_object(record)[:12]
    return record
