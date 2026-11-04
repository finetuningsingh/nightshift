"""Trial directories, request files, and mount validation (spec 3.6).

CORAL-style collaboration, implemented natively: one isolated private
workspace per researcher plus a `shared/notes/` area that exists only inside
that trial. Researchers act through validated JSON request files; anything a
researcher writes is untrusted data.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import NightshiftError, PolicyError
from .hashing import canonical_json, hash_object
from .verifiers import get_verifier


class InvalidRequest(NightshiftError):
    """An untrusted researcher request failed strict validation."""


ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REQUEST_TYPES = ("generate", "filter", "train", "evaluate_dev", "submit", "note")
MAX_NOTE_CHARS = 10_000
MAX_PROMPT_IDS = 512


def _check_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not ID_RE.match(value) or ".." in value:
        raise InvalidRequest(f"{field_name} must be a plain id, got {value!r}")
    return value


@dataclass(frozen=True)
class ResearcherRequest:
    type: str
    request_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    researcher_id: str = ""

    def to_record(self) -> dict[str, Any]:
        return {"type": self.type, "request_id": self.request_id, **self.payload}


def validate_request(
    raw: Any,
    allowed_recipe_keys: set[str] | None = None,
    researcher_id: str = "",
) -> ResearcherRequest:
    """Strictly validate one untrusted request object."""
    if not isinstance(raw, dict):
        raise InvalidRequest("request must be a JSON object")
    request_type = raw.get("type")
    if request_type not in REQUEST_TYPES:
        raise InvalidRequest(
            f"unknown request type {request_type!r}; expected one of {REQUEST_TYPES}"
        )

    def only_fields(*fields: str) -> None:
        extra = set(raw) - {"type", "request_id", *fields}
        if extra:
            raise InvalidRequest(
                f"{request_type} request has unexpected field(s): {sorted(extra)}"
            )

    payload: dict[str, Any] = {}
    if request_type == "generate":
        only_fields("prompt_ids", "count", "sampling_params")
        prompt_ids = raw.get("prompt_ids")
        if not isinstance(prompt_ids, list) or not prompt_ids or len(prompt_ids) > MAX_PROMPT_IDS:
            raise InvalidRequest("generate.prompt_ids must be a non-empty list of ids")
        payload["prompt_ids"] = [_check_id(p, "prompt_ids[]") for p in prompt_ids]
        count = raw.get("count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or not (1 <= count <= 64):
            raise InvalidRequest("generate.count must be an integer in 1..64")
        payload["count"] = count
        sampling = raw.get("sampling_params", {})
        if not isinstance(sampling, dict):
            raise InvalidRequest("generate.sampling_params must be an object")
        unknown = set(sampling) - {"temperature", "top_p", "max_tokens"}
        if unknown:
            raise InvalidRequest(f"unknown sampling param(s): {sorted(unknown)}")
        temperature = sampling.get("temperature", 0.7)
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) \
                or not (0.0 <= float(temperature) <= 2.0):
            raise InvalidRequest("sampling_params.temperature must be in 0..2")
        top_p = sampling.get("top_p", 1.0)
        if not isinstance(top_p, (int, float)) or isinstance(top_p, bool) \
                or not (0.0 <= float(top_p) <= 1.0):
            raise InvalidRequest("sampling_params.top_p must be in 0..1")
        max_tokens = sampling.get("max_tokens", 128)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) \
                or not (1 <= max_tokens <= 2048):
            raise InvalidRequest("sampling_params.max_tokens must be an integer in 1..2048")
        payload["sampling_params"] = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
    elif request_type == "filter":
        only_fields("sample_set_id", "verifier_ids")
        payload["sample_set_id"] = _check_id(raw.get("sample_set_id"), "sample_set_id")
        verifier_ids = raw.get("verifier_ids", [])
        if not isinstance(verifier_ids, list):
            raise InvalidRequest("filter.verifier_ids must be a list")
        for verifier_id in verifier_ids:
            try:
                get_verifier(str(verifier_id))
            except Exception as exc:
                raise InvalidRequest(str(exc)) from exc
        payload["verifier_ids"] = [str(v) for v in verifier_ids]
    elif request_type == "train":
        only_fields("sample_set_id", "recipe_overrides")
        payload["sample_set_id"] = _check_id(raw.get("sample_set_id"), "sample_set_id")
        overrides = raw.get("recipe_overrides", {})
        if not isinstance(overrides, dict):
            raise InvalidRequest("train.recipe_overrides must be an object")
        allowed = allowed_recipe_keys if allowed_recipe_keys is not None else set()
        unknown = set(overrides) - allowed
        if unknown:
            raise InvalidRequest(
                f"recipe override key(s) not allowed by the pack: {sorted(unknown)}"
            )
        for key, value in overrides.items():
            if not isinstance(value, (str, int, float, bool)):
                raise InvalidRequest(f"recipe override {key!r} must be a scalar")
        payload["recipe_overrides"] = dict(overrides)
    elif request_type in ("evaluate_dev", "submit"):
        only_fields("checkpoint_id")
        payload["checkpoint_id"] = _check_id(raw.get("checkpoint_id"), "checkpoint_id")
    elif request_type == "note":
        only_fields("text")
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_NOTE_CHARS:
            raise InvalidRequest(f"note.text must be 1..{MAX_NOTE_CHARS} characters")
        payload["text"] = text

    request_id = raw.get("request_id")
    if request_id is None:
        request_id = f"{request_type}-{hash_object(payload)[:12]}"
    else:
        request_id = _check_id(request_id, "request_id")
    return ResearcherRequest(
        type=request_type, request_id=request_id, payload=payload, researcher_id=researcher_id
    )


@dataclass
class TrialWorkspace:
    trial_id: str
    path: Path
    researchers: dict[str, Path]
    shared_notes: Path
    pack_version: str = ""
    seed: int = 0

    def outbox(self, researcher_id: str) -> Path:
        return self.researchers[researcher_id] / "outbox"

    def inbox(self, researcher_id: str) -> Path:
        return self.researchers[researcher_id] / "inbox"

    def controller_dir(self) -> Path:
        return self.path / "controller"


def create_trial_workspace(
    root: str | Path,
    trial_id: str,
    researcher_ids: list[str],
    pack_version: str = "",
    seed: int = 0,
    initial_notes_dir: str | Path | None = None,
    initial_notes: Mapping[str, str] | None = None,
) -> TrialWorkspace:
    """Fresh trial directory: private workspaces + trial-only shared notes."""
    _check_id(trial_id, "trial_id")
    if not researcher_ids:
        raise InvalidRequest("a trial needs at least one researcher")
    trial_path = Path(root) / trial_id
    if trial_path.exists() and any(trial_path.iterdir()):
        raise PolicyError(f"trial directory already exists and is not empty: {trial_path}")
    researchers: dict[str, Path] = {}
    for researcher_id in researcher_ids:
        _check_id(researcher_id, "researcher_id")
        workspace = trial_path / "researchers" / researcher_id
        (workspace / "outbox").mkdir(parents=True, exist_ok=True)
        (workspace / "inbox").mkdir(parents=True, exist_ok=True)
        researchers[researcher_id] = workspace
    shared_notes = trial_path / "shared" / "notes"
    shared_notes.mkdir(parents=True, exist_ok=True)
    if initial_notes_dir is not None and Path(initial_notes_dir).is_dir():
        for note in sorted(Path(initial_notes_dir).iterdir()):
            if note.is_file() and not note.is_symlink():
                shutil.copy2(note, shared_notes / note.name)
    for name, text in (initial_notes or {}).items():
        safe = Path(str(name)).name
        (shared_notes / safe).write_text(str(text), encoding="utf-8")
    (trial_path / "controller").mkdir(parents=True, exist_ok=True)
    record = {
        "trial_id": trial_id,
        "pack_version": pack_version,
        "seed": seed,
        "researchers": list(researchers),
    }
    (trial_path / "trial.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    return TrialWorkspace(
        trial_id=trial_id,
        path=trial_path,
        researchers=researchers,
        shared_notes=shared_notes,
        pack_version=pack_version,
        seed=seed,
    )


def append_shared_note(trial: TrialWorkspace, author: str, text: str) -> Path:
    note_file = trial.shared_notes / "notes.md"
    with note_file.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {author}\n\n{text.strip()}\n")
    return note_file


def read_shared_notes(trial: TrialWorkspace) -> str:
    parts: list[str] = []
    for note in sorted(trial.shared_notes.iterdir()):
        if note.is_file():
            parts.append(note.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _atomic_write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(canonical_json(payload), encoding="utf-8")
    tmp.replace(path)
    return path


def write_request(
    trial: TrialWorkspace, researcher_id: str, request: ResearcherRequest
) -> Path:
    return _atomic_write_json(
        trial.outbox(researcher_id) / f"{request.request_id}.json", request.to_record()
    )


def write_inbox_result(
    trial: TrialWorkspace, researcher_id: str, request_id: str, result: Any
) -> Path:
    _check_id(request_id, "request_id")
    return _atomic_write_json(trial.inbox(researcher_id) / f"{request_id}.json", result)


def load_outbox(
    trial: TrialWorkspace,
    researcher_id: str,
    allowed_recipe_keys: set[str] | None = None,
) -> tuple[list[ResearcherRequest], list[dict[str, str]]]:
    """Parse every outbox file; malformed agent files become errors, never crashes."""
    valid: list[ResearcherRequest] = []
    errors: list[dict[str, str]] = []
    for path in sorted(trial.outbox(researcher_id).glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            valid.append(
                validate_request(raw, allowed_recipe_keys, researcher_id=researcher_id)
            )
        except (json.JSONDecodeError, InvalidRequest, OSError) as exc:
            errors.append({"file": path.name, "error": str(exc)})
    return valid, errors


# --- Container mount-spec validation (spec 3.6, A11) -----------------------

@dataclass(frozen=True)
class MountSpec:
    source: str
    target: str
    mode: str = "ro"  # "ro" or "rw"


def parse_mount_spec(raw: Any) -> MountSpec:
    if isinstance(raw, MountSpec):
        return raw
    if isinstance(raw, dict):
        source = raw.get("source") or raw.get("src") or raw.get("host")
        target = raw.get("target") or raw.get("dst") or raw.get("container")
        mode = raw.get("mode") or ("ro" if raw.get("read_only") else "rw")
        if not source or not target:
            raise PolicyError(f"mount spec needs source and target: {raw!r}")
        return MountSpec(str(source), str(target), str(mode))
    if isinstance(raw, str):
        parts = raw.split(":")
        if len(parts) == 2:
            return MountSpec(parts[0], parts[1], "rw")
        if len(parts) == 3:
            return MountSpec(parts[0], parts[1], parts[2])
    raise PolicyError(f"unparseable mount spec: {raw!r}")


_CREDENTIAL_COMPONENTS = {
    ".ssh", ".aws", ".gnupg", ".kube", "credentials", ".credentials",
    "secrets", ".secrets", "id_rsa", "id_ed25519", ".netrc", ".docker",
}


def _forbidden_reason(raw_source: str, resolved: Path) -> str | None:
    lowered = raw_source.lower()
    parts = [p.lower() for p in resolved.parts]
    name = resolved.name.lower()
    if "docker.sock" in lowered or "podman.sock" in lowered or name.endswith(".sock"):
        return "container runtime socket mounts are forbidden"
    if raw_source.startswith("~") or "$home" in lowered or "${home}" in lowered:
        return "home directory mounts are forbidden"
    home = Path.home()
    home_parts = [p.lower() for p in home.parts]
    if parts[: len(home_parts)] == home_parts[: len(parts)] and len(parts) <= len(home_parts):
        return "home directory mounts are forbidden"
    if len(parts) >= 3 and parts[1] in ("home", "users") and len(parts) <= 3:
        return "home directory mounts are forbidden"
    if str(resolved) in ("/", "/mnt", "/Volumes", "/media") or len(parts) <= 1:
        return "whole-volume mounts are forbidden"
    if _CREDENTIAL_COMPONENTS & set(parts):
        return "credential path mounts are forbidden"
    return None


def validate_mount_specs(
    mounts: list[Any],
    workspace: str | Path,
    shared_notes: str | Path,
    pack_dir: str | Path,
    train_prompts: str | Path,
    private_paths: list[str | Path] | tuple[str | Path, ...] = (),
) -> list[MountSpec]:
    """Allow only: own workspace (rw), shared notes (rw), pack (ro), train prompts (ro)."""
    allowed: list[tuple[str, Path, set[str]]] = [
        ("workspace", Path(workspace).expanduser().resolve(), {"rw", "ro"}),
        ("shared_notes", Path(shared_notes).expanduser().resolve(), {"rw", "ro"}),
        ("pack", Path(pack_dir).expanduser().resolve(), {"ro"}),
        ("train_prompts", Path(train_prompts).expanduser().resolve(), {"ro"}),
    ]
    private = [Path(p).expanduser().resolve() for p in private_paths]
    result: list[MountSpec] = []
    seen_targets: set[str] = set()
    for raw in mounts:
        spec = parse_mount_spec(raw)
        if spec.mode not in ("ro", "rw"):
            raise PolicyError(f"mount mode must be 'ro' or 'rw': {spec!r}")
        if not spec.target.startswith("/") or ".." in Path(spec.target).parts:
            raise PolicyError(f"mount target must be an absolute container path: {spec.target!r}")
        if spec.target in seen_targets:
            raise PolicyError(f"duplicate mount target {spec.target!r}")
        seen_targets.add(spec.target)
        resolved = Path(spec.source).expanduser().resolve()
        reason = _forbidden_reason(spec.source, resolved)
        if reason:
            raise PolicyError(f"rejected mount {spec.source!r}: {reason}")
        for private_path in private:
            if (
                resolved == private_path
                or private_path in resolved.parents
                or resolved in private_path.parents
            ):
                raise PolicyError(
                    f"rejected mount {spec.source!r}: overlaps a private split path"
                )
        matched: tuple[str, Path, set[str]] | None = None
        for label, root, modes in allowed:
            if resolved == root or root in resolved.parents:
                matched = (label, root, modes)
                break
        if matched is None:
            raise PolicyError(
                f"rejected mount {spec.source!r}: not inside the researcher's workspace, "
                "trial shared notes, pack, or train_prompts"
            )
        if spec.mode not in matched[2]:
            raise PolicyError(
                f"rejected mount {spec.source!r}: {matched[0]} may not be mounted {spec.mode}"
            )
        result.append(spec)
    return result
