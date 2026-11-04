"""Per-role model routing.

``load_routing(path)`` returns one ``Endpoint`` per role. Defaults: every role uses ``muse-spark-1.3`` on the Meta API at
the effort in ``DEFAULT_EFFORT``; ``reviewer_hard`` and ``auditor_final`` use ``muse-wonder-1.0`` with ``muse-spark-1.3`` as
fallback, and when the fallback answers the effort is ``xhigh``. A ``-contributor`` model (which lets Meta train on the
content) is never a default; it can only be chosen by naming it in the config.

``nightshift.toml`` (all optional)::

    [models.default]            # applies to every role
    base_url = "https://api.meta.ai/v1"
    model = "muse-spark-1.3"
    api_key_env = "MODEL_API_KEY"
    reasoning_effort = "medium"
    fallback_model = ""         # empty string removes a fallback
    completion_cap_ceiling = 32000   # integer: no request asks for more (the docs state no maximum; above it is a 400)
    api = "chat"                # "chat" (Chat Completions, the default) or "responses" (opt-in, see below)
    allow_partner_responses = false  # true: allow api = "responses" with a base_url other than the Meta Model API

    [roles.reviewer]            # overrides for one role (same keys)
    model = "muse-spark-1.2"

Request timeouts are not configured here: ``client_for`` leaves ``timeout`` to ``MetaChatClient``, which allows 300 s per
request and 600 s for a call whose ``max_completion_tokens`` is above 8000 (reasoning calls are slow); pass ``timeout=``
in ``client_kw`` to override both. A role whose primary model answers 404 ``model_not_found`` (for instance Muse Wonder
on a host that does not serve it) keeps using its ``fallback_model`` for the rest of the client's life.

``api = "responses"`` selects ``MetaResponsesClient`` for that role (reasoning carried across tool turns; see
``nightshift/models/responses.py`` and ``docs/META_API_NOTES.md``). It is meant for the roles that run tool loops
(``[roles.supervisor]`` and ``[roles.implementer]``); the single-turn structured roles gain nothing from it. It is UNTESTED against
the real service, which is why ``chat`` stays the default. A Responses role with a ``base_url`` other than the Meta Model API is a
``ModelConfigError`` at load time unless ``allow_partner_responses = true`` (partner hosts are not known to implement Responses);
an unknown ``api`` value is a ``ModelConfigError`` too.

Setting ``model`` or ``base_url`` in ``[models.default]`` also moves ``reviewer_hard`` and ``auditor_final`` off Muse Wonder
(and drops their fallback): a partner host is assumed not to serve it. Override those roles explicitly to keep Wonder.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from ..types import ModelClient
from .base import EFFORTS, ModelConfigError, supports_max_effort
from .meta import DEFAULT_BASE_URL, DEFAULT_MODEL, MetaChatClient, check_base_url

ROLES = ("supervisor", "designer", "implementer", "reviewer", "reviewer_hard", "reflector", "judge", "auditor_final")
DEFAULT_EFFORT = {"supervisor": "medium", "designer": "low", "implementer": "low", "reviewer": "high",
                  "reviewer_hard": "high", "reflector": "medium", "judge": "high", "auditor_final": "high"}
HARD_ROLES = ("reviewer_hard", "auditor_final")
WONDER_MODEL = "muse-wonder-1.0"
FALLBACK_EFFORT = "xhigh"
_KEYS = ("base_url", "model", "api_key_env", "reasoning_effort", "fallback_model", "completion_cap_ceiling", "api",
         "allow_partner_responses")
_INT_KEYS = ("completion_cap_ceiling",)
_BOOL_KEYS = ("allow_partner_responses",)
APIS = ("chat", "responses")


@dataclass
class Endpoint:
    base_url: str
    model: str
    api_key_env: str
    reasoning_effort: str | None
    fallback_model: str | None
    fallback_reasoning: str | None = None     # effort used when the fallback model answers
    completion_cap_ceiling: int | None = None  # max_completion_tokens is clamped to this; None means no configured limit
    api: str = "chat"                          # "chat" or "responses" (see the module docstring)
    allow_partner_responses: bool = False      # api = "responses" is otherwise refused for a non-Meta base_url


def _default_endpoint(role: str) -> Endpoint:
    if role in HARD_ROLES:
        return Endpoint(DEFAULT_BASE_URL, WONDER_MODEL, "MODEL_API_KEY", DEFAULT_EFFORT[role], DEFAULT_MODEL, FALLBACK_EFFORT)
    return Endpoint(DEFAULT_BASE_URL, DEFAULT_MODEL, "MODEL_API_KEY", DEFAULT_EFFORT[role], None, None)


def _check_section(name: str, section: Any) -> dict[str, Any]:
    if not isinstance(section, Mapping):
        raise ModelConfigError(f"[{name}] must be a table")
    unknown = sorted(set(section) - set(_KEYS))
    if unknown:
        raise ModelConfigError(f"[{name}] has unknown key(s) {', '.join(unknown)}; allowed: {', '.join(_KEYS)}")
    for key, value in section.items():
        if key in _INT_KEYS:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ModelConfigError(f"[{name}] {key} must be a positive integer")
        elif key in _BOOL_KEYS:
            if not isinstance(value, bool):
                raise ModelConfigError(f"[{name}] {key} must be true or false")
        elif not isinstance(value, str):
            raise ModelConfigError(f"[{name}] {key} must be a string")
    return dict(section)


def _apply(ep: Endpoint, values: Mapping[str, Any], *, where: str) -> Endpoint:
    out = replace(ep)
    if "base_url" in values:
        out.base_url = values["base_url"]
    if "model" in values:
        out.model = values["model"].strip()
    if "api_key_env" in values:
        out.api_key_env = values["api_key_env"].strip()
    if "reasoning_effort" in values:
        out.reasoning_effort = values["reasoning_effort"].strip().lower() or None
    if "fallback_model" in values:
        out.fallback_model = values["fallback_model"].strip() or None
        out.fallback_reasoning = out.fallback_reasoning if out.fallback_model else None
    if "completion_cap_ceiling" in values:
        out.completion_cap_ceiling = values["completion_cap_ceiling"]
    if "api" in values:
        out.api = values["api"].strip().lower()
    if "allow_partner_responses" in values:
        out.allow_partner_responses = values["allow_partner_responses"]
    if not out.model:
        raise ModelConfigError(f"{where}: model must not be empty")
    if not out.api_key_env:
        raise ModelConfigError(f"{where}: api_key_env must not be empty")
    return out


def check_api(role: str, ep: Endpoint) -> None:
    """Raise ``ModelConfigError`` unless ``ep.api`` is a known protocol and, for ``responses``, the host is the Meta Model API
    (or ``allow_partner_responses`` is set). Called by ``load_routing`` and again by ``client_for``, so an ``Endpoint`` built in
    code is held to the same rule."""
    where = f"role {role!r}"
    if ep.api not in APIS:
        raise ModelConfigError(f"{where}: api {ep.api!r} must be one of {', '.join(APIS)}")
    if ep.api == "responses" and not ep.allow_partner_responses and check_base_url(ep.base_url) != DEFAULT_BASE_URL:
        raise ModelConfigError(
            f"{where}: api = \"responses\" is only supported for the Meta Model API ({DEFAULT_BASE_URL}), but base_url is another "
            f"host. Partner hosts are not known to implement the Responses API; use api = \"chat\" for this role, or set "
            f"allow_partner_responses = true if you know the host serves /responses")


def _validate(role: str, ep: Endpoint) -> None:
    where = f"role {role!r}"
    ep.base_url = check_base_url(ep.base_url)
    check_api(role, ep)
    effort = ep.reasoning_effort
    if effort is not None and effort not in EFFORTS + ("none", "max"):
        raise ModelConfigError(f"{where}: reasoning_effort {effort!r} must be one of none, {', '.join(EFFORTS)}, max")
    if effort == "max" and not supports_max_effort(ep.model):
        raise ModelConfigError(f"{where}: reasoning_effort 'max' is only valid for muse-wonder-* and muse-spark-1.3 "
                               f"(not -contributor or other models), not {ep.model!r}")


def load_routing(path: Path | str | None) -> dict[str, Endpoint]:
    """Endpoints for every role in ``ROLES``. ``path=None`` gives the defaults; a path that does not exist, is not valid
    TOML, or contains an unknown role/key/value raises ``ModelConfigError``."""
    routing = {role: _default_endpoint(role) for role in ROLES}
    if path is not None:
        p = Path(path)
        try:
            data = tomllib.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ModelConfigError(f"config file not found: {p}") from None
        except (tomllib.TOMLDecodeError, UnicodeDecodeError, OSError) as exc:
            raise ModelConfigError(f"cannot read {p}: {exc}") from None
        models = data.get("models", {})
        if not isinstance(models, Mapping) or set(models) - {"default"}:
            raise ModelConfigError("[models] may only contain [models.default]")
        default = _check_section("models.default", models.get("default", {}))
        for role in ROLES:
            if role in HARD_ROLES and ("model" in default or "base_url" in default):
                routing[role] = replace(routing[role], model=DEFAULT_MODEL, fallback_model=None, fallback_reasoning=None)
            routing[role] = _apply(routing[role], default, where="[models.default]")
        roles = data.get("roles", {})
        if not isinstance(roles, Mapping):
            raise ModelConfigError("[roles] must be a table of tables")
        for role, section in roles.items():
            if role not in ROLES:
                raise ModelConfigError(f"[roles.{role}]: unknown role; valid roles are {', '.join(ROLES)}")
            routing[role] = _apply(routing[role], _check_section(f"roles.{role}", section), where=f"[roles.{role}]")
    for role, ep in routing.items():
        _validate(role, ep)
    return routing


def client_for(role: str, routing: Mapping[str, Endpoint], gate: Any = None, actor: str | None = None,
               **client_kw: Any) -> ModelClient:
    """A ``ModelClient`` for ``role``. With a ``gate`` (a ``BudgetGate``) the client is wrapped in
    ``budget.MeteredClient`` charging ``actor`` (default: the role). The client is a ``MetaChatClient`` unless the role's
    ``api`` is ``responses`` (``MetaResponsesClient``, same protocol). ``client_kw`` goes to the client (tests pass
    ``transport``, ``sleep`` ...). Raises ``ModelConfigError`` if the role is unknown, its ``api`` is unknown or not allowed for
    its host, or no API key is set."""
    if role not in routing:
        raise ModelConfigError(f"unknown role {role!r}; valid roles are {', '.join(routing) or '(none)'}")
    ep = routing[role]
    check_api(role, ep)
    if ep.completion_cap_ceiling is not None:
        client_kw.setdefault("completion_cap_ceiling", ep.completion_cap_ceiling)
    if ep.api == "responses":
        from .responses import MetaResponsesClient    # lazy: chat-only runs never import it
        factory: Any = MetaResponsesClient
    else:
        factory = MetaChatClient
    client = factory(ep.model, base_url=ep.base_url, api_key_env=ep.api_key_env, default_reasoning=ep.reasoning_effort,
                     fallback_model=ep.fallback_model, fallback_reasoning=ep.fallback_reasoning, **client_kw)
    if gate is None:
        return client
    from ..budget import MeteredClient  # lazy: budget imports models.pricing
    return MeteredClient(client, gate, actor or role)
