"""Choose the agent suite for a backend (docs/ARCHITECTURE.md 6.6).

``scripted`` builds the offline fault-injection suite; ``meta`` builds the Muse Spark suite. Both builders are imported
lazily so that importing this module needs neither of them (and never touches the network or an API key).

Two options exist for ``meta`` only: ``plant`` attaches a scripted attacker suite (``suite.planted``) for the red-team lanes
of ``nightshift run --plant``, and ``background`` hands operator-supplied background material to the supervisor and designer.
Asking for either with ``scripted`` is an error, never a silent no-op.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..types import AgentSuite, TaskSpec

BACKENDS = ("scripted", "meta")


def build_suite(backend: str, task: TaskSpec, run_dir: Path | str, routing: Mapping[str, Any] | None, gate: Any,
                ledger: Any, *, implementer: str = "patch", control: bool = False, plant: Sequence[str] = (),
                background: str = "") -> AgentSuite:
    """Return the ``AgentSuite`` for ``backend``.

    ``control`` only affects the scripted backend (it selects the clean control manifest) and is not passed to ``meta``;
    ``routing`` (``None`` means the default routing), ``gate`` and ``implementer`` are used by ``meta`` only. An unknown
    backend raises ``ValueError`` before anything is imported. Errors from the builders (a missing key, an unknown implementer) propagate unchanged.

    ``plant`` (behavior names of the task's scripted bank, e.g. ``["tamper", "overfit"]``) is validated first (an unknown
    behavior, or one the bank lacks, raises ``ValueError`` before any model client is built) and then attached as
    ``suite.planted``: a scripted suite whose designer and implementer serve ONLY the lanes that carry a ``scripted_id``. The
    supervisor, reviewer, reflector and judges stay the real model roles. ``background`` is forwarded to the LLM suite only
    when it is non-empty, so the builder's own default applies otherwise. Both raise ``ValueError`` with ``scripted``.
    """
    plant = list(plant)
    if backend == "scripted":
        if plant:
            raise ValueError("plant needs the meta backend: the scripted backend already plants its own faults")
        if background:
            raise ValueError("background needs the meta backend: the scripted backend has no language model to read it")
        from nightshift.agents.scripted import build_scripted_suite

        return build_scripted_suite(task, run_dir, control=control, ledger=ledger)
    if backend == "meta":
        if plant:
            from nightshift.agents.scripted import planted_proposals

            planted_proposals(task, plant)          # raises ValueError before any client exists
        from nightshift.agents.llm import build_llm_suite

        if routing is None:             # the builder needs a routing table; the defaults are the documented ones
            from nightshift.models.config import load_routing

            routing = load_routing(None)
        extra = {"background": background} if background else {}
        suite = build_llm_suite(task, run_dir, routing, gate, ledger, implementer=implementer, **extra)
        if plant:
            from nightshift.agents.scripted import build_scripted_suite

            suite.planted = build_scripted_suite(task, run_dir, control=False, ledger=ledger)
        return suite
    shown = ascii(backend[:24]) if isinstance(backend, str) else f"<{type(backend).__name__}>"
    raise ValueError(f"unknown backend {shown}; choose one of {', '.join(BACKENDS)}")
