"""Workflow composition (docs/ARCHITECTURE.md 7.1): which stages a lane runs, and in which order.

Guarantees
----------
* ``validate_composition`` never raises and never executes anything: it returns readable problems (empty list = valid).
  The stage list comes from a model, so every value echoed in a message is passed through ``ascii()`` and capped.
* ``compose`` returns the ``StageSpec`` objects in the given order, or raises ``types.InvalidComposition``.
* Mandatory stages (implement, run, grade, review, record) can be neither removed nor reordered. Optional stages are
  ``design`` (only before ``implement``), ``review_hard`` and ``repair`` (only after ``review``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..types import InvalidComposition

STAGE_NAMES: tuple[str, ...] = ("design", "implement", "run", "grade", "review", "review_hard", "repair", "record")
MANDATORY: tuple[str, ...] = ("implement", "run", "grade", "review", "record")
OPTIONAL: tuple[str, ...] = tuple(s for s in STAGE_NAMES if s not in MANDATORY)
DEFAULT_STAGES: list[str] = ["design", "implement", "run", "grade", "review", "repair", "record"]

ROLE_OF: dict[str, str] = {
    "design": "designer", "implement": "implementer", "run": "engine", "grade": "engine", "review": "reviewer",
    "review_hard": "reviewer_hard", "repair": "implementer", "record": "engine",
}

MAX_STAGES = 16        # a composition longer than this is rejected without looking at the rest
MAX_ERRORS = 8         # bound on the number of messages returned
_SHOW_CHARS = 24


@dataclass(frozen=True)
class StageSpec:
    """One step of a lane: its name, the role (agent or engine) that performs it, and whether it can be omitted."""

    name: str
    role: str
    mandatory: bool


def _spec(name: str) -> StageSpec:
    return StageSpec(name=name, role=ROLE_OF[name], mandatory=name in MANDATORY)


TEMPLATES: dict[str, list[StageSpec]] = {"explore": [_spec(n) for n in DEFAULT_STAGES]}


def _show(value: Any) -> str:
    """A model-supplied value rendered safely for a message: ASCII-escaped, quoted and capped."""
    if not isinstance(value, str):
        return f"<{type(value).__name__}>"
    text = ascii(value[:_SHOW_CHARS])
    return text + ("..." if len(value) > _SHOW_CHARS else "")


def validate_composition(stages: Any) -> list[str]:
    """Return the rule violations of a stage-name list (empty means valid). Pure; never raises.

    R1 names are known strings and appear once. R2 ``implement`` is followed by ``run, grade, review, record`` in that
    order and ``design`` (if present) comes first. R3 ``record`` is last. R4 ``review_hard`` and ``repair`` only after
    ``review``. R5 every mandatory stage is present.
    """
    if not isinstance(stages, (list, tuple)):
        return [f"stages must be a list of stage names, got {_show(stages)}"]
    if not stages:
        return ["stages is empty; the mandatory stages are " + ", ".join(MANDATORY)]
    if len(stages) > MAX_STAGES:
        return [f"too many stages ({len(stages)}); a lane has at most {len(STAGE_NAMES)} distinct stages"]

    errors: list[str] = []
    names: list[str] = []
    seen: set[str] = set()
    for item in stages:
        if not isinstance(item, str) or item not in STAGE_NAMES:
            errors.append(f"unknown stage {_show(item)}; known stages are " + ", ".join(STAGE_NAMES))
        elif item in seen:
            errors.append(f"stage {_show(item)} appears more than once")
        else:
            seen.add(item)
            names.append(item)
    if errors:
        return errors[:MAX_ERRORS]

    missing = [s for s in MANDATORY if s not in seen]
    if missing:
        errors.append("mandatory stage(s) missing: " + ", ".join(missing))
    pos = {s: i for i, s in enumerate(names)}
    if "record" in pos and pos["record"] != len(names) - 1:
        errors.append("'record' must be the last stage")
    mandatory_present = [s for s in MANDATORY if s in pos]
    if [s for s in names if s in MANDATORY] != mandatory_present:
        errors.append("mandatory stages must appear in the order " + ", ".join(MANDATORY))
    if "design" in pos and "implement" in pos and pos["design"] > pos["implement"]:
        errors.append("'design' must come before 'implement'")
    for optional in ("review_hard", "repair"):
        if optional in pos:
            if "review" not in pos:
                errors.append(f"'{optional}' needs a 'review' stage before it")
            elif pos[optional] < pos["review"]:
                errors.append(f"'{optional}' is only valid after 'review'")
    return errors[:MAX_ERRORS]


def compose(stages: Any) -> list[StageSpec]:
    """The ``StageSpec`` list for a valid composition, in the given order. Raises ``InvalidComposition`` otherwise."""
    errors = validate_composition(stages)
    if errors:
        raise InvalidComposition(errors)
    return [_spec(name) for name in stages]
