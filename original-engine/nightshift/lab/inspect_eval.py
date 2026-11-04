"""Inspect AI (UK AISI) task for the lab's IFEvalG-scored splits: the experiment runner with a pinned, versioned dataset.

``inspect_ai`` is imported inside the functions, so the rest of the lab does not need it.

What this module fixes, so a model evaluation cannot drift:

* **Dataset hygiene.** The dataset is read from the lab root only (``data/assignments/<id>/dev.jsonl``), its sha256 must equal the revision frozen in
  the assignment manifest, rows follow the same strict checks as ``ScoreService`` (unique keys, non-empty prompt, ground truth present), and the
  hash goes into the task metadata and the sample ids. Only the public DEV split is available here: selection, held-out and test data stay behind
  the ledger guard of the ``score`` command, and this module never opens them.
* **Scoring.** The pinned ``IfEvalG`` backend of ``nightshift.lab.verifiers`` (the same one the ``score`` command uses). A sample's value holds
  ``prompt_all`` (every constraint passed), ``fraction`` and ``truncated``; the answer text is the final answer only, cleaned as the harness
  generator cleans it.
* **Decoding.** ``GenerationParams.recommended`` plus the ``reasoning_strength`` chat-template field, one explicit system message, a fixed seed.
* **Versioning.** The task carries ``version=TASK_VERSION``, bumped whenever this module changes what is asked or how it is scored. Every run
  records the git commit (and whether the tree was dirty), the lab's ``scorer_version``, the verifier pins, the dataset sha256, the base-model
  revision, the adapter listing hash, the library versions and the decoding settings in the eval log's metadata.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from nightshift.lab.contracts import (
    AssignmentManifest,
    ContractError,
    GenerationParams,
    LabLayout,
    ScorerError,
    loads_strict,
)

TASK_VERSION = "1"  # bump when the prompt, system message, decoding, answer cleaning or scoring rule changes
TASK_NAME = "glimmer_ifevalg"


def _clean(text: str) -> str:
    return text.replace("\x00", "").encode("utf-8", errors="replace").decode("utf-8")


def load_dev_rows(layout: LabLayout, assignment_id: str) -> tuple[list, str]:
    """The frozen DEV rows of one assignment and their sha256. Refuses a file that differs from the manifest's revision."""
    manifest = _manifest(layout)
    assignment = manifest.get(assignment_id)
    path = layout.dev_data(assignment_id)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    want = assignment.data_revisions["dev_sha256"]
    if digest != want:
        raise ScorerError(f"{path} has sha256 {digest}, but the frozen manifest says {want}; the dataset was changed after the freeze")
    rows, seen = [], set()
    for n, line in enumerate(raw.decode("utf-8").split("\n"), 1):  # not splitlines(): U+2028 may sit inside a JSON string
        if not line.strip():
            continue
        row = loads_strict(line)
        if not isinstance(row, dict) or any(k not in row for k in ("key", "prompt", "ground_truth", "instruction_ids")):
            raise ScorerError(f"{path} line {n}: a dev row needs key, prompt, ground_truth, instruction_ids")
        row["key"] = str(row["key"])
        if not row["key"] or row["key"] in seen or not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise ScorerError(f"{path} line {n}: empty or duplicate key, or empty prompt")
        seen.add(row["key"])
        rows.append(row)
    if not rows:
        raise ScorerError(f"{path} has no rows")
    return rows, digest


def _manifest(layout: LabLayout) -> AssignmentManifest:
    from nightshift.lab.contracts import read_record

    return read_record(layout.manifest_path, AssignmentManifest)


def git_state(repo: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def provenance(layout: LabLayout, assignment_id: str, params: GenerationParams, *, dataset_sha256: str, scorer_version: str, pins: dict,
               base_revision: str, adapter_sha256: Optional[str], rental_id: Optional[str], repo: Path) -> dict:
    import inspect_ai

    return {
        "task_version": TASK_VERSION, "lab_id": layout.root.name, "assignment_id": assignment_id, "split": "dev",
        "dataset_sha256": dataset_sha256, "scorer_version": scorer_version, "verifier_pins": pins,
        "base_revision": base_revision, "adapter_sha256": adapter_sha256, "rental_id": rental_id,
        "decoding": params.to_dict(), "git": git_state(repo),
        "versions": {"inspect_ai": inspect_ai.__version__, "python": sys.version.split()[0]},
    }


def ifevalg_scorer(backend: Any):
    """Inspect scorer around a pinned ``IfEvalG`` backend; value: prompt_all (0/1), fraction, truncated (0/1)."""
    from inspect_ai.scorer import Score, mean, scorer, stderr

    @scorer(metrics={"prompt_all": [mean(), stderr()], "fraction": [mean()], "truncated": [mean()]})
    def _scorer():
        async def score(state, target):
            row = dict(state.metadata["row"])
            truncated = state.output.stop_reason == "max_tokens"
            text = "" if truncated else _clean(state.output.completion or "")  # the harness rule (scorer.py): a truncated generation fails in every arm
            res = backend.score([row], [text], mode="strict")[0]
            value = {"prompt_all": 1 if res.passed_strict else 0, "fraction": float(res.fraction), "truncated": 1 if truncated else 0}
            return Score(value=value, answer=text[:200], metadata={"constraints": [{"id": i, "passed": f} for i, f in zip(res.instruction_ids, res.strict)],
                                                                 "error": res.error})
        return score

    return _scorer()


def build_task(layout: LabLayout, assignment_id: str, backend: Any, meta: dict, params: GenerationParams, *, solver: Any = None, limit_note: str = ""):
    """The Inspect ``Task``. ``solver`` replaces generation (the replay check uses it); the default asks the model."""
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import GenerateConfig
    from inspect_ai.solver import generate, system_message

    rows, digest = load_dev_rows(layout, assignment_id)
    samples = [Sample(id=r["key"], input=r["prompt"], target="", metadata={"row": r}) for r in rows]
    dataset = MemoryDataset(samples, name=f"{layout.root.name}:{assignment_id}:dev:{digest[:16]}", location=str(layout.dev_data(assignment_id)))
    config = GenerateConfig(temperature=params.temperature, top_p=params.top_p, max_tokens=params.max_tokens, seed=0,
                            extra_body={"top_k": params.top_k, "chat_template_kwargs": {"reasoning_strength": params.reasoning_strength}})  # top_k is not an OpenAI field: the provider drops it unless sent here
    return Task(
        dataset=dataset, solver=solver or [system_message(params.system_prompt), generate()], scorer=ifevalg_scorer(backend), config=config,
        name=TASK_NAME, version=TASK_VERSION, metadata={**meta, "note": limit_note},
    )
