"""Test doubles for the outer loop (owner 4): a scripted ``ProposalSource`` and an in-memory ``PackStore``.

``ScriptedProposalSource`` replays a list of replies (a ``Proposal``, a ``ProposalResult``, an exception to raise, or a callable that
gets the ``ProposalContext``) and keeps every context it was given, so a test can read the exact prompt the proposer would have seen.

``InMemoryPackStore`` implements ``PackStore`` with the same STRICT apply as the real ``PackRepo`` (it calls
``packs.apply_proposal_edits``), fake commit ids derived from the content, and a lint that passes unless the test says otherwise.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from nightshift.lab import packs
from nightshift.lab.contracts import (
    ContractError,
    LintResult,
    PackRef,
    Proposal,
    ProposalContext,
    ProposalResult,
)

REPO_PATH = "packs/pack.git"


class ScriptedProposalSource:
    def __init__(self, script: Sequence[Any], *, model: str = "muse-wonder-1.0", usd: float = 0.05):
        self._script = list(script)
        self.model, self.usd = model, usd
        self.contexts: list[ProposalContext] = []

    def propose(self, context: ProposalContext) -> ProposalResult:
        self.contexts.append(context)
        if not self._script:
            raise AssertionError("the proposer script is exhausted")
        item = self._script.pop(0)
        if callable(item):
            item = item(context)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, ProposalResult):
            return item
        if isinstance(item, Proposal):
            return ProposalResult(item, self.model, self.model, False, self.usd, 1000, 500, json.dumps(item.to_dict()))
        raise AssertionError(f"unusable script item {item!r}")


class InMemoryPackStore:
    """A ``PackStore`` without git. ``lint_messages[label]`` makes the lint of that candidate fail with those messages."""

    def __init__(self, files: Optional[Mapping[str, str]] = None, *, editable_paths: Sequence[str] = packs.DEFAULT_EDITABLE_PATHS):
        self._files: dict[str, dict[str, str]] = {}
        self._parent: dict[str, Optional[str]] = {}
        self._tags: dict[str, str] = {}
        self.main: Optional[str] = None
        self.lint_messages: dict[str, tuple] = {}
        self.accepted: list[str] = []
        self._is_editable = packs.make_is_editable(editable_paths)
        self._initial = dict(files) if files is not None else {"task.yaml": "task:\n  name: x\n", "notes/seed.md": "# notes\n\n## Log\n"}

    @staticmethod
    def _id(files: Mapping[str, str], parent: Optional[str], label: str) -> str:
        return hashlib.sha1(json.dumps([sorted(files.items()), parent, label]).encode()).hexdigest()

    def _ref(self, commit: str) -> PackRef:
        return PackRef(REPO_PATH, commit)

    def _get(self, pack: PackRef) -> dict[str, str]:
        if pack.commit not in self._files:
            raise ContractError(f"unknown pack commit {pack.commit[:12]}")
        return self._files[pack.commit]

    def init_v0(self, template_dir: Any = None) -> PackRef:
        if self.main is not None:
            raise ContractError("already has a v0")
        c = self._id(self._initial, None, "v0")
        self._files[c], self._parent[c], self._tags["v0"], self.main = dict(self._initial), None, c, c
        return self._ref(c)

    def checkout(self, pack: PackRef, dest: Any) -> None:
        d = Path(dest)
        if d.exists() and any(d.iterdir()):
            raise ContractError("destination is not empty")
        for path, text in self._get(pack).items():
            (d / path).parent.mkdir(parents=True, exist_ok=True)
            (d / path).write_text(text, encoding="utf-8")

    def tree_sha256(self, pack: PackRef) -> str:
        return packs.listing_sha256({p: t.encode() for p, t in self._get(pack).items()})

    def apply(self, proposal: Proposal, parent: PackRef, version_label: str) -> tuple[PackRef, str]:
        if version_label in self._tags:
            raise ContractError(f"tag pack/{version_label} already exists")
        old = self._get(parent)
        new = packs.apply_proposal_edits(old, proposal, is_editable=self._is_editable)
        c = self._id(new, parent.commit, version_label)
        self._files[c], self._parent[c], self._tags[version_label] = new, parent.commit, c
        diff = "".join(f"--- a/{p}\n+++ b/{p}\n" for p in sorted(set(new) | set(old)) if new.get(p) != old.get(p))
        return self._ref(c), diff

    def lint(self, pack: PackRef) -> LintResult:
        label = next((k for k, v in self._tags.items() if v == pack.commit), "")
        msgs = self.lint_messages.get(label)
        parent = self._parent.get(pack.commit)
        changed = tuple(sorted(p for p in self._get(pack) if parent is None or self._files[parent].get(p) != self._get(pack)[p]))
        return LintResult(False, msgs, changed) if msgs else LintResult(True, (), changed)

    def accept(self, pack: PackRef, version_label: str) -> None:
        self._get(pack)
        if self.main == pack.commit:
            return  # idempotent like PackRepo.accept: a repeat for the current main changes nothing
        self.main = pack.commit
        self.accepted.append(version_label)


# --------------------------------------------------------------------------------------------------
# a small lab for the outer-loop tests: real LabLedger, FoldReader and stats (owner 6 and owner 3), scripted trials
# --------------------------------------------------------------------------------------------------
H64 = "a" * 64
H40 = "b" * 40
BASE_COMMIT = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
SCORER_VERSION = "sv1-0123456789abcdef"
EVO, HELD = "a-evo-1", "a-held-1"


class FakeClock:
    def __init__(self, start: float = 1_793_000_000.0, step: float = 1.0):
        self.now, self.step = start, step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def make_budget():
    from nightshift.lab.contracts import BudgetSpec

    return BudgetSpec(researcher_tokens=2_000_000, gpu_minutes=120.0, dollars=25.0, wall_seconds=7200.0)


def make_shape():
    from nightshift.lab.contracts import Shape

    return Shape("a100", 1, 12, 300)


def make_base():
    from nightshift.lab.contracts import BaseModelRef

    return BaseModelRef("meta-models/Muse-Glimmer-30B", BASE_COMMIT)


def make_manifest(**kw):
    from nightshift.lab.contracts import Assignment, AssignmentManifest, AssignmentRole, ItemMode, MixedRowPolicy

    def revs(c: str) -> dict:
        return {"train_sha256": c * 64, "dev_sha256": chr(ord(c) + 1) * 64, "selection_sha256": chr(ord(c) + 2) * 64}

    base = dict(
        assignments=(
            Assignment(EVO, AssignmentRole.EVOLUTION, ("count", "words"), revs("1")),
            Assignment(HELD, AssignmentRole.HELDOUT, ("copy", "firstlast"), revs("4")),
        ),
        evolution_families=("count", "words"),
        heldout_families=("copy", "firstlast"),
        mixed_row_policy=MixedRowPolicy.DROPPED,
        item_mode=ItemMode.SALTED_DRAW,
        source_revisions={
            "ifrlvr_parquet_sha256": "7" * 64,
            "ifbench_commit": "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d",
            "open_instruct_commit": "b781e782a8fc530577f235aefe282dd046cb8ad2",
            "split_salt_sha256": "9" * 64,
        },
        created_utc="2026-11-01T12:00:00+00:00",
    )
    base.update(kw)
    return AssignmentManifest(**base)


def make_rule(**kw):
    from nightshift.lab.contracts import FailedTrialPolicy, GenerationParams, Rule, Split, SpreadEstimate

    base = dict(
        metric="prompt_all",
        split=Split.SELECTION,
        scorer_version=SCORER_VERSION,
        assignments=(EVO,),
        spread=SpreadEstimate(0.02, 4, ("t-0001", "t-0002", "t-0003", "t-0004", "t-0005"), 1),
        alpha=0.05,
        min_gain=0.0,
        score_seeds=(0, 1, 2),
        score_params=GenerationParams.recommended(8000),
        screen_batch_size=3,
        screen_trials_per_candidate=2,
        confirm_trials_per_candidate=3,
        confirm_trials_per_incumbent=3,
        variance_guard_ratio=1.5,
        failed_trial_policy=FailedTrialPolicy.INVALIDATES_CANDIDATE,
        assumed_gain=0.12,
        minimum_detectable_gain=0.04,
        degenerate_override=None,
        max_candidates=6,
        stop_after_rejections=None,
        repeated_decisions_note="Screen then confirm; the selection results are an adaptive search trace, not evidence of a gain.",
        frozen_utc="2026-11-01T12:00:00+00:00",
    )
    base.update(kw)
    return Rule(**base)


def make_proposal(path: str = "notes/seed.md", old: str = "## Log", new: str = "## Log (newest first)", *, lever=None, summary: str = "bound the notes"):
    from nightshift.lab.contracts import EditOp, Lever, PackEdit, Proposal, SearchReplace

    return Proposal(
        hypothesis="Researchers re-read the whole log every turn.",
        lever=lever or Lever.CONTEXT_NOTES,
        change_summary=summary,
        expected_effect="Less context per turn.",
        risk="Loses detail.",
        edits=(PackEdit(path, EditOp.SEARCH_REPLACE, None, (SearchReplace(old, new),)),),
    )


class OuterLab:
    """A lab directory with a real master ledger, a frozen rule/manifest/budget, the terms gate, a v0 pack and (optionally) the v0
    spread trials. ``run_trial`` is the scripted ``TrialExecutor``: ``value_fn(spec) -> float | None`` is the selection score of the
    trial (None: the trial failed with no submission)."""

    def __init__(self, tmp: str | os.PathLike, *, rule=None, with_gate: bool = True, with_spread: bool = True, value_fn=None, store=None, store_factory=None):
        from nightshift.lab.contracts import (
            FrozenFile, Gate, GateOutcome, GateRecord, LabInit, LabKind, LabLayout, PackCommitted, SpreadMeasured, SpreadPlan,
            TrialScore, comparability_key, write_record,
        )
        from nightshift.lab.fold import FoldReader
        from nightshift.lab.ledger import LabLedger

        self.clock = FakeClock()
        self.layout = LabLayout(Path(tmp) / "lab")
        self.layout.root.mkdir(parents=True)
        self.ledger = LabLedger.create(self.layout.master_ledger_dir, clock=self.clock, fsync=False)
        self.reader = FoldReader(self.layout, ledger=self.ledger)
        self.store = store or (store_factory(self.layout) if store_factory else InMemoryPackStore())
        self.budget, self.manifest = make_budget(), make_manifest()
        self.rule = rule or make_rule()
        self.value_fn = value_fn or (lambda spec: 0.7)
        self.executed: list = []
        L = LabKind
        self.ledger.append(L.INITIALIZED, "coordinator", LabInit("lab1", "2026-11-01T12:00:00+00:00", "ax43"))
        for kind, name, path, rec in (
            (L.ASSIGNMENTS_FROZEN, "assignments", self.layout.manifest_path, self.manifest),
            (L.BUDGET_FROZEN, "budget", self.layout.budget_path, self.budget),
        ):
            write_record(path, rec)
            self.ledger.append(kind, "coordinator", FrozenFile(name, self.layout.file_ref(path)))
        if with_gate:
            self.ledger.append(L.GATE_RECORDED, "coordinator", GateRecord(Gate.TERMS_10_1_IX, GateOutcome.CLARIFIED, "cleared in writing", "owner", "2026-11-01T12:00:00+00:00"))
        self.v0 = self.store.init_v0()
        self.ledger.append(L.PACK_COMMITTED, "coordinator", PackCommitted("v0", self.v0, None, None))
        if with_spread:
            self.ledger.append(L.SPREAD_PLANNED, "coordinator", SpreadPlan("prompt_all", SCORER_VERSION, (EVO,), 5, "2026-11-01T12:00:00+00:00"))
            for seed in range(5):
                self.run_trial(self.make_spec("v0", self.v0, EVO, seed))
            state = self.reader.state()
            ts = [TrialScore(t, EVO, state.trials[t].selection_score_id, "prompt_all", SCORER_VERSION, comparability_key(state.trials_started[t], "prompt_all"),
                             state.scores[state.trials[t].selection_score_id].headline_mean()) for t in state.trials]
            self.ledger.append(L.SPREAD_MEASURED, "coordinator", SpreadMeasured("prompt_all", SCORER_VERSION, 5, 0, tuple(ts), self.rule.spread))
        write_record(self.layout.rule_path, self.rule)
        self.ledger.append(L.RULE_FROZEN, "coordinator", FrozenFile("rule", self.layout.file_ref(self.layout.rule_path)))

    # -- specs and the scripted executor
    def make_spec(self, label: str, pack, assignment: str, seed: int):
        from nightshift.lab.contracts import Phase, TrialSpec

        state = self.reader.state()
        return TrialSpec(
            trial_id=self.reader.next_id(state, "t"), phase=Phase.SEARCH, arm=None, pack_version=label, pack=pack, assignment_id=assignment,
            seed=seed, budget=self.budget, shape=make_shape(), base=make_base(),
            data_revisions=dict(self.manifest.get(assignment).data_revisions), scorer_version=SCORER_VERSION,
            score_params=self.rule.score_params, score_seeds=self.rule.score_seeds, final_request_id=None,
        )

    def run(self, spec):
        return self.run_trial(spec)

    def run_trial(self, spec):
        from nightshift.lab.contracts import (
            AdapterFile, AdapterRef, BudgetUsage, FileRef, GenerationParams, LabKind, Phase, ScoreRecord, SeedAggregate, Split,
            Submission, TrialRecord, TrialStatus, Verifier, utc_iso,
        )

        self.executed.append(spec)
        self.ledger.append(LabKind.TRIAL_STARTED, "coordinator", spec)
        value = self.value_fn(spec)
        state = self.reader.state()
        now = utc_iso(self.clock())
        used = BudgetUsage(1_000_000, 60.0, 3.0, 4.0, 1800.0)
        if value is None:
            rec = TrialRecord(spec, TrialStatus.FAILED, now, now, None, used, None, None, None, None, None, None, None, {}, "scripted failure")
            self.ledger.append(LabKind.TRIAL_FINISHED, f"trial:{spec.trial_id}", rec)
            return rec
        aid = self.reader.next_id(state, "ad")
        adapter = AdapterRef.from_files(aid, [AdapterFile("adapter_config.json", "1" * 64, 1000), AdapterFile("adapter_model.safetensors", __import__("hashlib").sha256(aid.encode()).hexdigest(), 5000)])

        def record(split, score_id, v):
            return ScoreRecord(
                score_id=score_id, request_id=score_id, phase=Phase.SEARCH, assignment_id=spec.assignment_id, split=split, trial_id=spec.trial_id,
                arm=None, final_request_id=None, base=spec.base, adapter_sha256=adapter.sha256, scorer_version=SCORER_VERSION,
                verifier=Verifier.IFEVALG, verifier_pins={"ifevalg_instructions_py": "dfc4629a"}, params=spec.score_params,
                per_seed=tuple(SeedAggregate(s, 100, {"prompt_all": v, "prompt_fraction": min(1.0, v + 0.05)}, 2, 0) for s in spec.score_seeds),
                n_prompts=100, per_prompt=FileRef(f"scores/{score_id}/per_prompt.jsonl", H64, 10), headline_metric="prompt_all", created_utc=now,
            )

        dev_id = self.reader.next_id(state, "sc")
        sel_id = self.reader.next_id(state, "sc", taken=[dev_id])
        self.ledger.append(LabKind.SCORE_RECORDED, "scorer", record(Split.DEV, dev_id, min(1.0, value + 0.1)))
        self.ledger.append(LabKind.SCORE_RECORDED, "scorer", record(Split.SELECTION, sel_id, value))
        rec = TrialRecord(
            spec, TrialStatus.COMPLETED, now, now, H64, used, Submission(aid, H64, H40, None), adapter, dev_id, sel_id, None, None, None,
            {"isolation": "test"}, None,
        )
        self.ledger.append(LabKind.TRIAL_FINISHED, f"trial:{spec.trial_id}", rec)
        return rec
