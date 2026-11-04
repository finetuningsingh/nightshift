"""``build_fake_world(tmp) -> FakeWorld``: the dependency container of ``cli.main(argv, world=...)`` built from the five owners' fakes.

What is REAL in this world: the ledgers (``LabLedger``), the fold (``FoldReader``), the budget meters and the lab-wide cap, the scorer
(``ScoreService`` with the ledger-derived guard from ``cli.make_score_guard``, the access table, the data-file hash checks and the real
score files), ``score_session.score_adapter``, the contamination scan, the trial runner with its metering proxy and control API (on a real
loopback port), the statistics (``stats`` as the ``DecisionRule``), the outer controller and the strict pack apply, the report.

What is FAKE: the researcher runtime (owner 5's ``FakeRuntime``: a scripted researcher that submits one recipe through the control API),
the trainer (owner 2's ``FakeTrainer``: it writes a real adapter and a real dev score), the model server and the generator and the
verifiers (owner 1's fakes), the proposer (owner 4's ``ScriptedProposalSource``) and the pack store (``InMemoryPackStore``), the GPU
provider behind the ``rentals`` command (``FakeProvider``).

The numbers a selection or test score takes are decided by ``world.controls`` (a function of the pack version, the trial seed and the
generation seed), so a test can make a candidate lose its screening, win and be confirmed, or win the screen and fail the confirmation.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from nightshift.lab import stats
from nightshift.lab.budget import LabCap, TrialBudgetMeter
from nightshift.lab.cli import CliWorld, LazyScorer, make_score_guard
from nightshift.lab.contamination import build_scanner, scan_job
from nightshift.lab.contracts import (
    AssignmentManifest,
    BaseModelRef,
    LabConfig,
    LabLayout,
    Phase,
    ScoreRequest,
    Shape,
    Split,
    TrialSpec,
    Verifier,
    load_frozen,
)
from nightshift.lab.fold import FoldReader
from nightshift.lab.gpu_rentals import RentalManager
from nightshift.lab.ledger import LabLedger
from nightshift.lab.scorer import ScoreService
from nightshift.lab.trial import TrialRunner
from tests.support.lab_fake_generator import FakeGenerator, FakeModelServer, FakeVerifierBackend
from tests.support.lab_fake_gpu import FakeProvider, FakeTrainer, FakeWatchdogLauncher, fake_key_factory
from tests.support.lab_fake_proposer import InMemoryPackStore, ScriptedProposalSource
from tests.support.lab_fake_runtime import FakeRuntime

BASE_COMMIT = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
IFBENCH_COMMIT = "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d"
OPEN_INSTRUCT_COMMIT = "b781e782a8fc530577f235aefe282dd046cb8ad2"
FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "lab" / "ifbench_mini"
META_KEY = "FAKE_META_KEY"
THUNDER_KEY = "FAKE_THUNDER_KEY"


class FakeClock:
    """Thread-safe, strictly increasing epoch seconds: every call advances by ``step`` (so trials have a wall time and events have order)."""

    def __init__(self, start: float = 1_793_000_000.0, step: float = 0.001):
        self.now, self.step = start, step
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            self.now += self.step
            return self.now

    def advance(self, seconds: float) -> float:
        with self._lock:
            self.now += seconds
            return self.now


class Controls:
    """What a score is. ``selection_value(spec, split, gen_seed)`` is the fraction of prompts that pass for one trial and one generation
    seed. ``effect[pack_version]`` is added to ``base`` (so ``effect['cand-002'] = 0.25`` makes that pack 0.25 better than v0);
    ``arm_value`` sets the test-split level of each arm. The per-trial wobble is deterministic (trial seed and generation seed), so the
    spread of v0 is not zero."""

    def __init__(self) -> None:
        self.base = 0.5
        self.effect: dict = {}
        self.wobble = 0.125
        self.arm_value = {"stock": 0.30, "v0": 0.40, "evolved": 0.55}
        self.heldout_effect = {"v0": 0.0, "evolved": 0.15}
        self.calls: list = []

    def selection_value(self, spec: Optional[TrialSpec], split: Split, gen_seed: int) -> float:
        if spec is None:  # arm 1 (stock) has no trial
            return self.arm_value["stock"]
        phase = (spec.seed + gen_seed) % 6  # one prompt better or worse in one generation seed of three: trial means move by 1/3 prompt
        wob = self.wobble * (1.0 if phase == 0 else (-1.0 if phase == 3 else 0.0))
        if spec.phase is Phase.FINAL:
            arm = spec.arm.value if spec.arm is not None else "v0"
            level = self.arm_value[arm] if split is Split.TEST else self.base + self.heldout_effect[arm]
            return min(1.0, max(0.0, level + wob))
        return min(1.0, max(0.0, self.base + self.effect.get(spec.pack_version, 0.0) + wob))


class ControlledScorer:
    """Wraps the REAL ``ScoreService``: remembers which request the calling thread is scoring so the fake generator can answer as that
    trial's adapter would. Everything else (access table, guard, hash checks, files) is the real thing."""

    def __init__(self, inner: ScoreService, tls: threading.local):
        self._inner, self._tls = inner, tls

    @property
    def scorer_version(self) -> str:
        return self._inner.scorer_version

    def precheck(self, request: ScoreRequest) -> None:
        self._inner.precheck(request)

    def score(self, request: ScoreRequest, generator: Any, model: str) -> Any:
        self._tls.request = request
        try:
            return self._inner.score(request, generator, model)
        finally:
            self._tls.request = None


class FakeWorld(CliWorld):
    """A ``CliWorld`` that also names the fakes, so a test can script and inspect them."""

    controls: Any = None
    fakes: Any = None


class _Fakes:
    """Plain holder: ``trainer``, ``runtime``, ``model_server``, ``proposer``, ``packs``, ``generator``, ``provider``."""


def build_fake_world(tmp: str | Path, *, lab_budget_usd: float = 200.0, affordable_trials: int = 60, upstream_base_url: str = "http://127.0.0.1:9/v1") -> FakeWorld:
    tmp = Path(tmp)
    layout = LabLayout(tmp / "lab")
    layout.root.mkdir(parents=True, exist_ok=True)
    clock = FakeClock()
    config = LabConfig(
        lab_id="lab-fake", control_host="control.invalid", researcher_host="researcher.invalid", researcher_user="lab",
        base=BaseModelRef("meta-models/Muse-Glimmer-30B", BASE_COMMIT), shape=Shape("a100", 1, 12, 300),
        pins={"coral_commit": "0123dfb939b35228cf2c1fde224cd0561e727408", "opencode_version": "1.19.3", "ifbench_commit": IFBENCH_COMMIT,
              "open_instruct_commit": OPEN_INSTRUCT_COMMIT},
        secret_env={"meta_api_key": META_KEY, "thunder_api_key": THUNDER_KEY}, upstream_base_url=upstream_base_url,
        lab_budget_usd=lab_budget_usd, affordable_trials=affordable_trials,
    )
    ledger = LabLedger.create(layout.master_ledger_dir, clock=clock, fsync=False)
    reader = FoldReader(layout, ledger=ledger)
    lab_cap = LabCap(config.lab_budget_usd, 0.0, clock=clock)
    controls = Controls()
    tls = threading.local()
    tls.request = None
    key_cache: dict = {}

    def keys_of(path: Path) -> list:
        if path not in key_cache:
            key_cache[path] = [json.loads(line)["key"] for line in path.read_text().splitlines() if line.strip()]
        return key_cache[path]

    def respond(model: str, prompt: Any, params: Any, gen_seed: int) -> str:
        request: Optional[ScoreRequest] = getattr(tls, "request", None)
        if request is None:
            return "OK"
        if request.split is Split.TEST:
            path = layout.test_data()
        elif request.split is Split.SELECTION:
            path = layout.selection_data(request.assignment_id)
        else:
            path = layout.dev_data(request.assignment_id)
        keys = keys_of(path)
        spec = reader.state().trials_started.get(request.trial_id) if request.trial_id else None
        value = controls.selection_value(spec, request.split, gen_seed)
        controls.calls.append((request.request_id, request.split.value, gen_seed, value))
        k = int(round(value * len(keys)))
        return "OK" if keys.index(prompt.prompt_id) < k else "NO"

    def flags(row: Any, text: str) -> list:
        ids = row.get("instruction_id_list") if row.get("instruction_id_list") is not None else row.get("instruction_ids")
        return [text == "OK"] * len(ids)

    generator = FakeGenerator(respond)

    def generator_factory(endpoint: Any) -> Any:
        return generator

    def make_scorer() -> Any:
        manifest = load_frozen(layout, reader.state().manifest, AssignmentManifest)
        verifiers = {
            Verifier.IFEVALG: FakeVerifierBackend(Verifier.IFEVALG, rule=flags, pins={"fake_ifevalg": "1"}),
            Verifier.IFBENCH_UPSTREAM: FakeVerifierBackend(Verifier.IFBENCH_UPSTREAM, rule=flags, pins={"fake_ifbench": "1"}),
        }
        service = ScoreService(layout, manifest, verifiers, guard=make_score_guard(layout, reader), clock=clock)
        return ControlledScorer(service, tls)

    scorer = LazyScorer(make_scorer)

    def new_trial_ledger(trial_id: str) -> Any:
        return LabLedger.create(layout.trial_ledger_dir(trial_id), clock=clock, fsync=False)

    def meter_factory(spec: TrialSpec, tledger: Any) -> Any:
        return TrialBudgetMeter(spec, tledger, spec.trial_id, clock=clock, lab_cap=lab_cap)

    scan_box: dict = {}

    def contamination_scan(**kw: Any) -> Any:
        if "scanner" not in scan_box:
            manifest = load_frozen(layout, reader.state().manifest, AssignmentManifest)
            scan_box["manifest"], scan_box["scanner"] = manifest, build_scanner(layout, manifest)
        return scan_job(layout, scan_box["manifest"], clock=clock, scanner=scan_box["scanner"], **kw)

    fakes = _Fakes()
    fakes.trainer = FakeTrainer(layout, clock=clock, dev_score=0.5, gpu_minutes=5.0)
    fakes.runtime = FakeRuntime(layout, clock=clock)
    fakes.model_server = FakeModelServer()
    fakes.proposer = ScriptedProposalSource([])
    fakes.packs = InMemoryPackStore()
    fakes.generator = generator
    fakes.provider = FakeProvider(clock)
    fakes.launcher = FakeWatchdogLauncher()
    rentals = RentalManager(layout.rental_ledger, fakes.provider, clock=clock, launcher=fakes.launcher, key_factory=fake_key_factory,
                            sleep=lambda s: clock.advance(s), stopper=fakes.launcher.stop)

    runner = TrialRunner(
        layout, ledger, new_trial_ledger, meter_factory, fakes.runtime, fakes.trainer, scorer, fakes.model_server, generator_factory, fakes.packs,
        reader, clock=clock, port_allocator=lambda: 0, upstream_base_url=config.upstream_base_url, api_key=lambda: "fake-meta-key-0123456789abcdef",
        contamination_scan=contamination_scan, poll_seconds=0.01,
    )
    world = FakeWorld(
        layout=layout, config=config, clock=clock, ledger=ledger, state_reader=reader, new_trial_ledger=new_trial_ledger,
        meter_factory=meter_factory, scorer=scorer, generator_factory=generator_factory, trainer=fakes.trainer, model_server=fakes.model_server,
        runtime=fakes.runtime, packs=fakes.packs, decision_rule=stats, trial_executor=runner, proposal_source=fakes.proposer, lab_cap=lab_cap,
        rentals=rentals, close=None,
    )
    world.controls = controls
    world.fakes = fakes
    return world


def fixture_args(world: FakeWorld, *, n_selection: int = 8) -> list:
    """The ``freeze-manifest`` arguments that split the synthetic ``ifbench_mini`` rows (never real IFBench rows)."""
    return [
        "--rows", str(FIXTURES / "rows.jsonl"), "--test-file", str(FIXTURES / "ifbench_test_mini.jsonl"),
        "--test-file", str(FIXTURES / "ifbench_test_legacy_mini.jsonl"),
        "--plan", f"a-evo-1:evolution:ifeval25,counting,format_new:{n_selection}:10:0", "--plan", "a-held-1:heldout:copy,firstlast:4:6:0",
        "--evolution-families", "ifeval25,counting,format_new,keywords_new", "--heldout-families", "copy,firstlast",
    ]


# --------------------------------------------------------------------------------------------------
# a driver for tests: run commands and walk the lab through its stages
# --------------------------------------------------------------------------------------------------
import io  # noqa: E402

from nightshift.lab import cli  # noqa: E402
from tests.support.lab_fake_proposer import make_proposal  # noqa: E402

STAGES = ("init", "manifest", "budget", "gate", "pack", "plan", "spread_trials", "spread", "rule", "search", "closed", "analysis",
          "final_v0", "final_evolved", "final_stock", "revealed")
BUDGET_ARGS = ["--researcher-tokens", "2000000", "--gpu-minutes", "120", "--dollars", "25", "--wall-seconds", "7200"]
RULE_ARGS = ["--assumed-gain", "0.2", "--max-candidates", "4", "--screen-batch-size", "2"]
# What the scripted lab does to the packs: cand-001 loses its screening; cand-002 wins and is confirmed; in round two cand-003 and cand-004
# are close to the new incumbent, one of them wins the screen and fails the confirmation, the other loses the screen.
EFFECTS = {"cand-001": -0.1, "cand-002": 0.25, "cand-003": 0.20, "cand-004": 0.26}


class LabDriver:
    """``run`` calls ``cli.main`` with the world; ``stage(name)`` runs every command up to that stage once."""

    def __init__(self, tmp: str | Path, *, skip: tuple = (), **world_kw: Any) -> None:
        self.skip = tuple(skip)
        self.world = build_fake_world(tmp, **world_kw)
        self.root = str(self.world.layout.root)
        self.done: list = []
        self.outputs: dict = {}
        self.search_outputs: list = []

    @property
    def layout(self) -> LabLayout:
        return self.world.layout

    def state(self) -> Any:
        return self.world.state_reader.state()

    def run(self, *argv: str) -> tuple:
        out, err = io.StringIO(), io.StringIO()
        rc = cli.main([self.root, *argv], world=self.world, out=out, err=err)
        return rc, out.getvalue(), err.getvalue()

    def ok(self, *argv: str) -> Any:
        rc, out, err = self.run(*argv)
        if rc != 0:
            raise AssertionError(f"{' '.join(argv)} -> {rc}: {err.strip()}")
        try:
            return json.loads(out)
        except ValueError:
            return out

    def refused(self, *argv: str) -> str:
        rc, out, err = self.run(*argv)
        if rc != 1:
            raise AssertionError(f"{' '.join(argv)} -> {rc} (wanted 1): {out[:200]} {err[:200]}")
        return err

    def stage(self, name: str) -> "LabDriver":
        for s in STAGES[: STAGES.index(name) + 1]:
            if s not in self.done:
                if s not in self.skip:
                    getattr(self, "_do_" + s)()
                self.done.append(s)
        return self

    def _do_init(self) -> None:
        self.ok("init")

    def _do_manifest(self) -> None:
        self.ok("freeze-manifest", *fixture_args(self.world))

    def _do_budget(self) -> None:
        self.ok("freeze-budget", *BUDGET_ARGS)

    def _do_gate(self) -> None:
        self.ok("record-gate", "--outcome", "clarified", "--statement", "cleared in writing for the test", "--decided-by", "owner")

    def _do_pack(self) -> None:
        self.ok("commit-pack")

    def _do_plan(self) -> None:
        self.ok("plan-spread", "--trials", "5")

    def _do_spread_trials(self) -> None:
        for _ in range(5):
            self.ok("trial")

    def _do_spread(self) -> None:
        self.outputs["spread"] = self.ok("spread")

    def _do_rule(self) -> None:
        self.outputs["rule"] = self.ok("freeze-rule", *RULE_ARGS)

    def script_search(self) -> None:
        self.world.controls.effect = dict(EFFECTS)
        self.world.fakes.proposer._script.extend([
            make_proposal(old="## Log", new="## Log A"), make_proposal(old="## Log", new="## Log B"),
            make_proposal(old="## Log B", new="## Log C"), make_proposal(old="## Log B", new="## Log D"),
        ])

    def _do_search(self) -> None:
        self.script_search()
        self.outputs["outer"] = self.ok("outer", "--rounds", "2", "--trial-llm-usd", "2")
        self.search_outputs = [self.ok("status"), self.run("report")[1]]

    def _do_closed(self) -> None:
        self.ok("close-search", "--reason", "the scripted search is over")

    def _do_analysis(self) -> None:
        self.ok("freeze-final-analysis", "--trials-per-pack", "5")

    def _do_final_v0(self) -> None:
        self.ok("final", "--arm", "v0")

    def _do_final_evolved(self) -> None:
        self.ok("final", "--arm", "evolved")

    def _do_final_stock(self) -> None:
        self.ok("final", "--arm", "stock")
        self.outputs["report_before_reveal"] = self.run("report")[1]

    def _do_revealed(self) -> None:
        self.outputs["reveal"] = self.ok("reveal")
