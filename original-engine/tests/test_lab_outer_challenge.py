"""Tests for nightshift.lab.outer_challenge (the bounded challenge of a proposed pack revision) and its use by the outer loop.

The recipe under test is the real starting pack (nightshift/lab/assets/pack_muse_v0). The critic and the proposer are fakes; nothing calls a
model, the network, a GPU or a shell. Hostile-looking strings are data handed to the pure checker, never executed.

Run: PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_lab_outer_challenge
"""

import json
import tempfile
import unittest
from pathlib import Path

from nightshift.lab import outer_challenge as oc
from nightshift.lab.contracts import (
    AccessDenied,
    ChallengeSummary,
    CandidateRecord,
    ContractError,
    EditOp,
    FileRef,
    Lever,
    Outcome,
    PackEdit,
    PackRef,
    Proposal,
    ProposalContext,
    ProposalResult,
    SearchReplace,
    dumps_record,
    loads_strict,
)
from nightshift.lab.proposer import ProposerUnavailable, UnparseableProposal
from tests.support.lab_fake_proposer import InMemoryPackStore, OuterLab, ScriptedProposalSource, make_proposal

PACK = Path(oc.__file__).resolve().parent / "assets" / "pack_muse_v0"
RUN, CFG = "seed/recipe/run.sh", "seed/recipe/config.yaml"
WONDER = "muse-wonder-full-1.0"


def v0_files() -> dict:
    files = {p.relative_to(PACK).as_posix(): p.read_text(encoding="utf-8") for p in sorted(PACK.rglob("*")) if p.is_file()}
    files["notes/seed.md"] += "\n## Log\n"  # the outer-loop fixtures append below this heading
    return files


def edit(files: dict, path: str, old: str, new: str) -> dict:
    assert files[path].count(old) == 1, (path, old)
    out = dict(files)
    out[path] = files[path].replace(old, new)
    return out


def sr(path: str, old: str, new: str, summary: str = "a change") -> Proposal:
    return Proposal("hypothesis text", Lever.OTHER, summary, "expected effect", "risk", (PackEdit(path, EditOp.SEARCH_REPLACE, None, (SearchReplace(old, new),)),))


MAX_STEPS = sr(CFG, "num_epochs: 2\n", "num_epochs: 2\nmax_steps: 300\n", "train two epochs")
THREE_EPOCHS = sr(CFG, "num_epochs: 2\n", "num_epochs: 3\n", "train three epochs")


def codes(findings, *, severity=None, new=None):
    return sorted({f.code for f in findings if (severity is None or f.severity == severity) and (new is None or f.new == new)})


class RenderTests(unittest.TestCase):
    def test_the_v0_recipe_renders_as_a_full_run_would_use_it(self):
        r = oc.recipes(v0_files())["seed/recipe"]
        self.assertEqual(r.config_path, CFG)
        self.assertEqual(r.shell["MAX_TOKENS"], "6000")  # the full-run default, not the SMOKE block's 1024
        self.assertEqual(r.shell["GEN_TIMEBOX_SECONDS"], "2700")
        self.assertEqual(r.shell["TRAIN_FILE"], "$DATA_DIR/train.jsonl")
        self.assertEqual((r.config["num_epochs"], r.config["micro_batch_size"], r.config["gradient_accumulation_steps"]), (2, 1, 4))
        self.assertIsNone(r.config["lora_model_dir"])
        self.assertNotIn("max_steps", r.config)  # the anchor is a comment
        self.assertFalse(any("max_steps" in l for l in r.lines), "the smoke-only sed is not part of a full run")
        d = r.dose()
        self.assertEqual(r.rows_bound(), 500)  # 250 prompts x min(4, 2) kept: an UPPER bound
        self.assertEqual((d["rows"], d["rows_kind"], d["batch"], d["steps"], d["steps_kind"], d["examples_seen"]), (500, "at_most", 4, 250, "at_most", 1000))
        self.assertEqual((d["epochs"], d["epochs_kind"]), (2.0, "exact"))  # by epochs: num_epochs is what trains, whatever the row count

    def test_the_dose_is_stated_with_its_bound_and_exact_from_a_known_row_count(self):
        v0 = v0_files()
        by_steps = oc.recipes(edit(v0, CFG, "num_epochs: 2\n", "num_epochs: 2\nmax_steps: 250\n"))["seed/recipe"]
        d = by_steps.dose()
        self.assertEqual((d["steps"], d["steps_kind"], d["examples_seen"]), (250, "exact", 1000))
        self.assertEqual(d["epochs_kind"], "at_least")  # fewer filtered rows than the bound means MORE passes
        self.assertAlmostEqual(d["epochs"], 2.0)
        self.assertIn(">= 2.00 epochs over <= 500 SFT rows", oc._dose_text(d))
        exact = by_steps.dose(rows=320)  # the filtered SFT file has 320 rows
        self.assertEqual((exact["epochs_kind"], exact["rows_kind"]), ("exact", "exact"))
        self.assertAlmostEqual(exact["epochs"], 1000 / 320)
        self.assertIn("3.12 epochs over 320 accepted SFT rows", oc._dose_text(exact))
        by_epochs = oc.recipes(v0)["seed/recipe"].dose(rows=320)
        self.assertEqual((by_epochs["steps"], by_epochs["steps_kind"], by_epochs["epochs"]), (160, "exact", 2.0))  # ceil(320 / 4) x 2
        self.assertIn("steps <= 250", oc._dose_text(oc.recipes(v0)["seed/recipe"].dose()))
        # no row bound at all: epochs under max_steps are unresolved
        steps_files = edit(v0, CFG, "num_epochs: 2\n", "num_epochs: 2\nmax_steps: 250\n")
        nobound = oc.recipes(edit(steps_files, RUN, 'PROMPT_LIMIT="${PROMPT_LIMIT:-250}"', 'PROMPT_LIMIT="$1"'))["seed/recipe"].dose()
        self.assertEqual((nobound["rows_kind"], nobound["epochs_kind"]), ("unresolved", "unresolved"))
        self.assertIn("epochs unresolved", oc._dose_text(nobound))

    def test_yaml_scalars(self):
        y = oc.yaml_top_level('a: 1\nb: "x # y"\nc: true  # note\nd:\n  - 1\ne: 2e-4\n f: 9\nmax_steps: 0\nmax_steps: 7\n')
        self.assertEqual((y["a"], y["b"], y["c"], y["d"], y["e"], y["max_steps"]), (1, "x # y", True, None, 2e-4, 7))
        self.assertNotIn("f", y)  # indented: not top level


class CheckerTests(unittest.TestCase):
    def setUp(self):
        self.v0 = v0_files()

    def check(self, cand, gpu=72.0, parent=None):
        return oc.check_rendered_config(parent or self.v0, cand, gpu_minutes=gpu)

    def test_a_notes_edit_has_no_findings(self):
        self.assertEqual(self.check(edit(self.v0, "notes/seed.md", "## Log", "## Log (newest first)")), ())

    def test_a_positive_max_steps_overrides_num_epochs(self):
        f = self.check(edit(self.v0, CFG, "num_epochs: 2\n", "num_epochs: 2\nmax_steps: 300\n"))
        self.assertEqual(codes(oc.blocking(f)), ["max_steps_overrides_epochs"])
        msg = next(x.message for x in f if x.code == "max_steps_overrides_epochs")
        self.assertIn("num_epochs: 2", msg)
        self.assertIn("1200 examples", msg)  # 300 steps x batch 4
        self.assertIn(">= 2.40 epochs over <= 500 SFT rows", msg)
        self.assertIn("training_dose_changed", codes(f, severity=oc.SOFT))
        # zero or negative max_steps does not override
        self.assertEqual(oc.blocking(self.check(edit(self.v0, CFG, "num_epochs: 2\n", "num_epochs: 2\nmax_steps: 0\n"))), ())

    def test_a_full_run_that_writes_max_steps_is_flagged_but_the_smoke_block_is_not(self):
        cand = edit(self.v0, RUN, 'TRAIN_CONFIG=config.yaml\n', "TRAIN_CONFIG=config.yaml\nsed -i 's/^# SMOKE_MAX_STEPS_ANCHOR.*/max_steps: 40/' config.yaml\n")
        self.assertIn("max_steps_overrides_epochs", codes(oc.blocking(self.check(cand))))
        smoke = edit(self.v0, RUN, "  MAX_TRAIN_EXAMPLES=4\n", "  MAX_TRAIN_EXAMPLES=4\n  echo max_steps smoke only\n")
        self.assertEqual(oc.blocking(self.check(smoke)), ())

    def test_the_training_dose_is_reported_and_a_change_is_soft(self):
        f = self.check(edit(self.v0, CFG, "num_epochs: 2\n", "num_epochs: 3\n"))
        self.assertEqual(oc.blocking(f), ())
        changed = next(x for x in f if x.code == "training_dose_changed")
        self.assertIn("steps <= 250 x batch 4", changed.message)
        self.assertIn("steps <= 375 x batch 4", changed.message)
        self.assertIn("training_dose", codes(f, severity=oc.INFO))
        empty = self.check(edit(self.v0, RUN, 'MIN_EXAMPLES="${MIN_EXAMPLES:-20}"', 'MIN_EXAMPLES="${MIN_EXAMPLES:-900}"'))
        self.assertEqual(codes(oc.blocking(empty)), ["training_dose_empty"])
        zero = self.check(edit(self.v0, RUN, 'PROMPT_LIMIT="${PROMPT_LIMIT:-250}"', 'PROMPT_LIMIT="${PROMPT_LIMIT:-0}"'))
        self.assertEqual(codes(oc.blocking(zero)), ["training_dose_empty"])

    def test_sampling_and_truncation(self):
        lower = self.check(edit(self.v0, RUN, 'MAX_TOKENS="${MAX_TOKENS:-6000}"', 'MAX_TOKENS="${MAX_TOKENS:-2000}"'))
        self.assertEqual(oc.blocking(lower), ())
        self.assertIn("finish_reason=length", next(x.message for x in lower if x.code == "max_tokens_lowered"))
        over = self.check(edit(self.v0, RUN, 'MAX_TOKENS="${MAX_TOKENS:-6000}"', 'MAX_TOKENS="${MAX_TOKENS:-9000}"'))
        self.assertEqual(codes(oc.blocking(over)), ["max_tokens_exceed_model_len"])
        keep = self.check(edit(self.v0, RUN, 'MAX_KEEP_PER_PROMPT="${MAX_KEEP_PER_PROMPT:-2}"', 'MAX_KEEP_PER_PROMPT="${MAX_KEEP_PER_PROMPT:-6}"'))
        self.assertIn("keep_cap_above_samples", codes(keep, severity=oc.SOFT))
        chars = self.check(edit(self.v0, RUN, 'MAX_SFT_CHARS="${MAX_SFT_CHARS:-14000}"', 'MAX_SFT_CHARS="${MAX_SFT_CHARS:-30000}"'))
        self.assertIn("sft_chars_exceed_sequence_len", codes(chars, severity=oc.SOFT))
        xhigh = self.check(edit(self.v0, RUN, 'REASONING_STRENGTH="${REASONING_STRENGTH:-high}"', 'REASONING_STRENGTH="${REASONING_STRENGTH:-xhigh}"'))
        self.assertIn("reasoning_raised_without_tokens", codes(xhigh, severity=oc.SOFT))

    def test_warm_starts_are_hard(self):
        for old, new in (("lora_model_dir:\n", "lora_model_dir: /home/ubuntu/lab/work/out/adapter\n"), ("lora_model_dir:\n", 'lora_model_dir: "$(rm -rf ~)"\n'),
                         ("num_epochs: 2\n", "num_epochs: 2\nresume_from_checkpoint: ./outputs/checkpoint-50\n"),
                         ("num_epochs: 2\n", "num_epochs: 2\nauto_resume_from_checkpoints: true\n")):
            with self.subTest(new=new):
                self.assertEqual(codes(oc.blocking(self.check(edit(self.v0, CFG, old, new)))), ["warm_start"])
        base = self.check(edit(self.v0, RUN, 'SMOKE="${SMOKE:-0}"\n', 'SMOKE="${SMOKE:-0}"\nexport GLIMMER_BASE=/work/out/merged\n'))
        self.assertEqual(codes(oc.blocking(base)), ["warm_start"])
        # base_model is overridden by the harness: the edit is pointless but not a warm start
        bm = self.check(edit(self.v0, CFG, "base_model: meta-models/Muse-Glimmer-30B\n", "base_model: some/other-model\n"))
        self.assertEqual(oc.blocking(bm), ())
        self.assertIn("base_model_ignored", codes(bm, severity=oc.SOFT))

    def test_time_boxes_against_the_trial_gpu_minutes(self):
        long = edit(self.v0, RUN, 'GEN_TIMEBOX_SECONDS="${GEN_TIMEBOX_SECONDS:-2700}"', 'GEN_TIMEBOX_SECONDS="${GEN_TIMEBOX_SECONDS:-4000}"')
        self.assertEqual(codes(oc.blocking(self.check(long, gpu=72))), ["timebox_exceeds_gpu_budget"])  # 4000 + 180 + 900 s > 72 min
        self.assertEqual(oc.blocking(self.check(long, gpu=120)), ())
        self.assertIn("timebox_changed", codes(self.check(long, gpu=120), severity=oc.SOFT))
        rt = edit(self.v0, RUN, 'REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-1200}"', 'REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-3000}"')
        self.assertIn("request_timeout_exceeds_timebox", codes(self.check(rt), severity=oc.SOFT))
        # the incumbent itself fits: 2700 + 180 + 900 s = 63 min of 72
        self.assertEqual(oc._state_findings(oc.recipes(self.v0)["seed/recipe"], 72.0), [])

    def test_evaluation_and_data_origin_changes_are_hard(self):
        cases = {
            "origin check": dict(self.v0, **{"seed/recipe/origin_check.py": self.v0["seed/recipe/origin_check.py"] + "\nraise SystemExit(0)\n"}),
            "dev data": edit(self.v0, RUN, 'TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.jsonl}"', 'TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/dev.jsonl}"'),
            "data dir": edit(self.v0, RUN, 'DATA_DIR="${GLIMMER_DATA_DIR:-/work/data}"', 'DATA_DIR="${GLIMMER_DATA_DIR:-/tmp/other}"'),
            "origin record": edit(self.v0, RUN, 'SAMPLES_JSONL="$OUT/glimmer_samples.jsonl"', 'SAMPLES_JSONL="$OUT/samples_copy.jsonl"'),
            "download": edit(self.v0, RUN, 'SMOKE="${SMOKE:-0}"\n', 'SMOKE="${SMOKE:-0}"\nwget -q https://example.invalid/extra.jsonl -O /work/data/extra.jsonl\n'),
            "deleted origin check": {k: v for k, v in self.v0.items() if k != "seed/recipe/origin_check.py"},
        }
        for name, cand in cases.items():
            with self.subTest(name):
                self.assertTrue(set(codes(oc.blocking(self.check(cand)))) & {"evaluation_or_origin_changed", "external_data_or_model"}, self.check(cand))

    def test_editing_another_part_of_a_line_that_names_the_data_is_not_an_origin_change(self):
        cand = edit(self.v0, RUN, 'python3 generate.py --base-url "$BASE_URL" --train-file "$TRAIN_FILE" \\',
                    'python3 generate.py --base-url "$BASE_URL" --train-file "$TRAIN_FILE" --min-p 0.05 \\')
        self.assertEqual(oc.blocking(self.check(cand)), ())
        # a comment that mentions dev.jsonl is not code
        self.assertEqual(oc.blocking(self.check(edit(self.v0, RUN, 'SMOKE="${SMOKE:-0}"\n', 'SMOKE="${SMOKE:-0}"\n# never train on dev.jsonl\n'))), ())

    def test_a_hard_finding_the_incumbent_already_has_does_not_block(self):
        parent = edit(self.v0, CFG, "num_epochs: 2\n", "num_epochs: 2\nmax_steps: 300\n")
        f = self.check(dict(parent, **{"seed/recipe/README.md": parent["seed/recipe/README.md"] + "\nmore\n"}), parent=parent)
        self.assertIn("max_steps_overrides_epochs", codes(f, severity=oc.HARD, new=False))
        self.assertEqual(oc.blocking(f), ())

    def test_unified_diff(self):
        d = oc.unified_diff(self.v0, edit(self.v0, CFG, "num_epochs: 2\n", "num_epochs: 3\n"))
        self.assertIn("--- a/seed/recipe/config.yaml", d)
        self.assertIn("-num_epochs: 2", d)
        self.assertIn("+num_epochs: 3", d)


# --------------------------------------------------------------------------------------------------
class FakeCritic:
    def __init__(self, replies, model=WONDER):
        self.replies, self.model, self.prompts = list(replies), model, []

    def critique(self, prompt):
        self.prompts.append(prompt)
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, oc.CriticReply):
            return item
        return oc.CriticReply(json.dumps(item) if isinstance(item, dict) else item, self.model, self.model, False)


ACCEPT = {"verdict": "accept", "challenges": [], "summary": "fine"}
REVISE = {"verdict": "revise", "challenges": ["max_steps: 300 trains 2.4 epochs, not the 2 the summary claims"], "summary": "mismatch"}


def result(p: Proposal, model=WONDER, served=WONDER) -> ProposalResult:
    return ProposalResult(p, model, served, served != model, 0.0, 10, 10, json.dumps(p.to_dict()))


class ChallengerTests(unittest.TestCase):
    def setUp(self):
        self.v0 = v0_files()
        self.ctx = ProposalContext("cand-001", PackRef("packs/pack.git", "b" * 40), "v0", "SYSTEM", "PUBLIC CONTEXT the proposer saw")

    def run_challenge(self, first, *, critic, script=(), forbidden=frozenset({"SECRET-SELECTION-PROMPT-TEXT"})):
        proposer = ScriptedProposalSource(list(script), model=WONDER, usd=0.0)
        out = oc.Challenger(critic).run(ctx=self.ctx, result=result(first), parent_files=self.v0, gpu_minutes=72.0, proposer=proposer, forbidden=forbidden)
        return out, proposer

    def test_a_clean_candidate_gets_one_critic_call_and_no_revision(self):
        critic = FakeCritic([ACCEPT])
        out, proposer = self.run_challenge(THREE_EPOCHS, critic=critic)
        self.assertEqual((len(critic.prompts), len(proposer.contexts)), (1, 0))
        self.assertFalse(out.revised or out.refused)
        self.assertEqual((out.critic_model, out.critic_served_model), (WONDER, WONDER))
        self.assertEqual(out.critic_sha256, oc._sha(json.dumps(ACCEPT)))
        self.assertEqual(out.report["critic"]["verdict"], "accept")
        self.assertEqual(out.report["revision"], {"attempted": False, "applied": False})
        # what the critic saw: the public context, the stated intent, the diff and the findings, nothing else
        p = critic.prompts[0]
        self.assertIn("PUBLIC CONTEXT the proposer saw", p)
        self.assertIn("+num_epochs: 3", p)
        self.assertIn("training_dose_changed", p)
        self.assertIn("train three epochs", p)
        self.assertNotIn("SYSTEM", p.split("## Public context")[1].split("##")[0])  # the proposer's own instructions are not passed on

    def test_a_hard_finding_gets_exactly_one_revision_and_a_fixed_revision_passes(self):
        critic = FakeCritic([REVISE])
        out, proposer = self.run_challenge(MAX_STEPS, critic=critic, script=[THREE_EPOCHS])
        self.assertEqual((len(critic.prompts), len(proposer.contexts)), (1, 1))
        self.assertTrue(out.revised)
        self.assertFalse(out.refused)
        self.assertIs(out.result.proposal, THREE_EPOCHS)
        rp = proposer.contexts[0].user_prompt
        self.assertTrue(rp.startswith("PUBLIC CONTEXT the proposer saw"))
        self.assertIn("ONE revision is allowed", rp)
        self.assertIn("max_steps_overrides_epochs", rp)
        self.assertIn("not the 2 the summary claims", rp)  # the critic's text, fenced
        self.assertIn("<<<DATA:reviewer-challenge", rp)
        self.assertEqual([f["code"] for f in out.report["initial_findings"] if f["severity"] == "hard"], ["max_steps_overrides_epochs"])
        self.assertEqual(out.report["blocking"], [])
        self.assertTrue(out.report["revision"]["applied"])
        self.assertEqual(set(out.artefacts), {"critic_prompt.txt", "critic_response.txt", "revision_prompt.txt", "revision_response.txt"})

    def test_a_revision_that_keeps_the_hard_finding_is_refused_and_there_is_no_second_round(self):
        critic = FakeCritic([REVISE])
        out, proposer = self.run_challenge(MAX_STEPS, critic=critic, script=[MAX_STEPS, THREE_EPOCHS])
        self.assertEqual((len(critic.prompts), len(proposer.contexts)), (1, 1))
        self.assertTrue(out.refused)
        self.assertEqual([f.code for f in out.blocking], ["max_steps_overrides_epochs"])
        self.assertTrue(out.report["refused"])

    def test_the_critic_alone_can_ask_for_the_one_revision(self):
        critic = FakeCritic([{"verdict": "revise", "challenges": ["three epochs overfits 500 rows"], "summary": "s"}])
        out, proposer = self.run_challenge(THREE_EPOCHS, critic=critic, script=[sr(CFG, "num_epochs: 2\n", "num_epochs: 2\nweight_decay: 0.01\n")])
        self.assertEqual(len(proposer.contexts), 1)
        self.assertTrue(out.revised and not out.refused)

    def test_an_unreadable_critic_reply_does_not_ask_for_a_revision(self):
        out, proposer = self.run_challenge(THREE_EPOCHS, critic=FakeCritic(["I think it is fine."]))
        self.assertEqual(out.report["critic"]["verdict"], "unreadable")
        self.assertEqual(len(proposer.contexts), 0)

    def test_a_failed_critic_is_recorded_and_the_checker_still_decides(self):
        out, proposer = self.run_challenge(MAX_STEPS, critic=FakeCritic([ProposerUnavailable("muse exec failed")]), script=[MAX_STEPS])
        self.assertIn("muse exec failed", out.report["critic"]["error"])
        self.assertIsNone(out.critic_sha256)
        self.assertEqual(len(proposer.contexts), 1)
        self.assertTrue(out.refused)
        self.assertIn("(no reviewer reply)", proposer.contexts[0].user_prompt)

    def test_a_critic_fallback_model_is_recorded(self):
        critic = FakeCritic([oc.CriticReply(json.dumps(ACCEPT), WONDER, "muse-spark-1.3", True)])
        out, _ = self.run_challenge(THREE_EPOCHS, critic=critic)
        self.assertEqual((out.critic_model, out.critic_served_model), (WONDER, "muse-spark-1.3"))
        self.assertTrue(out.report["critic"]["fallback_used"])

    def test_a_revision_that_cannot_be_read_or_applied_leaves_the_first_proposal(self):
        out, _ = self.run_challenge(MAX_STEPS, critic=FakeCritic([REVISE]), script=[UnparseableProposal("no JSON", raw_response="nope")])
        self.assertFalse(out.revised)
        self.assertTrue(out.refused)
        self.assertIn("no JSON", out.report["revision"]["error"])
        out2, _ = self.run_challenge(MAX_STEPS, critic=FakeCritic([REVISE]), script=[sr(CFG, "not in the file", "x")])
        self.assertIn("does not apply", out2.report["revision"]["error"])
        self.assertTrue(out2.refused)

    def test_a_proposal_that_does_not_apply_is_left_to_the_normal_path(self):
        critic = FakeCritic([ACCEPT])
        out, _ = self.run_challenge(sr(CFG, "not in the file", "x"), critic=critic)
        self.assertIsNone(out)
        self.assertEqual(critic.prompts, [])

    def test_a_denied_value_in_the_diff_sends_nothing_and_refuses_the_candidate(self):
        critic = FakeCritic([ACCEPT])
        leak = sr("notes/seed.md", "## Log", "## Log\nSECRET-SELECTION-PROMPT-TEXT")
        out, proposer = self.run_challenge(leak, critic=critic, script=[THREE_EPOCHS])
        self.assertEqual((critic.prompts, proposer.contexts), ([], []))  # no critic call, no revision call
        self.assertTrue(out.refused)
        self.assertEqual([f.code for f in out.blocking], ["private_value_in_candidate"])
        self.assertNotIn("SECRET-SELECTION-PROMPT-TEXT", json.dumps(out.report) + out.diff + "".join(out.artefacts.values()))
        self.assertEqual(out.artefacts, {})
        self.assertTrue(out.report["refused"])

    def test_the_critic_prompt_is_checked_before_it_is_sent(self):
        # the diff is clean; the stated intent (part of the critic prompt) carries the denied value
        p = Proposal("we know SECRET-SELECTION-PROMPT-TEXT", Lever.OTHER, "s", "e", "r", THREE_EPOCHS.edits)
        critic = FakeCritic([ACCEPT])
        out, proposer = self.run_challenge(p, critic=critic, script=[THREE_EPOCHS])
        self.assertEqual((critic.prompts, proposer.contexts), ([], []))
        self.assertIn("forbidden value", out.report["critic"]["not_sent"])
        self.assertRegex(out.report["critic"]["prompt_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn("critic_prompt.txt", out.artefacts)
        self.assertEqual(codes(out.blocking), ["private_value_in_candidate"])

    def test_the_revision_prompt_is_checked_before_it_is_sent(self):
        # the critic's reply carries the denied value; the revision prompt would quote it
        critic = FakeCritic([{"verdict": "revise", "challenges": ["SECRET-SELECTION-PROMPT-TEXT"], "summary": "s"}])
        out, proposer = self.run_challenge(MAX_STEPS, critic=critic, script=[THREE_EPOCHS])
        self.assertEqual(len(critic.prompts), 1)
        self.assertEqual(proposer.contexts, [])  # not sent
        self.assertIn("forbidden value", out.report["revision"]["not_sent"])
        self.assertNotIn("revision_prompt.txt", out.artefacts)
        self.assertFalse(out.revised)
        self.assertEqual(codes(out.blocking), ["max_steps_overrides_epochs"])  # the first proposal stands and is judged

    def test_a_revision_that_writes_a_denied_value_is_refused(self):
        leak = sr("notes/seed.md", "## Log", "## Log\nSECRET-SELECTION-PROMPT-TEXT")
        out, proposer = self.run_challenge(MAX_STEPS, critic=FakeCritic([REVISE]), script=[leak])
        self.assertEqual(len(proposer.contexts), 1)
        self.assertTrue(out.revised and out.refused)
        self.assertEqual(codes(out.blocking), ["private_value_in_candidate"])
        self.assertNotIn("SECRET-SELECTION-PROMPT-TEXT", out.diff)

    def test_a_send_guard_can_deny_or_stop(self):
        def deny(prompt):
            raise AccessDenied("not approved on the lab host")

        class Stop(Exception):
            pass

        def stop(prompt):
            raise Stop()

        critic = FakeCritic([ACCEPT])
        proposer = ScriptedProposalSource([], model=WONDER, usd=0.0)
        out = oc.Challenger(critic, send_guard=deny).run(ctx=self.ctx, result=result(THREE_EPOCHS), parent_files=self.v0, gpu_minutes=72.0,
                                                          proposer=proposer, forbidden=frozenset({"x" * 40}))
        self.assertTrue(out.refused)
        self.assertEqual(critic.prompts, [])
        with self.assertRaises(Stop):  # anything else (the rehearsal's "awaiting approval") propagates, still before the send
            oc.Challenger(critic, send_guard=stop).run(ctx=self.ctx, result=result(THREE_EPOCHS), parent_files=self.v0, gpu_minutes=72.0,
                                                       proposer=proposer, forbidden=frozenset({"x" * 40}))
        self.assertEqual(critic.prompts, [])

    def test_without_a_critic_the_checker_and_one_revision_still_run(self):
        out, proposer = self.run_challenge(MAX_STEPS, critic=None, script=[THREE_EPOCHS])
        self.assertEqual(out.report["critic"], {"error": "no critic configured"})
        self.assertTrue(out.revised and not out.refused)


# --------------------------------------------------------------------------------------------------
class RehearsalScriptTests(unittest.TestCase):
    """scripts/outer_rehearsal.py: no prompt is sent before the lab host approved its exact text; replies are cached for the rerun."""

    def setUp(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("outer_rehearsal", Path(oc.__file__).resolve().parents[2] / "scripts" / "outer_rehearsal.py")
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.ctx = ProposalContext("cand-001", PackRef("packs/pack.git", "b" * 40), "v0", "SYSTEM", "PUBLIC CONTEXT")

    class Inner:
        model = WONDER

        def __init__(self):
            self.sent, self.last_usage, self.last_run = [], {"input_tokens": 1}, None

        def propose(self, ctx):
            self.sent.append(ctx.user_prompt)
            return result(THREE_EPOCHS)

        def critique(self, prompt):
            self.sent.append(prompt)
            return oc.CriticReply(json.dumps(REVISE), WONDER, WONDER, False)

    def test_nothing_is_sent_until_approved_then_the_rerun_uses_the_cache(self):
        guard = self.mod.LabHostApproval(self.dir / "approval")
        inner = self.Inner()
        prop = self.mod._Cached(inner, self.dir / "cache", guard)
        with self.assertRaises(self.mod.AwaitingApproval):
            prop.propose(self.ctx)
        self.assertEqual(inner.sent, [])  # stopped BEFORE the send
        (pending,) = (self.dir / "approval" / "pending").glob("*.txt")
        self.assertEqual(pending.read_text(), "SYSTEM\n\nPUBLIC CONTEXT")
        (self.dir / "approval" / "approved" / pending.stem).write_text("")  # what check-prompts --approve writes on a pass
        r1 = prop.propose(self.ctx)
        r2 = self.mod._Cached(inner, self.dir / "cache", guard).propose(self.ctx)  # a rerun
        self.assertEqual(len(inner.sent), 1)  # Muse is called once
        self.assertEqual(r1.proposal, r2.proposal)

    def test_a_denied_prompt_is_refused_by_the_challenger_and_never_sent(self):
        guard = self.mod.LabHostApproval(self.dir / "approval")
        inner = self.Inner()
        critic = self.mod._Cached(inner, self.dir / "cache", guard)
        prompt = oc.critic_prompt(self.ctx, result(THREE_EPOCHS), oc.unified_diff(v0_files(), edit(v0_files(), CFG, "num_epochs: 2\n", "num_epochs: 3\n")),
                                  oc.check_rendered_config(v0_files(), edit(v0_files(), CFG, "num_epochs: 2\n", "num_epochs: 3\n"), gpu_minutes=72.0))
        (self.dir / "approval" / "denied" / self.mod._sha(prompt)).write_text("")
        out = oc.Challenger(critic, send_guard=guard).run(ctx=self.ctx, result=result(THREE_EPOCHS), parent_files=v0_files(), gpu_minutes=72.0,
                                                          proposer=ScriptedProposalSource([]), forbidden=frozenset())
        self.assertEqual(inner.sent, [])
        self.assertTrue(out.refused)
        self.assertIn("lab host", out.report["critic"]["not_sent"])


# --------------------------------------------------------------------------------------------------
class ContractTests(unittest.TestCase):
    def summary(self, **kw):
        base = dict(report=FileRef("candidates/cand-001/challenge.json", "a" * 64, 9), blocking=(), critic_model=WONDER, critic_served_model=WONDER,
                    critic_sha256="c" * 64, revised=True, refused=False)
        base.update(kw)
        return ChallengeSummary(**base)

    def test_round_trip_and_old_records_without_the_key(self):
        from tests.test_lab_contracts import candidate

        old = candidate()
        self.assertNotIn("challenge", old.to_dict())
        self.assertIsNone(CandidateRecord.from_dict(loads_strict(dumps_record(old))).challenge)
        new = candidate(challenge=self.summary())
        back = CandidateRecord.from_dict(loads_strict(dumps_record(new)))
        self.assertEqual(back, new)
        refused = candidate(pack=None, lint_passed=False, lint_messages=("refused",), challenge=self.summary(blocking=("warm_start",), refused=True, critic_sha256=None))
        self.assertEqual(CandidateRecord.from_dict(loads_strict(dumps_record(refused))), refused)

    def test_rules(self):
        from tests.test_lab_contracts import candidate

        with self.assertRaises(ContractError):
            self.summary(blocking=("warm_start",), refused=False)
        with self.assertRaises(ContractError):
            self.summary(refused=True)
        with self.assertRaises(ContractError):
            self.summary(critic_sha256="not a hash")
        with self.assertRaises(ContractError):  # a refused candidate has no commit and did not pass lint
            candidate(challenge=self.summary(blocking=("warm_start",), refused=True))
        d = candidate(challenge=self.summary()).to_dict()
        d["challenge"]["extra"] = 1
        with self.assertRaises(ContractError):
            CandidateRecord.from_dict(d)


# --------------------------------------------------------------------------------------------------
class OuterLoopTests(unittest.TestCase):
    def build(self, script, critic_replies):
        from nightshift.lab import stats
        from nightshift.lab.budget import LabCap
        from nightshift.lab.outer_loop import OuterController

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.lab = OuterLab(tmp.name, value_fn=lambda spec: 0.6, store=InMemoryPackStore(v0_files()))
        self.addCleanup(self.lab.ledger.close)
        self.proposer = ScriptedProposalSource(list(script), model=WONDER, usd=0.0)
        self.critic = FakeCritic(critic_replies)
        self.ctl = OuterController(self.lab.layout, self.lab.ledger, self.lab.reader, self.lab.store, self.proposer, self.lab, stats, clock=self.lab.clock,
                                   lab_cap=LabCap(1000.0), hosts=("ax43.example",), challenger=oc.Challenger(self.critic))

    def test_refused_revised_and_clean_candidates_are_recorded(self):
        note = make_proposal("notes/seed.md", "## Log", "## Log\n- idea 3")
        # cand-001: max_steps, revised into the same mistake -> refused; cand-002: max_steps, revised into three epochs; cand-003: clean
        self.build([MAX_STEPS, MAX_STEPS, MAX_STEPS, THREE_EPOCHS, note], [REVISE, REVISE, ACCEPT])
        ds = {d.candidate_id: d for d in self.ctl.run_round()}
        st = self.lab.reader.state()
        self.assertEqual(len(self.critic.prompts), 3)  # one critic call per candidate
        self.assertEqual(len(self.proposer.contexts), 5)  # three proposals, two revisions
        c1, c2, c3 = (st.candidates[f"cand-00{i}"] for i in (1, 2, 3))
        # refused: recorded, no commit, never run
        self.assertEqual(ds["cand-001"].outcome, Outcome.INVALID)
        self.assertIsNone(c1.pack)
        self.assertFalse(c1.lint_passed)
        self.assertTrue(c1.lint_messages[0].startswith("refused by the rendered-config check: max_steps_overrides_epochs"))
        self.assertEqual((c1.challenge.refused, c1.challenge.revised, c1.challenge.blocking), (True, True, ("max_steps_overrides_epochs",)))
        self.assertNotIn("cand-001", st.packs)
        self.assertNotIn("cand-001", self.lab.store._tags)
        self.assertTrue(all(s.pack_version != "cand-001" for s in self.lab.executed))
        self.assertIn("+max_steps: 300", (self.lab.layout.root / c1.diff.path).read_text())
        # revised: the revision is what is committed and run
        self.assertTrue(c2.lint_passed)
        self.assertEqual((c2.challenge.refused, c2.challenge.revised), (False, True))
        self.assertEqual(c2.reason.split("\n")[0], "train three epochs")
        self.assertIn("num_epochs: 3", self.lab.store._files[c2.pack.commit][CFG])
        self.assertNotIn("max_steps: 300", self.lab.store._files[c2.pack.commit][CFG])
        # clean: no revision
        self.assertEqual((c3.challenge.refused, c3.challenge.revised), (False, False))
        # the report and its artefacts
        rep = json.loads((self.lab.layout.root / c2.challenge.report.path).read_text())
        self.assertEqual(rep["critic"]["requested_model"], WONDER)
        self.assertEqual(rep["critic"]["served_model"], WONDER)
        self.assertEqual(rep["critic"]["response_sha256"], c2.challenge.critic_sha256)
        self.assertTrue(rep["revision"]["applied"])
        for name, ref in rep["files"].items():
            self.assertEqual(FileRef.from_dict(ref), self.lab.layout.file_ref(self.lab.layout.candidate_dir("cand-002") / name))
        self.assertEqual(st.malformed, [])

    def test_in_the_outer_loop_a_denied_value_is_refused_and_recorded_before_any_critic_send(self):
        # "ax43.example" is a host the controller denies (the same denylist the proposer prompt passes, built on the control host)
        leak = make_proposal("notes/seed.md", "## Log", "## Log\n- ran on ax43.example")
        note = make_proposal("notes/seed.md", "## Log", "## Log\n- idea 2")
        self.build([leak, note, THREE_EPOCHS], [ACCEPT, ACCEPT])
        ds = {d.candidate_id: d for d in self.ctl.run_round()}  # no AccessDenied escapes; the round goes on
        st = self.lab.reader.state()
        self.assertEqual(len(self.critic.prompts), 2, "cand-001 never reached the critic")
        self.assertTrue(all("ax43.example" not in p for p in self.critic.prompts))
        c1 = st.candidates["cand-001"]
        self.assertEqual(ds["cand-001"].outcome, Outcome.INVALID)
        self.assertIsNone(c1.pack)
        self.assertEqual(c1.challenge.blocking, ("private_value_in_candidate",))
        self.assertTrue(c1.lint_messages[0].startswith("refused by the rendered-config check: private_value_in_candidate"))
        self.assertNotIn("ax43.example", (self.lab.layout.root / c1.diff.path).read_text())
        rep = json.loads((self.lab.layout.root / c1.challenge.report.path).read_text())
        self.assertTrue(rep["refused"])
        self.assertNotIn("ax43.example", json.dumps(rep))
        self.assertEqual(len(self.proposer.contexts), 3)  # no revision call for cand-001
        self.assertTrue(st.candidates["cand-002"].lint_passed)

    def test_the_critic_and_revision_prompts_carry_no_private_values(self):
        from nightshift.lab.outer_context import forbidden_values

        self.build([MAX_STEPS, THREE_EPOCHS, THREE_EPOCHS, THREE_EPOCHS], [REVISE, ACCEPT, ACCEPT])
        self.ctl.run_round()
        forbidden = forbidden_values(self.lab.reader.state(), self.lab.layout, ("ax43.example",))
        self.assertTrue(forbidden)
        texts = self.critic.prompts + [c.user_prompt for c in self.proposer.contexts]
        for t in texts:
            for v in forbidden:
                if len(v) >= 8 and not v.replace(".", "").replace("%", "").isdigit():
                    self.assertNotIn(v, t)
        # and the denylist check itself ran on the critic prompt: the same check the proposer prompt passes
        from nightshift.lab.outer_context import assert_no_forbidden

        for t in texts:
            assert_no_forbidden(t, forbidden)


if __name__ == "__main__":
    unittest.main()
