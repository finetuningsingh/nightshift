"""Spec 3.5/3.7: generation boundaries, filtering, recipe forcing, fake training."""

import pytest

from nightshift.checkpoints import validate_checkpoint_dir
from nightshift.data import PublicDataView, demo_registry
from nightshift.errors import PolicyError
from nightshift.generation import FakeGenerator, load_sample_set, write_sample_set
from nightshift.pack import create_pack, parse_recipe_keys
from nightshift.training import FakeTrainer, RealTrainer, TrainingError, render_recipe


def public_rows():
    return PublicDataView(demo_registry())


def test_fake_generation_only_approved_train_ids_and_deterministic():
    view = public_rows()
    generator = FakeGenerator(quality=1.0)
    prompt_ids = view.train_prompt_ids()[:3]
    first = generator.generate(view.train_prompts(), prompt_ids, count=2, seed=7)
    second = generator.generate(view.train_prompts(), prompt_ids, count=2, seed=7)
    assert first == second
    assert all(sample["fake"] is True for sample in first)
    assert {sample["prompt_id"] for sample in first} == set(prompt_ids)
    with pytest.raises(PolicyError):
        generator.generate(view.train_prompts(), ["sel-01"], seed=7)
    with pytest.raises(PolicyError):
        generator.generate(view.train_prompts(), ["test-01"], seed=7)


def test_generate_and_filter_stamps_verifier_versions(tmp_path):
    view = public_rows()
    generator = FakeGenerator(quality=1.0)
    samples, outcome = generator.generate_and_filter(
        view, view.train_prompt_ids(), count=1, seed=3
    )
    assert outcome.accepted and not outcome.rejected
    assert outcome.pass_rate == 1.0
    assert all(s["verifier_results"] for s in outcome.accepted)
    assert all(
        result["version"] == "1.0.0"
        for sample in outcome.accepted for result in sample["verifier_results"]
    )
    sample_set = write_sample_set(outcome.accepted, tmp_path / "set")
    assert sample_set.fake is True
    loaded = load_sample_set(tmp_path / "set")
    assert [s["sample_id"] for s in loaded.samples] == [s["sample_id"] for s in samples]

    noisy = FakeGenerator(quality=0.0)
    _samples, rejected_outcome = noisy.generate_and_filter(
        view, view.train_prompt_ids(), count=1, seed=3
    )
    assert rejected_outcome.rejected and not rejected_outcome.accepted


MALICIOUS_TEMPLATE = """\
base_model: someone-else/evil-model
load_in_4bit: true
adapter: qlora
learning_rate: 0.0002
lora_r: 32
num_epochs: 99
max_steps: 100000
seed: 12345
output_dir: /etc/passwd-dir
datasets:
  - path: /private/selection.jsonl
    type: alpaca
"""


def test_recipe_controller_keys_are_forced_and_only_allowed_keys_vary():
    rendered = render_recipe(
        MALICIOUS_TEMPLATE,
        overrides={"learning_rate": 0.001},
        allowed_keys={"learning_rate", "lora_r", "num_epochs"},
        base_model="meta-models/Muse-Glimmer-30B",
        base_revision="abc123pinned",
        dataset_path="/run/dataset.jsonl",
        output_dir="/run/checkpoint",
        seed=17,
        max_steps_ceiling=50,
        max_epochs_ceiling=2,
    )
    keys = parse_recipe_keys(rendered.text)
    assert keys["base_model"] == '"meta-models/Muse-Glimmer-30B"'
    assert keys["output_dir"] == '"/run/checkpoint"'
    assert keys["max_steps"] == "50"
    assert keys["num_epochs"] == "2"
    assert keys["seed"] == "17"
    assert keys["learning_rate"] in ("0.001",)
    assert "/run/dataset.jsonl" in rendered.text
    assert "selection.jsonl" not in rendered.text
    assert "abc123pinned" in rendered.text
    with pytest.raises(PolicyError):
        render_recipe(MALICIOUS_TEMPLATE, overrides={"base_model": "x"},
                      allowed_keys={"learning_rate"})
    with pytest.raises(PolicyError):
        render_recipe(MALICIOUS_TEMPLATE, overrides={"optimizer": "adafactor"},
                      allowed_keys={"learning_rate"})
    with pytest.raises(PolicyError):
        render_recipe(MALICIOUS_TEMPLATE, overrides={"output_dir": "/tmp/x"},
                      allowed_keys={"learning_rate"})


def test_fake_training_writes_valid_labelled_fake_adapter(tmp_path):
    view = public_rows()
    pack = create_pack(tmp_path / "pack")
    generator = FakeGenerator(quality=1.0)
    _samples, outcome = generator.generate_and_filter(
        view, view.train_prompt_ids(), count=1, seed=11
    )
    result = FakeTrainer().train(
        outcome.accepted,
        checkpoint_dir=tmp_path / "ckpt",
        pack_dir=pack,
        seed=11,
        work_dir=tmp_path / "training",
    )
    assert result.fake is True and result.quality is not None and result.quality > 0
    assert result.checkpoint_hash == validate_checkpoint_dir(result.checkpoint_path).checkpoint_hash
    assert result.to_record()["fake"] is True
    repeated = FakeTrainer().train(
        outcome.accepted,
        checkpoint_dir=tmp_path / "ckpt-2",
        pack_dir=pack,
        seed=11,
        work_dir=tmp_path / "training-2",
    )
    assert repeated.checkpoint_hash == result.checkpoint_hash  # deterministic
    with pytest.raises(TrainingError):
        FakeTrainer().train([], checkpoint_dir=tmp_path / "empty")


def test_real_training_rejects_fake_samples_before_any_provider_call(tmp_path):
    pack = create_pack(tmp_path / "pack")
    view = public_rows()
    samples = FakeGenerator(quality=1.0).generate(
        view.train_prompts(), view.train_prompt_ids()[:1], seed=1
    )
    from nightshift.providers import RentalRequest

    with pytest.raises(TrainingError, match="fake-labelled"):
        RealTrainer().train(
            samples,
            checkpoint_dir=tmp_path / "real-ckpt",
            pack_dir=pack,
            request=RentalRequest(compute_seconds=10, setup_seconds=5,
                                  transfer_seconds=5, evaluation_seconds=5),
            work_dir=tmp_path / "real-training",
        )
