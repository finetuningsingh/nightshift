"""Sample generation on approved train prompts, then filtering (spec 3.5).

Training targets come only from the local base model's own generations on
approved `train_prompts`. The fake generator is deterministic and labelled
fake; its output may only flow into fake training. The real boundary runs a
local-model job through the provider/job interface and is NOT YET VALIDATED
LIVE. vLLM/HF are never imported at module import time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_BASE_MODEL, PLACEHOLDER_REVISION, assert_pinned_base_model
from .data import PublicDataView
from .errors import NightshiftError, PolicyError, ProviderError
from .hashing import canonical_json, hash_object, sha256_hex
from .providers import JobSpec, RentalRequest
from .verifiers import check_text, filter_samples, normalise_check

FAKE_GENERATOR_ID = "nightshift-fake-generator-1.0.0"


class GenerationError(NightshiftError):
    """Sample generation failed or was refused."""


def _uniform(seed_text: str) -> float:
    return int(sha256_hex(seed_text)[:8], 16) / 0xFFFFFFFF


def synthesize_constrained_text(checks: list[dict[str, Any]]) -> str:
    """Deterministically build a short toy text satisfying the checks.

    Used only by fake components (the fake generator and the fake Inspect
    model's outputs). It never supplies real training targets.
    """
    keywords: list[str] = []
    forbidden_chars: set[str] = set()
    max_words: int | None = None
    min_words = 1
    want_lower = want_upper = want_json = want_single = False
    json_keys: list[str] = []
    for raw in checks or []:
        verifier_id, params = normalise_check(raw)
        if verifier_id == "required_keyword":
            raw_kw = params.get("keywords", params.get("keyword", []))
            keywords.extend([raw_kw] if isinstance(raw_kw, str) else list(raw_kw))
        elif verifier_id == "max_words":
            value = int(params.get("max_words", params.get("limit", 40)))
            max_words = value if max_words is None else min(max_words, value)
        elif verifier_id == "min_words":
            min_words = max(min_words, int(params.get("min_words", params.get("limit", 1))))
        elif verifier_id == "lowercase":
            want_lower = True
        elif verifier_id == "uppercase":
            want_upper = True
        elif verifier_id == "no_whitespace":
            want_single = True
        elif verifier_id == "forbidden_characters":
            chars = params.get("characters", params.get("chars", ""))
            forbidden_chars.update(list(chars) if isinstance(chars, str) else list(chars))
        elif verifier_id == "valid_json":
            want_json = True
            json_keys = list(params.get("required_keys", params.get("keys", [])))
    if want_json:
        value = keywords[0] if keywords else "toy"
        obj = {key: value for key in (json_keys or ["answer"])}
        return json.dumps(obj, sort_keys=True)
    tokens = [str(k) for k in keywords] or ["ready"]
    filler = "TOY" if want_upper else "toy"
    while len(" ".join(tokens).split()) < min_words:
        tokens.append(filler)
    if want_single:
        text = "-".join(" ".join(tokens).split())
    else:
        words = " ".join(tokens).split()
        if max_words is not None:
            words = words[:max_words]
        text = " ".join(words)
    if want_lower:
        text = text.lower()
    if want_upper:
        text = text.upper()
    for char in forbidden_chars:
        text = text.replace(char, "")
    return text or ("TOY" if want_upper else "toy")


def violating_text() -> str:
    """A generic answer that breaks the toy constraints."""
    return (
        "This rambling answer ignores every stated toy constraint completely "
        "and keeps adding extra unnecessary words without stopping!"
    )


def checks_for_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(row.get("checks") or row.get("verifiers") or [])


SAMPLE_SET_KINDS = ("generated", "filtered")


def samples_content_hash(samples: list[dict[str, Any]]) -> str:
    """Content hash over the canonical full samples (not just their ids)."""
    return hash_object(list(samples))


@dataclass
class SampleSet:
    sample_set_id: str
    samples: list[dict[str, Any]]
    fake: bool
    path: Path | None = None
    kind: str = "generated"
    content_hash: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_set_id": self.sample_set_id,
            "kind": self.kind,
            "content_hash": self.content_hash,
            "n": len(self.samples),
            "fake": bool(self.fake),
            "sample_ids": [s.get("sample_id") for s in self.samples],
        }


def write_sample_set(
    samples: list[dict[str, Any]], dest_dir: str | Path, kind: str = "generated"
) -> SampleSet:
    if kind not in SAMPLE_SET_KINDS:
        raise GenerationError(f"unknown sample set kind {kind!r}")
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    fake = all(bool(s.get("fake")) for s in samples) if samples else False
    content_hash = samples_content_hash(samples)
    sample_set_id = "samples-" + hash_object(
        {"kind": kind, "content_hash": content_hash, "fake": fake}
    )[:12]
    with (dest / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(canonical_json(sample) + "\n")
    record = SampleSet(
        sample_set_id, samples, fake, dest, kind=kind, content_hash=content_hash
    )
    (dest / "sample_set.json").write_text(
        json.dumps(record.to_record(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return record


def load_sample_set(path: str | Path) -> SampleSet:
    dest = Path(path)
    samples = [
        json.loads(line)
        for line in (dest / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = json.loads((dest / "sample_set.json").read_text(encoding="utf-8"))
    kind = str(meta.get("kind") or "")
    if kind not in SAMPLE_SET_KINDS:
        raise GenerationError(
            f"sample set at {dest} has unknown kind {kind!r}; refusing it"
        )
    expected_hash = str(meta.get("content_hash") or "")
    actual_hash = samples_content_hash(samples)
    if not expected_hash or actual_hash != expected_hash:
        raise GenerationError(
            f"sample set at {dest} failed its content-hash check "
            f"(stored {expected_hash!r}, recomputed {actual_hash!r}); "
            "the sample file may have been tampered with; refusing it"
        )
    fake = all(bool(s.get("fake")) for s in samples) if samples else False
    if bool(meta.get("fake")) != fake:
        raise GenerationError(
            f"sample set at {dest} has a fake flag that disagrees with "
            "its samples; refusing it"
        )
    return SampleSet(
        str(meta["sample_set_id"]), samples, fake, dest,
        kind=kind, content_hash=actual_hash,
    )


def verify_trainable_sample_set(
    sample_set: SampleSet, train_rows: list[dict[str, Any]]
) -> None:
    """Train-boundary provenance check (spec 3.5, 3.7).

    Only a filtered set may train, and every sample is re-verified here
    against ALL checks of its approved training row, regardless of which
    verifier subset any researcher asked filtering to apply.
    """
    if sample_set.kind != "filtered":
        raise PolicyError(
            f"training refused: sample set {sample_set.sample_set_id} is "
            f"{sample_set.kind!r}, not 'filtered'; only verifier-filtered "
            "local generations may become training targets"
        )
    rows_by_id = {row["id"]: row for row in train_rows}
    for sample in sample_set.samples:
        prompt_id = sample.get("prompt_id")
        row = rows_by_id.get(str(prompt_id))
        if row is None:
            raise PolicyError(
                f"training refused: sample prompt id {prompt_id!r} is not "
                "in approved train_prompts"
            )
        text = sample.get("text")
        if not isinstance(text, str) or not text:
            raise PolicyError(
                f"training refused: sample for prompt {prompt_id!r} has "
                "no non-empty string text"
            )
        results = check_text(text, checks_for_row(row))
        failed = [r.verifier_id for r in results if not r.passed]
        if failed:
            raise PolicyError(
                f"training refused: sample for prompt {prompt_id!r} does "
                f"not pass all checks of its approved training row "
                f"(failed: {failed})"
            )


@dataclass
class FilterOutcome:
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]

    @property
    def pass_rate(self) -> float:
        total = len(self.accepted) + len(self.rejected)
        return len(self.accepted) / total if total else 0.0

    def to_record(self, fake: bool) -> dict[str, Any]:
        return {
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "pass_rate": self.pass_rate,
            "fake": bool(fake),
        }


def filter_sample_set(
    samples: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    verifier_ids: list[str] | tuple[str, ...] | None = None,
) -> FilterOutcome:
    checks_by_prompt = {row["id"]: checks_for_row(row) for row in train_rows}
    accepted, rejected = filter_samples(samples, checks_by_prompt, verifier_ids)
    return FilterOutcome(accepted, rejected)


class FakeGenerator:
    """Deterministic fake local-model generator. Every sample is fake."""

    fake = True
    generator_id = FAKE_GENERATOR_ID

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        base_revision: str = PLACEHOLDER_REVISION,
        quality: float = 0.75,
    ):
        self.base_model = base_model
        self.base_revision = base_revision
        self.quality = float(quality)

    def generate(
        self,
        train_rows: list[dict[str, Any]],
        prompt_ids: list[str],
        count: int = 1,
        seed: int = 0,
        sampling_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        by_id = {row["id"]: row for row in train_rows}
        unknown = [p for p in prompt_ids if p not in by_id]
        if unknown:
            raise PolicyError(
                f"generation refused: prompt id(s) not in approved train_prompts: {unknown}"
            )
        samples: list[dict[str, Any]] = []
        for prompt_id in prompt_ids:
            row = by_id[prompt_id]
            checks = checks_for_row(row)
            for index in range(int(count)):
                good = _uniform(f"fake-gen:{seed}:{prompt_id}:{index}") <= self.quality
                text = synthesize_constrained_text(checks) if good else violating_text()
                sample_id = "sample-" + sha256_hex(
                    f"{seed}:{prompt_id}:{index}:{text}"
                )[:12]
                samples.append(
                    {
                        "sample_id": sample_id,
                        "prompt_id": prompt_id,
                        "prompt": row["prompt"],
                        "text": text,
                        "generator": self.generator_id,
                        "base_model": self.base_model,
                        "base_revision": self.base_revision,
                        "sampling_params": dict(sampling_params or {}),
                        "fake": True,
                    }
                )
        return samples

    def generate_and_filter(
        self,
        view: PublicDataView,
        prompt_ids: list[str],
        count: int = 1,
        seed: int = 0,
        sampling_params: dict[str, Any] | None = None,
        verifier_ids: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[list[dict[str, Any]], FilterOutcome]:
        train_rows = view.train_prompts()
        samples = self.generate(train_rows, prompt_ids, count, seed, sampling_params)
        return samples, filter_sample_set(samples, train_rows, verifier_ids)


class RealLocalGenerator:
    """Local-model generation on rented GPU. NOT YET VALIDATED LIVE.

    Builds a provider job (declared seconds are reserved by the caller's
    ResourceManager before any create) and refuses to fabricate samples when
    no worker artefact comes back.
    """

    fake = False
    validated_live = False
    generator_id = "nightshift-real-local-generator-unvalidated"

    def __init__(
        self,
        resource_manager: Any = None,
        config: Any = None,
        engine: str = "vllm",
    ):
        self.resource_manager = resource_manager
        self.config = config
        self.engine = engine

    def build_job(self, prompt_ids: list[str], work_dir: str = "/work/generation") -> JobSpec:
        command = (
            "python", "-m", "nightshift.real_worker", "generate",
            "--engine", self.engine, "--work-dir", work_dir,
        )
        return JobSpec(
            job_id="generate-" + hash_object(list(prompt_ids))[:12],
            kind="generate",
            command=command,
        )

    def generate(
        self,
        train_rows: list[dict[str, Any]],
        prompt_ids: list[str],
        request: RentalRequest,
        count: int = 1,
        seed: int = 0,
        sampling_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        approved = {row["id"] for row in train_rows}
        unknown = [p for p in prompt_ids if p not in approved]
        if unknown:
            raise PolicyError(
                f"generation refused: prompt id(s) not in approved train_prompts: {unknown}"
            )
        if self.config is not None:
            assert_pinned_base_model(self.config)
        if self.resource_manager is None:
            raise ProviderError(
                "real local generation is not yet validated live and has no "
                "ResourceManager/provider attached; refusing to generate"
            )
        execution = self.resource_manager.run_and_destroy(
            request, self.build_job(prompt_ids)
        )
        produced = execution.result.output.get("samples")
        if not produced:
            raise ProviderError(
                "real local generation is not yet validated live: the provider "
                "job returned no sample artefact"
            )
        # Every worker-returned sample is validated before acceptance: a
        # faulty or hostile worker must not inject arbitrary targets.
        samples = list(produced)
        expected = len(prompt_ids) * int(count)
        if len(samples) != expected:
            raise ProviderError(
                f"real local generation returned {len(samples)} sample(s) "
                f"for a request of {expected}; refusing the artefact"
            )
        for sample in samples:
            if not isinstance(sample, dict):
                raise ProviderError(
                    "real local generation returned a non-object sample; "
                    "refusing the artefact"
                )
            if str(sample.get("prompt_id") or "") not in approved:
                raise ProviderError(
                    f"real local generation returned a sample for unknown "
                    f"prompt id {sample.get('prompt_id')!r}; refusing it"
                )
            text = sample.get("text")
            if not isinstance(text, str) or not text:
                raise ProviderError(
                    "real local generation returned a sample with no "
                    "non-empty string text; refusing it"
                )
            if sample.get("fake") is not False:
                raise ProviderError(
                    "real local generation returned a sample whose fake "
                    "flag is not explicitly false; refusing it"
                )
        return samples


def make_generator(config: Any, resource_manager: Any = None) -> Any:
    if config.fake:
        return FakeGenerator(config.models.base_model, config.models.base_revision)
    assert_pinned_base_model(config)
    return RealLocalGenerator(resource_manager=resource_manager, config=config)
