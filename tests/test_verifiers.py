"""Spec 3.5: pinned training-only verifiers carry ids and versions."""

from nightshift.verifiers import (
    VERIFIER_SET_VERSION,
    check_text,
    filter_samples,
    verifier_catalogue,
)


def test_each_verifier_behaves_and_is_versioned():
    cases = [
        ({"id": "max_words", "params": {"max_words": 3}}, "one two three", True),
        ({"id": "max_words", "params": {"max_words": 3}}, "one two three four", False),
        ({"id": "min_words", "params": {"min_words": 2}}, "one", False),
        ({"id": "required_keyword", "params": {"keyword": "Lantern"}}, "a lantern glows", True),
        ({"id": "forbidden_keyword", "params": {"keyword": "spam"}}, "no spam here", False),
        ({"id": "forbidden_characters", "params": {"characters": "!"}}, "quiet now", True),
        ({"id": "valid_json", "params": {"required_keys": ["a"]}}, '{"a": 1}', True),
        ({"id": "valid_json"}, "not json", False),
        ({"id": "lowercase"}, "all small words", True),
        ({"id": "lowercase"}, "Has Capital", False),
        ({"id": "uppercase"}, "ALL CAPS NOW", True),
        ({"id": "no_whitespace"}, "onetoken", True),
    ]
    for check, text, expected in cases:
        (result,) = check_text(text, [check])
        assert result.passed is expected, (check, text)
        assert result.verifier_id == check["id"]
        assert result.version == "1.0.0"


def test_filter_samples_stamps_ids_and_versions():
    checks = {"p1": [{"id": "required_keyword", "params": {"keyword": "lantern"}}]}
    samples = [
        {"prompt_id": "p1", "text": "the lantern is lit"},
        {"prompt_id": "p1", "text": "a candle instead"},
    ]
    accepted, rejected = filter_samples(samples, checks)
    assert [s["text"] for s in accepted] == ["the lantern is lit"]
    assert [s["text"] for s in rejected] == ["a candle instead"]
    record = accepted[0]["verifier_results"][0]
    assert record["verifier_id"] == "required_keyword" and record["version"] == "1.0.0"
    assert accepted[0]["verifier_set"] == VERIFIER_SET_VERSION


def test_catalogue_is_pinned():
    catalogue = verifier_catalogue()
    assert {entry["id"] for entry in catalogue} >= {
        "max_words", "required_keyword", "forbidden_characters", "valid_json", "lowercase",
    }
    assert all(entry["version"] for entry in catalogue)
