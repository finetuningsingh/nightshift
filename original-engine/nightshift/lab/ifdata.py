"""Data for the lab: JSONL rows, salted deterministic splits, per-assignment files and the assignment manifest.

Input is JSONL, one object per line with the required keys ``key``, ``prompt``, ``ground_truth`` (the Python-literal string open-instruct's
verifier reads), ``instruction_ids`` (list of constraint ids); optional ``constraint`` (the constraint sentences, tab-separated: they are
removed from the prompt to get the base prompt). Converting the IF-RLVR parquet is the opt-in script ``scripts/lab_convert_parquet.py``,
the only place ``pyarrow`` is imported.

Port of the reader's ``build_splits.py`` (deterministic for a given salt, no test leakage) extended to assignments:

1. drop duplicates (same row id or same prompt text), unsatisfiable ``nth_paragraph`` rows, prompts over ``max_prompt_chars``;
2. decontaminate base prompts against every IFBench test file by 13-gram, ignoring template n-grams (constraint boilerplate that occurs in
   at least ``template_df`` rows);
3. partition by constraint family: the EVOLUTION pool holds rows whose constraints all lie in evolution families, the HELD-OUT pool rows whose
   constraints all lie in held-out families; rows spanning both (or using an id that belongs to no family) are dropped and counted
   (``MixedRowPolicy.DROPPED``);
4. order each pool by ``sha256(salt | row_id)`` (so the result does not depend on input order) and draw, per assignment in plan order:
   selection first, then dev, then train. A selection row reserves its BASE prompt: no other row with that base prompt is used anywhere, and
   no selection or held-out base prompt reaches a train or dev file of any assignment. Stronger: a train or dev row is not drawn when its base
   prompt shares a non-template 13-gram with a selection or held-out base prompt. So the public train and dev files scan clean, and a 13-gram
   hit in a training dataset is a real hit, not a shared base prompt or a boilerplate artefact. The template n-grams (those occurring in at least
   ``template_df`` rows) are written to ``private/template_ngrams.json`` and the contamination scan ignores the same set.

The salt is a random 256-bit secret under ``private/split_salt``; only its sha256 is recorded in the manifest. Selection files carry
``ground_truth`` and never leave the control host. Output rows: ``key`` (the 20-hex row id), ``source_key``, ``prompt``, ``ground_truth``,
``instruction_ids``, ``family`` (the row's constraint families joined with ``+``), ``base_prompt`` (the prompt without its constraint sentences).

``AssignmentManifest.item_mode`` says how selection and held-out items were made. ``build_splits`` draws rows from the public dataset with the
secret salt (``SALTED_DRAW``); the residual risk (a researcher who retrieves the public dataset could train on those rows) is covered by the
contamination scan and stated in the report.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from nightshift.lab.contracts import (
    CONTAMINATION_NGRAM,
    DATA_REVISION_KEYS,
    REV_DEV_SHA256,
    REV_SELECTION_SHA256,
    REV_TRAIN_SHA256,
    SRC_IFBENCH_COMMIT,
    SRC_IFRLVR_PARQUET_SHA256,
    SRC_OPEN_INSTRUCT_COMMIT,
    SRC_SPLIT_SALT_SHA256,
    TEST_ASSIGNMENT_ID,
    Assignment,
    AssignmentManifest,
    AssignmentRole,
    ContractError,
    ItemMode,
    LabLayout,
    MixedRowPolicy,
    ScorerError,
    loads_strict,
    sha256_file,
    utc_iso,
)
from nightshift.lab.rewards import RewardsError, parse_ground_truth

# ---- constraint families (explicit; nothing is inferred) -----------------------------------------------------------------------
FAMILIES: dict = {
    # IFBench-Train constraints (29), grouped by what they test
    "copy": ["copy:copy", "copy:copying_multiple", "copy:copying_simple", "copy:repeat_phrase", "new:copy_span_idx"],
    "firstlast": ["first_word:first_word_answer", "first_word:first_word_sent", "last_word:last_word_answer", "last_word:last_word_sent"],
    "counting": ["count:count_increment_word", "count:count_unique", "count:counting_composition", "count:lowercase_counting",
                 "letters:letter_counting", "letters:letter_counting2"],
    "keywords_new": ["keywords:exclude_word_harder", "keywords:keyword_specific_position", "keywords:no_adjacent_consecutive", "keywords:palindrome",
                     "keywords:start_end", "keywords:word_count_different_numbers", "keywords:word_once"],
    "format_new": ["detectable_format:bigram_wrapping", "detectable_format:sentence_hyphens", "detectable_format:square_brackets",
                   "paragraphs:paragraphs", "paragraphs:paragraphs2", "punctuation:punctuation_dot", "punctuation:punctuation_exclamation"],
    # the 25 classic IFEval ids (all present in the IFBench package registry)
    "ifeval25": [
        "change_case:capital_word_frequency", "change_case:english_capital", "change_case:english_lowercase", "combination:repeat_prompt",
        "combination:two_responses", "detectable_content:number_placeholders", "detectable_content:postscript",
        "detectable_format:constrained_response", "detectable_format:json_format", "detectable_format:multiple_sections",
        "detectable_format:number_bullet_lists", "detectable_format:number_highlighted_sections", "detectable_format:title",
        "keywords:existence", "keywords:forbidden_words", "keywords:frequency", "keywords:letter_frequency", "language:response_language",
        "length_constraints:nth_paragraph_first_word", "length_constraints:number_paragraphs", "length_constraints:number_sentences",
        "length_constraints:number_words", "punctuation:no_comma", "startend:end_checker", "startend:quotation",
    ],
}
_FAMILY_OF = {cid: fam for fam, ids in FAMILIES.items() for cid in ids}
REQUIRED_ROW_KEYS = ("key", "prompt", "ground_truth", "instruction_ids")
DEFAULT_MAX_PROMPT_CHARS = 12000
DEFAULT_TEMPLATE_DF = 20
TEMPLATE_NGRAMS_FILE = "template_ngrams.json"  # under private/: the boilerplate n-grams the split builder ignored; the contamination scan ignores the same
SALT_BYTES = 32


def family_of(instruction_id: str) -> Optional[str]:
    return _FAMILY_OF.get(instruction_id)


# ---- salt ----------------------------------------------------------------------------------------------------------------------
def salt_sha256(salt: bytes) -> str:
    return hashlib.sha256(salt).hexdigest()


def ensure_split_salt(layout: LabLayout) -> bytes:
    """Create the random 256-bit split salt at ``private/split_salt`` (mode 0600, directory 0700) if it does not exist, and return it.
    Called by ``init``. An existing salt is never replaced."""
    path = layout.split_salt_path
    if path.exists():
        return read_split_salt(layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(layout.private_dir, 0o700)
    salt = secrets.token_bytes(SALT_BYTES)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(salt.hex() + "\n")
    return salt


def read_split_salt(layout: LabLayout) -> bytes:
    try:
        text = layout.split_salt_path.read_text().strip()
        salt = bytes.fromhex(text)
    except (OSError, ValueError):
        raise ScorerError(f"cannot read the split salt at {layout.split_salt_path}") from None
    if len(salt) != SALT_BYTES:
        raise ScorerError("the split salt must be 256 random bits")
    return salt


# ---- rows ----------------------------------------------------------------------------------------------------------------------
def load_rows(path: str | os.PathLike[str], *, unique_keys: bool = True) -> list:
    """Read a JSONL file of data rows (strict JSON: no NaN, no duplicate JSON keys). Raises ScorerError for a missing field, an empty prompt
    or, unless ``unique_keys`` is False, a duplicate ``key`` (the SOURCE dataset repeats keys; the files the builder writes do not)."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScorerError(f"cannot read {p}: {exc.strerror or exc}") from None
    rows, seen = [], set()
    for n, line in enumerate(text.split("\n"), 1):  # NOT splitlines(): U+2028, U+0085 and friends may sit inside a JSON string
        if not line.strip():
            continue
        try:
            row = loads_strict(line)
        except ContractError as exc:
            raise ScorerError(f"{p.name} line {n}: {exc}") from None
        if not isinstance(row, dict):
            raise ScorerError(f"{p.name} line {n}: not a JSON object")
        missing = [k for k in REQUIRED_ROW_KEYS if k not in row]
        if missing:
            raise ScorerError(f"{p.name} line {n}: missing {missing}")
        if not isinstance(row["prompt"], str) or not row["prompt"].strip() or not isinstance(row["instruction_ids"], list) or not row["instruction_ids"]:
            raise ScorerError(f"{p.name} line {n}: empty prompt or no instruction_ids")
        row["key"] = str(row["key"])
        if unique_keys and row["key"] in seen:
            raise ScorerError(f"{p.name} line {n}: duplicate key {row['key']!r}")
        seen.add(row["key"])
        rows.append(row)
    return rows


def _toks(s: str) -> list:
    return re.findall(r"\w+", s.lower())


def ngrams(text: str, n: int = CONTAMINATION_NGRAM) -> set:
    t = _toks(text)
    return {" ".join(t[i:i + n]) for i in range(len(t) - n + 1)}


def normalize_text(s: str) -> str:
    return " ".join(s.split()).casefold()


# A reference prompt of fewer than ``ngram`` tokens has no n-gram, so a longer message that wraps it would pass the n-gram test. The
# contamination scanner (``contamination.MIN_SHORT_REFERENCE_TOKENS``, same value) matches such a reference of at least this many tokens as a
# contiguous run of word tokens; the builder applies the same rule so that what it keeps, the scanner does not flag. Shorter references are
# ordinary phrases and only the whole-text equality test applies.
MIN_SHORT_REFERENCE_TOKENS = 5


def short_phrases(texts: Sequence[str], n: int = CONTAMINATION_NGRAM) -> dict:
    """``{token count: {phrase}}`` of the texts that are too short for an n-gram but long enough to be matched as a run of tokens."""
    out: dict = {}
    for t in texts:
        toks = _toks(t)
        if MIN_SHORT_REFERENCE_TOKENS <= len(toks) < n:
            out.setdefault(len(toks), set()).add(" ".join(toks))
    return out


def has_short_phrase(text: str, phrases: Mapping[int, set]) -> bool:
    """Whether ``text`` contains one of ``phrases`` as a contiguous run of word tokens."""
    if not phrases:
        return False
    toks = _toks(text)
    for size, known in phrases.items():
        for i in range(len(toks) - size + 1):
            if " ".join(toks[i:i + size]) in known:
                return True
    return False


def long_runs(texts: Sequence[str], n: int = CONTAMINATION_NGRAM) -> dict:
    """``{first n-gram: [token tuple]}`` of the texts of at least ``n`` tokens, to find a WHOLE reference prompt inside a longer text. The
    n-gram test misses a whole prompt whose n-grams are all template boilerplate, and base-text stripping hides one that sits inside the
    constraint sentences; containing the entire prompt as a run of tokens is a leak whatever its n-grams are."""
    out: dict = {}
    for t in texts:
        toks = tuple(_toks(t))
        if len(toks) >= n:
            out.setdefault(" ".join(toks[:n]), set()).add(toks)
    return {k: sorted(v) for k, v in out.items()}


def has_long_run(text: str, runs: Mapping[str, Sequence[tuple]], n: int = CONTAMINATION_NGRAM) -> bool:
    """Whether ``text`` contains one of the ``long_runs`` references whole, as a contiguous run of word tokens."""
    if not runs:
        return False
    toks = _toks(text)
    for i in range(len(toks) - n + 1):
        for ref in runs.get(" ".join(toks[i:i + n]), ()):
            if tuple(toks[i:i + len(ref)]) == ref:
                return True
    return False


def _row_has_short_phrase(row: Mapping[str, Any], phrases: Mapping[int, set]) -> bool:
    return bool(phrases) and (has_short_phrase(row["prompt"], phrases) or has_short_phrase(base_text(row), phrases))


def base_text(row: Mapping[str, Any]) -> str:
    """The prompt with the constraint sentences removed (all of it when the row does not say which they are)."""
    t = row["prompt"]
    for part in str(row.get("constraint") or "").split("\t"):
        if part.strip():
            t = t.replace(part, " ")
    return t


@dataclass(frozen=True)
class _Reserved:
    """What later rows may not share with the rows drawn so far: non-template n-grams, and short base prompts (no n-gram) matched as token runs."""

    grams: set
    short: Mapping[int, set]
    runs: Mapping[str, Sequence[tuple]] = field(default_factory=dict)  # whole reference prompts of n tokens or more, found in a row's FULL prompt

    def hits(self, row: Mapping[str, Any]) -> bool:
        return bool(row["_g"] & self.grams) or bool(row["_gf"] & self.grams) or _row_has_short_phrase(row, self.short) or has_long_run(row["prompt"], self.runs)


def base_key(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(normalize_text(base_text(row)).encode("utf-8")).hexdigest()[:16]


def row_id_of(key: str, prompt: str) -> str:
    return hashlib.sha256((key + "\x00" + prompt).encode("utf-8")).hexdigest()[:20]


def _dump(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n" for r in rows)


def _write_text(path: Path, text: str, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if private:
        os.chmod(path.parent, 0o700)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    if private:
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _read_prompts(path: Path) -> list:
    out = []
    try:
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                out.append(loads_strict(line)["prompt"])
    except (OSError, ContractError, KeyError, TypeError) as exc:
        raise ScorerError(f"cannot read test prompts from {path}: {exc}") from None
    return out


# ---- plans ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AssignmentPlan:
    """What one assignment draws. ``families`` is a subset of the role's families. ``n_train`` 0 means every eligible row left after the
    other draws (use it only for the last plan of a role)."""

    assignment_id: str
    role: AssignmentRole
    families: tuple
    n_selection: int
    n_dev: int
    n_train: int

    def __post_init__(self) -> None:
        if self.assignment_id == TEST_ASSIGNMENT_ID:
            raise ScorerError(f"{TEST_ASSIGNMENT_ID!r} is reserved for the test set")
        object.__setattr__(self, "role", AssignmentRole(self.role))
        object.__setattr__(self, "families", tuple(self.families))
        if not self.families or min(self.n_selection, self.n_dev, self.n_train) < 0 or self.n_selection < 1 or self.n_dev < 1:
            raise ScorerError(f"plan {self.assignment_id}: needs families, n_selection >= 1, n_dev >= 1, n_train >= 0")


# ---- the split builder ---------------------------------------------------------------------------------------------------------
def build_splits(rows_jsonl: str | os.PathLike[str], ifbench_test_files: Sequence[str | os.PathLike[str]], out_dir: str | os.PathLike[str], *,
                 salt: bytes, evolution_families: Sequence[str], heldout_families: Sequence[str], plans: Sequence[AssignmentPlan],
                 max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS, ngram: int = CONTAMINATION_NGRAM,
                 template_df: int = DEFAULT_TEMPLATE_DF) -> Path:
    """Write ``data/assignments/<aid>/{train,dev}.jsonl`` and ``private/assignments/<aid>/selection.jsonl`` under the lab root ``out_dir`` and
    return the path of ``private/splits_report.json`` (counts, filters, sha256 of every output; no salt, no absolute path, no timestamp, so two
    runs with the same inputs are byte identical)."""
    layout = LabLayout(Path(out_dir))
    if not isinstance(salt, (bytes, bytearray)) or len(salt) < SALT_BYTES:
        raise ScorerError("salt must be at least 256 random bits")
    if ngram != CONTAMINATION_NGRAM:
        raise ScorerError(f"ngram must be {CONTAMINATION_NGRAM}")
    evo, held = tuple(evolution_families), tuple(heldout_families)
    unknown = sorted((set(evo) | set(held)) - set(FAMILIES))
    if unknown or set(evo) & set(held):
        raise ScorerError(f"family lists must be disjoint known families (unknown {unknown}, overlap {sorted(set(evo) & set(held))})")
    ids = [p.assignment_id for p in plans]
    if not plans or len(set(ids)) != len(ids):
        raise ScorerError("plans must be non-empty with unique assignment ids")
    for p in plans:
        allowed = set(evo) if p.role is AssignmentRole.EVOLUTION else set(held)
        if not set(p.families) <= allowed:
            raise ScorerError(f"plan {p.assignment_id}: families {sorted(set(p.families) - allowed)} are outside its role's partition")
    if not any(p.role is AssignmentRole.EVOLUTION for p in plans) or not any(p.role is AssignmentRole.HELDOUT for p in plans):
        raise ScorerError("plans need at least one evolution and one held-out assignment")
    held_ids = {i for f in held for i in FAMILIES[f]}
    evo_ids = {i for f in evo for i in FAMILIES[f]}

    stats: dict = collections.OrderedDict()
    raw = load_rows(rows_jsonl, unique_keys=False)
    stats["rows_in"] = len(raw)

    # 1. parse, dedup, unsatisfiable, long
    rows, seen_rid, seen_prompt = [], set(), set()
    dup = bad = 0
    for r in sorted(raw, key=lambda r: (r["key"], r["prompt"])):  # which duplicate survives must not depend on the input order
        try:
            ids_gt, kwargs_gt = parse_ground_truth(r["ground_truth"])
        except RewardsError:
            bad += 1
            continue
        rid = row_id_of(r["key"], r["prompt"])
        if rid in seen_rid or r["prompt"] in seen_prompt:
            dup += 1
            continue
        seen_rid.add(rid)
        seen_prompt.add(r["prompt"])
        r = dict(r, _rid=rid, _ids=ids_gt, _kw=kwargs_gt)
        rows.append(r)
    stats["dropped_unparsable_ground_truth"] = bad
    stats["dropped_duplicates"] = dup

    def unsatisfiable(r: Mapping[str, Any]) -> bool:
        for cid, kw in zip(r["_ids"], r["_kw"]):
            if cid == "length_constraints:nth_paragraph_first_word" and kw:
                n, m = kw.get("nth_paragraph"), kw.get("num_paragraphs")
                if n is not None and m is not None and n > m:
                    return True
        return False

    before = len(rows)
    rows = [r for r in rows if not unsatisfiable(r)]
    stats["dropped_unsatisfiable"] = before - len(rows)
    before = len(rows)
    rows = [r for r in rows if len(r["prompt"]) <= max_prompt_chars]
    stats["dropped_long_prompt"] = before - len(rows)

    # 2. decontamination of BASE prompts against every test file, ignoring template n-grams
    test_prompts: list = []
    for tf in ifbench_test_files:
        test_prompts += _read_prompts(Path(tf))
    test_g: set = set()
    for t in test_prompts:
        test_g |= ngrams(t, ngram)
    row_g = [ngrams(base_text(r)[:30000], ngram) for r in rows]
    for r, g in zip(rows, row_g):
        r["_g"] = g
        r["_gf"] = ngrams(r["prompt"][:30000], ngram)  # the scanner reads the FULL prompt, constraint text included
    df_count: collections.Counter = collections.Counter()
    for g in row_g:
        df_count.update(g & test_g)
    template = {g for g, c in df_count.items() if c >= template_df}
    leak = {i for i, g in enumerate(row_g) if (g & test_g) - template}
    test_norm = {normalize_text(t) for t in test_prompts}
    # a row leaks when its full prompt OR its base prompt is a test prompt: a short test prompt has no 13-gram, and appending the
    # constraint sentences to it must not hide it
    exact_leak = {i for i, r in enumerate(rows) if normalize_text(r["prompt"]) in test_norm or normalize_text(base_text(r)) in test_norm}
    # a short test prompt (5 to ngram-1 tokens) has no n-gram either: a row that contains it as a run of tokens leaks it, as the scanner sees it
    test_short = short_phrases(test_prompts, ngram)
    short_leak = {i for i, r in enumerate(rows) if _row_has_short_phrase(r, test_short)}
    # a whole test prompt of n tokens or more inside the row's full prompt (for instance inside its constraint text, which base_text strips)
    # leaks it too, and the train and dev files are visible to the researcher
    test_runs = long_runs(test_prompts, ngram)
    long_leak = {i for i, r in enumerate(rows) if has_long_run(r["prompt"], test_runs, ngram)}
    drop = leak | exact_leak | short_leak | long_leak
    stats["decontam_ngram"] = ngram
    stats["decontam_template_ngrams_ignored"] = len(template)
    stats["dropped_decontam"] = len(drop)
    rows = [r for i, r in enumerate(rows) if i not in drop]

    # 3. family partition
    evo_pool, held_pool = [], []
    mixed = unknown_id = 0
    for r in rows:
        fams = set()
        unmapped = False
        for cid in r["_ids"]:
            f = family_of(cid)
            if f is None:
                unmapped = True
            else:
                fams.add(f)
        if unmapped:
            unknown_id += 1
            continue
        in_evo = all(f in evo for f in fams)
        in_held = all(f in held for f in fams)
        r["_fams"] = frozenset(fams)
        r["family"] = "+".join(sorted(fams))
        if in_evo:
            evo_pool.append(r)
        elif in_held:
            held_pool.append(r)
        else:
            mixed += 1  # spans the partition, or uses a family that is in neither list
    stats["dropped_mixed_or_unlisted_family"] = mixed
    stats["dropped_unknown_constraint_id"] = unknown_id
    stats["evolution_pool"] = len(evo_pool)
    stats["heldout_pool"] = len(held_pool)

    # 4. deterministic order and draws
    def order(r: Mapping[str, Any]) -> str:
        return hashlib.sha256(bytes(salt) + b"|" + r["_rid"].encode()).hexdigest()

    evo_pool.sort(key=order)
    held_pool.sort(key=order)
    for r in evo_pool + held_pool:
        r["_base"] = base_key(r)
    pool_of = {AssignmentRole.EVOLUTION: evo_pool, AssignmentRole.HELDOUT: held_pool}
    ordered_plans = [p for p in plans if p.role is AssignmentRole.HELDOUT] + [p for p in plans if p.role is AssignmentRole.EVOLUTION]
    used: set = set()
    drawn: dict = {p.assignment_id: {} for p in plans}
    reserved_selection: set = set()

    all_rows = evo_pool + held_pool
    template_all: set = set(template)

    def template_in(refs: set) -> set:
        """n-grams of ``refs`` that occur in at least ``template_df`` rows of the pool: boilerplate shared by construction."""
        counts: collections.Counter = collections.Counter()
        for r in all_rows:
            counts.update((r["_g"] | r["_gf"]) & refs)
        return {g for g, c in counts.items() if c >= template_df}

    def reserve(rows_: Sequence[Mapping[str, Any]]) -> "_Reserved":
        refs: set = set()
        for r in rows_:
            refs |= r["_g"]
        tmpl = template_in(refs)
        template_all.update(tmpl)
        # the scanner takes the base prompt (the prompt when the base is empty) as the reference text
        short = short_phrases([base_text(r).strip() or r["prompt"] for r in rows_], ngram)
        # the scanner also takes each reference prompt (and base prompt) whole as a run of tokens, so a reserved prompt placed inside another
        # row's constraint text (which base_text strips, hiding it from the n-gram test) must keep that row out of the public files
        runs = long_runs([t for r in rows_ for t in (r["prompt"], base_text(r).strip()) if t], ngram)
        return _Reserved(refs - tmpl, short, runs)

    def take(plan: AssignmentPlan, n: int, exclude_bases: set, exclude_grams: "_Reserved", unique_base: bool) -> list:
        out, local = [], set()
        for r in pool_of[plan.role]:
            if len(out) == n and n > 0:
                break
            if r["_rid"] in used or not r["_fams"] <= set(plan.families) or r["_base"] in exclude_bases:
                continue
            if exclude_grams.hits(r):
                continue  # shares a non-template 13-gram with a selection or held-out base prompt, or contains a short one whole
            if unique_base and r["_base"] in local:
                continue
            out.append(r)
            local.add(r["_base"])
            used.add(r["_rid"])
        return out

    def draw(plan: AssignmentPlan, name: str, n: int, exclude_bases: set, exclude_grams: "_Reserved", unique_base: bool = False) -> list:
        chosen = take(plan, n, exclude_bases, exclude_grams, unique_base)
        if len(chosen) < n:
            raise ScorerError(f"assignment {plan.assignment_id}: only {len(chosen)} eligible rows for {name}, {n} requested")
        drawn[plan.assignment_id][name] = chosen
        return chosen

    for plan in ordered_plans:  # selection first: its base prompts are reserved for good
        sel = draw(plan, "selection", plan.n_selection, reserved_selection, _Reserved(set(), {}), True)
        reserved_selection |= {r["_base"] for r in sel}
    selection_rows = [r for p in ordered_plans for r in drawn[p.assignment_id]["selection"]]
    grams_after_selection = reserve(selection_rows)
    heldout_bases: set = {r["_base"] for r in selection_rows}
    heldout_rows: list = []
    for plan in [p for p in ordered_plans if p.role is AssignmentRole.HELDOUT]:
        for name, n in (("dev", plan.n_dev), ("train", plan.n_train)):
            chosen = draw(plan, name, n, reserved_selection, grams_after_selection)
            heldout_bases |= {r["_base"] for r in chosen}
            heldout_rows += chosen
    reserved_grams = reserve(selection_rows + heldout_rows)
    for plan in [p for p in ordered_plans if p.role is AssignmentRole.EVOLUTION]:
        for name, n in (("dev", plan.n_dev), ("train", plan.n_train)):
            draw(plan, name, n, heldout_bases, reserved_grams)

    # sanity: the invariants the design relies on
    seen_rows: dict = {}
    evo_files_bases: set = set()
    for plan in plans:
        for name, rs in drawn[plan.assignment_id].items():
            for r in rs:
                if r["_rid"] in seen_rows:
                    raise ScorerError(f"row {r['_rid']} is in two files")
                seen_rows[r["_rid"]] = (plan.assignment_id, name)
                if plan.role is AssignmentRole.EVOLUTION and any(i in held_ids for i in r["_ids"]):
                    raise ScorerError(f"held-out constraint id in evolution assignment {plan.assignment_id}")
                if plan.role is AssignmentRole.HELDOUT and any(i in evo_ids for i in r["_ids"]):
                    raise ScorerError(f"evolution constraint id in held-out assignment {plan.assignment_id}")
                if plan.role is AssignmentRole.EVOLUTION and name != "selection":
                    evo_files_bases.add(r["_base"])
                    if reserved_grams.hits(r):
                        raise ScorerError("an evolution train or dev row shares a 13-gram with a selection or held-out base prompt")
                if plan.role is AssignmentRole.HELDOUT and name != "selection" and grams_after_selection.hits(r):
                    raise ScorerError("a held-out train or dev row shares a 13-gram with a selection base prompt")
    if evo_files_bases & heldout_bases:
        raise ScorerError("a selection or held-out base prompt reached an evolution train or dev file")
    if any(normalize_text(r["prompt"]) in test_norm or normalize_text(base_text(r)) in test_norm or _row_has_short_phrase(r, test_short)
           for d in drawn.values() for rs in d.values() for r in rs):
        raise ScorerError("a test prompt reached an output file")

    # 5. write
    def out_row(r: Mapping[str, Any]) -> dict:
        return {"key": r["_rid"], "source_key": r["key"], "prompt": r["prompt"], "ground_truth": r["ground_truth"],
                "instruction_ids": r["_ids"], "family": r["family"], "base_prompt": base_text(r).strip()}

    report: dict = {"stats": dict(stats), "salt_sha256": salt_sha256(bytes(salt)), "item_mode": ItemMode.SALTED_DRAW.value,
                    "evolution_families": list(evo), "heldout_families": list(held), "assignments": {}}
    for plan in plans:
        aid = plan.assignment_id
        files = {"train": layout.train_data(aid), "dev": layout.dev_data(aid), "selection": layout.selection_data(aid)}
        entry: dict = {"role": plan.role.value, "families": list(plan.families)}
        for name, path in files.items():
            rs = drawn[aid][name]
            _write_text(path, _dump([out_row(r) for r in rs]), private=(name == "selection"))
            fam_counts = collections.Counter(r["family"] for r in rs)
            entry[name] = {"n": len(rs), "path": path.relative_to(layout.root).as_posix(), "sha256": sha256_file(path), "families": dict(sorted(fam_counts.items()))}
        report["assignments"][aid] = entry
    template_path = layout.private_dir / TEMPLATE_NGRAMS_FILE
    _write_text(template_path, json.dumps({"ngram": ngram, "template_df": template_df, "ngrams": sorted(template_all)}) + "\n", private=True)
    report["template_ngrams"] = {"n": len(template_all), "sha256": sha256_file(template_path)}
    report_path = layout.private_dir / "splits_report.json"
    _write_text(report_path, json.dumps(report, sort_keys=True, indent=1) + "\n", private=True)
    return report_path


def build_assignment_manifest(layout: LabLayout, plans: Sequence[AssignmentPlan], *, evolution_families: Sequence[str],
                              heldout_families: Sequence[str], ifrlvr_parquet_sha256: str, ifbench_commit: str, open_instruct_commit: str,
                              clock: Callable[[], float], item_mode: ItemMode = ItemMode.SALTED_DRAW) -> AssignmentManifest:
    """The manifest of the files ``build_splits`` wrote: each assignment's data revisions are the sha256 of its three files, read from disk,
    and ``split_salt_sha256`` is the hash of the salt under ``private/``. The caller freezes it (``write_record`` + ``lab.assignments_frozen``)."""
    assignments = []
    for p in plans:
        aid = p.assignment_id
        revs = {}
        for key, path in ((REV_TRAIN_SHA256, layout.train_data(aid)), (REV_DEV_SHA256, layout.dev_data(aid)), (REV_SELECTION_SHA256, layout.selection_data(aid))):
            if not path.is_file():
                raise ScorerError(f"{path} does not exist; run build_splits first")
            revs[key] = sha256_file(path)
        assert set(revs) == set(DATA_REVISION_KEYS)
        assignments.append(Assignment(aid, p.role, tuple(p.families), revs))
    return AssignmentManifest(
        assignments=tuple(assignments),
        evolution_families=tuple(evolution_families),
        heldout_families=tuple(heldout_families),
        mixed_row_policy=MixedRowPolicy.DROPPED,
        item_mode=item_mode,
        source_revisions={
            SRC_IFRLVR_PARQUET_SHA256: ifrlvr_parquet_sha256,
            SRC_IFBENCH_COMMIT: ifbench_commit,
            SRC_OPEN_INSTRUCT_COMMIT: open_instruct_commit,
            SRC_SPLIT_SALT_SHA256: salt_sha256(read_split_salt(layout)),
        },
        created_utc=utc_iso(clock()),
    )
