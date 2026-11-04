"""Scan of a job's training data (and recipe tarball) against selection, held-out and test prompts, on the control host.

The glimmer helper commands are the only training entry points; they write the processed dataset to ``train_data.jsonl`` in the job output
and the trainer fetches it. Before any selection score the trial runner calls ``scan_job``: a hit makes the trial CONTAMINATED and the adapter is
never scored on selection (docs/harness-design.md, "Closing the leaks", item 3). A recipe that trains on data it did not route through a helper
has no ``train_data`` and fails earlier.

What counts as a hit, per training ROW (one JSON line; a non-JSON line is a row of raw text; one text file of a tarball is a row):

* exact: some string value of the row equals a reference prompt (whitespace and case normalised);
* 13-gram: some string value of the row shares a 13-gram of word tokens (``\\w+``, lower case) with a reference prompt. A reference
  prompt of fewer than 13 tokens has no 13-gram; it counts as a hit when it occurs whole, as a run of tokens, inside a string value
  (references of fewer than ``MIN_SHORT_REFERENCE_TOKENS`` tokens are matched by the exact test only).

References are the prompts of every selection file, of every held-out assignment (selection, dev and train) and of the IFBench test file. The
one exception is a trial that runs ON a held-out assignment (a final-phase trial): the train and dev files of its own assignment are the public
data it is assigned and legitimately trains on, so they are not references for that trial (its own selection file, the other held-out
assignments' files and the test file still are). ``scan_job`` finds the trial's assignment in its spec. For
rows that carry a ``base_prompt`` (selection rows do) the n-grams come from the base prompt, with the constraint sentences removed. N-grams that
occur in at least ``template_df`` rows of the public train and dev files of the EVOLUTION assignments are template boilerplate shared by
construction and are ignored (the same rule the split builder uses against the test files). Because ``ifdata.build_splits`` keeps every
selection and held-out base prompt out of the train and dev pools, a hit in a training dataset is a real hit, not a shared base prompt.

Nothing is extracted to disk and no text reaches the report: ``ContaminationReport`` holds counts and file hashes only.
"""

from __future__ import annotations

import collections
import hashlib
import io
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

from nightshift.lab.contracts import (
    CONTAMINATION_NGRAM,
    TrialSpec,
    AssignmentManifest,
    AssignmentRole,
    ContaminationReport,
    ContractError,
    FileRef,
    LabLayout,
    ScorerError,
    loads_strict,
    read_record,
    read_regular_bytes,
    utc_iso,
)
from nightshift.lab.ifdata import DEFAULT_TEMPLATE_DF, TEMPLATE_NGRAMS_FILE, has_long_run, long_runs, ngrams, normalize_text

MAX_TRAIN_DATA_BYTES = 512 * 1024 * 1024
MAX_TAR_MEMBERS = 5000  # the recipe tarball format caps entries here (docs/harness-design.md, "File formats")
MAX_TAR_MEMBER_BYTES = 8 * 1024 * 1024
MAX_TAR_TOTAL_BYTES = 200 * 1024 * 1024
_TOKEN = re.compile(r"\w+")
# A reference prompt shorter than the n-gram size yields no n-gram, so wrapping it in a longer message would hide it from the n-gram test.
# Such a reference is instead matched as a contiguous run of word tokens inside the text. Below this many tokens a run is an ordinary
# phrase rather than a prompt, so only the exact test applies.
MIN_SHORT_REFERENCE_TOKENS = 5


@dataclass(frozen=True)
class ScanCounts:
    n_rows: int
    n_exact_hits: int
    n_ngram_hits: int

    def __add__(self, other: "ScanCounts") -> "ScanCounts":
        return ScanCounts(self.n_rows + other.n_rows, self.n_exact_hits + other.n_exact_hits, self.n_ngram_hits + other.n_ngram_hits)


def string_leaves(value: Any) -> Iterator[str]:
    """Every string in a JSON value (values and keys of nested objects and lists)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield from string_leaves(k)
            yield from string_leaves(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from string_leaves(v)


class ContaminationScanner:
    """Holds the reference sets. ``exact_texts`` are compared whole (normalised); ``gram_texts`` supply the n-grams. ``template_ngrams`` are
    ignored; when not given they are the reference n-grams that occur in at least ``template_df`` of the ``background`` texts."""

    def __init__(self, exact_texts: Iterable[str], gram_texts: Iterable[str], *, background: Iterable[str] = (),
                 template_ngrams: Optional[Iterable[str]] = None, template_df: int = DEFAULT_TEMPLATE_DF,
                 n: int = CONTAMINATION_NGRAM, reference_sha256: Sequence[str] = ()):
        if n != CONTAMINATION_NGRAM:
            raise ScorerError(f"n must be {CONTAMINATION_NGRAM}")
        self.n = n
        self.reference_sha256 = tuple(reference_sha256)
        exact_texts_list = list(exact_texts)
        self._exact = {normalize_text(t) for t in exact_texts_list if t.strip()}
        gram_texts = list(gram_texts)
        grams: set = set()
        for t in gram_texts:
            grams |= ngrams(t, n)
        if template_ngrams is not None:
            self.template_ngrams = frozenset(template_ngrams)
        else:  # no list from the split builder: n-grams of the references that occur in >= template_df background rows
            counts: collections.Counter = collections.Counter()
            for t in background:
                counts.update(ngrams(t, n) & grams)
            self.template_ngrams = frozenset(g for g, c in counts.items() if c >= template_df)
        self._grams = frozenset(grams - self.template_ngrams)
        short: dict = {}
        for t in gram_texts:
            toks = _TOKEN.findall(t.lower())
            if MIN_SHORT_REFERENCE_TOKENS <= len(toks) < n:
                short.setdefault(len(toks), set()).add(" ".join(toks))
        self._short = {k: frozenset(v) for k, v in short.items()}
        # A whole reference prompt (or base prompt) of n tokens or more, wrapped in more text, is a hit even when its base text supplies no
        # n-gram (a selection row whose base is short) or all of its n-grams are template boilerplate.
        self._runs = long_runs(list(exact_texts_list), n)
        self._variant_builder: Optional[Callable[[str], "ContaminationScanner"]] = None
        self._variants: dict = {}
        if not self._exact and not self._grams and not self._short and not self._runs:
            raise ScorerError("the contamination scanner has no reference prompts")

    def for_assignment(self, assignment_id: str) -> "ContaminationScanner":
        """The scanner for a trial that runs on this held-out assignment: its own train and dev files are not references. A scanner that was
        not built by ``build_scanner`` has no per-assignment variant and is returned unchanged."""
        if self._variant_builder is None:
            return self
        if assignment_id not in self._variants:
            self._variants[assignment_id] = self._variant_builder(assignment_id)
        return self._variants[assignment_id]

    @property
    def n_reference_ngrams(self) -> int:
        return len(self._grams)

    def has_exact(self, text: str) -> bool:
        return bool(text.strip()) and normalize_text(text) in self._exact

    def has_ngram(self, text: str) -> bool:
        toks = _TOKEN.findall(text.lower())
        n = self.n
        grams = self._grams
        for i in range(len(toks) - n + 1):
            if " ".join(toks[i:i + n]) in grams:
                return True
        if has_long_run(text, self._runs, n):
            return True
        for size, phrases in self._short.items():
            for i in range(len(toks) - size + 1):
                if " ".join(toks[i:i + size]) in phrases:
                    return True
        return False

    def scan_row(self, leaves: Iterable[str]) -> tuple:
        """``(exact, ngram)`` for one row given its string values."""
        exact = ngram = False
        for leaf in leaves:
            if not exact and self.has_exact(leaf):
                exact = True
            if not ngram and self.has_ngram(leaf):
                ngram = True
            if exact and ngram:
                break
        return exact, ngram

    def scan_jsonl(self, data: bytes) -> ScanCounts:
        n_rows = n_exact = n_gram = 0
        for line in data.decode("utf-8", errors="replace").split("\n"):
            if not line.strip():
                continue
            n_rows += 1
            try:
                leaves: Iterable[str] = list(string_leaves(json.loads(line)))
            except ValueError:
                leaves = [line]  # not JSON: a row of raw text
            exact, gram = self.scan_row(leaves)
            n_exact += exact
            n_gram += gram
        return ScanCounts(n_rows, n_exact, n_gram)

    def scan_tarball(self, data: bytes) -> ScanCounts:
        """Scan the text files of a recipe tarball (gzip tar). Each file is one row for the n-gram test; each of its lines is a candidate
        for the exact test. Nothing is extracted. A member that is not a regular file or a directory is an error: the control API
        refuses such tarballs, so one here means something bypassed it."""
        n_rows = n_exact = n_gram = 0
        total = 0
        try:
            tar = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
        except (tarfile.TarError, OSError, EOFError) as exc:
            raise ScorerError(f"recipe tarball is not a readable gzip tar: {exc}") from None
        with tar:
            members = 0
            for m in tar:
                members += 1
                if members > MAX_TAR_MEMBERS:
                    raise ScorerError("recipe tarball has too many entries")
                if m.isdir():
                    continue
                if not m.isreg():
                    raise ScorerError(f"recipe tarball member {m.name!r} is not a regular file")
                if m.size > MAX_TAR_MEMBER_BYTES:
                    raise ScorerError(f"recipe tarball member {m.name!r} is {m.size} bytes (cap {MAX_TAR_MEMBER_BYTES})")
                total += m.size
                if total > MAX_TAR_TOTAL_BYTES:
                    raise ScorerError("recipe tarball is too large to scan")
                fh = tar.extractfile(m)
                text = (fh.read() if fh else b"").decode("utf-8", errors="replace")
                n_rows += 1
                exact = any(self.has_exact(line) for line in text.splitlines()) or self.has_exact(text)
                gram = self.has_ngram(text)
                n_exact += exact
                n_gram += gram
        return ScanCounts(n_rows, n_exact, n_gram)


# --------------------------------------------------------------------------------------------------
# building the scanner from the lab
# --------------------------------------------------------------------------------------------------
def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _prompts_of(path: Path) -> list:
    out = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            try:
                row = loads_strict(line)
                row["prompt"]
            except (ContractError, KeyError, TypeError) as exc:
                raise ScorerError(f"{path.name}: a row without a prompt or invalid JSON ({exc})") from None
            out.append(row)
    return out


def build_scanner(layout: LabLayout, manifest: AssignmentManifest, *, test_files: Optional[Sequence[Path]] = None,
                  template_df: int = DEFAULT_TEMPLATE_DF, own_assignment: Optional[str] = None) -> ContaminationScanner:
    """The scanner for this lab: selection files of every assignment, all files of every held-out assignment and the test file(s) as
    references. Template n-grams come from ``private/template_ngrams.json`` (written by the split builder); a lab without that file falls
    back to the n-grams that occur in at least ``template_df`` rows of the evolution assignments' public train and dev files. Reads files
    directly (control host); a missing reference file is a ScorerError, never a silent skip. ``own_assignment`` names the held-out assignment
    a trial runs on: its train and dev files are read (and hashed) but are not references; see ``ContaminationScanner.for_assignment``."""
    exact: list = []
    grams: list = []
    background: list = []
    shas: list = []

    def need(path: Path) -> Path:
        if not path.is_file():
            raise ScorerError(f"contamination reference {path} does not exist")
        shas.append(_file_sha256(path))
        return path

    def add_reference(row: Mapping[str, Any]) -> None:
        exact.append(row["prompt"])
        if row.get("base_prompt"):
            exact.append(row["base_prompt"])
        grams.append(row.get("base_prompt") or row["prompt"])

    for a in manifest.assignments:
        aid = a.assignment_id
        for row in _prompts_of(need(layout.selection_data(aid))):
            add_reference(row)
        for p in (layout.train_data(aid), layout.dev_data(aid)):
            if a.role is AssignmentRole.HELDOUT:
                rows = _prompts_of(need(p))
                if aid != own_assignment:
                    for row in rows:
                        add_reference(row)
            elif p.is_file():
                background.extend(row.get("base_prompt") or row["prompt"] for row in _prompts_of(p))
    for tf in (list(test_files) if test_files is not None else [layout.test_data()]):
        for row in _prompts_of(need(Path(tf))):
            exact.append(row["prompt"])
            grams.append(row["prompt"])
    template: Optional[list] = None
    tpath = layout.private_dir / TEMPLATE_NGRAMS_FILE
    if tpath.is_file():
        need(tpath)
        try:
            template = list(loads_strict(tpath.read_text(encoding="utf-8"))["ngrams"])
        except (ContractError, KeyError, TypeError, OSError) as exc:
            raise ScorerError(f"cannot read {TEMPLATE_NGRAMS_FILE}: {exc}") from None
    scanner = ContaminationScanner(exact, grams, background=background, template_ngrams=template, template_df=template_df,
                                   reference_sha256=sorted(set(shas)))
    if own_assignment is None:
        scanner._variant_builder = lambda aid: build_scanner(layout, manifest, test_files=test_files, template_df=template_df, own_assignment=aid)
    return scanner


# --------------------------------------------------------------------------------------------------
# the call the trial runner makes
# --------------------------------------------------------------------------------------------------
def _read_ref(layout: LabLayout, ref: FileRef, max_bytes: int) -> bytes:
    try:
        data = read_regular_bytes(layout.root, ref.path, max_bytes=max_bytes)
    except (ContractError, OSError) as exc:
        raise ScorerError(f"cannot read {ref.path} for the contamination scan: {exc}") from None
    if hashlib.sha256(data).hexdigest() != ref.sha256:
        raise ScorerError(f"{ref.path} does not match its recorded sha256")
    return data


def _trial_assignment(layout: LabLayout, trial_id: str) -> Optional[str]:
    """The assignment the trial runs on, from its spec; None when there is no readable spec (the strictest scan then applies)."""
    try:
        return read_record(layout.trial_spec_path(trial_id), TrialSpec).assignment_id
    except (OSError, ContractError):
        return None


def scan_job(layout: LabLayout, manifest: AssignmentManifest, *, trial_id: str, job_id: str, train_data: FileRef, recipe: Optional[FileRef] = None,
             clock: Callable[[], float], scanner: Optional[ContaminationScanner] = None, assignment_id: Optional[str] = None) -> ContaminationReport:
    """Scan one job's processed training data and, when given, its recipe tarball. Returns the ``ContaminationReport`` (``scanned`` is the
    training data; the counts cover both). Building the scanner reads every reference file: pass a prebuilt ``scanner`` to scan many jobs."""
    own = assignment_id or _trial_assignment(layout, trial_id)
    if own is not None and not any(a.assignment_id == own and a.role is AssignmentRole.HELDOUT for a in manifest.assignments):
        own = None  # only a trial on a held-out assignment is assigned held-out train and dev data
    if scanner is None:
        scanner = build_scanner(layout, manifest, own_assignment=own)
    elif own is not None:
        scanner = scanner.for_assignment(own)
    counts = scanner.scan_jsonl(_read_ref(layout, train_data, MAX_TRAIN_DATA_BYTES))
    if recipe is not None:
        counts = counts + scanner.scan_tarball(_read_ref(layout, recipe, MAX_TAR_TOTAL_BYTES))
    return ContaminationReport(
        trial_id=trial_id, job_id=job_id, scanned=train_data, n_rows=counts.n_rows, n_exact_hits=counts.n_exact_hits,
        n_ngram_hits=counts.n_ngram_hits, ngram_n=scanner.n, reference_sha256=scanner.reference_sha256,
        passed=counts.n_exact_hits == 0 and counts.n_ngram_hits == 0, recorded_utc=utc_iso(clock()),
    )
