"""Adapters over the two third-party constraint-verifier packages, and the per-prompt result they return.

* ``IfbenchUpstream``: the UNCHANGED upstream IFBench verifiers (``evaluation_lib.test_instruction_following_strict`` / ``_loose``),
  used for the test split only. The adapter filters ``None`` kwargs itself (strict scoring mutates them in place, loose scoring crashes
  on them), calls upstream with a one-entry ``{prompt: response}`` dictionary (no prompt-text collisions) and touches nothing else.
* ``IfEvalG``: open-instruct's IFEvalG checkers (the code Ai2 trains with) for dev and selection. All 54 IF-RLVR constraint ids exist
  only there; IFBench's registry has 25 of them.

Neither package is vendored (decision D10). ``scripts/lab_setup_verifiers.sh`` fetches pinned revisions into
``<lab>/third_party/{IFBench,open-instruct}``; the file hashes below are checked BEFORE any file is imported, so a swapped file is
never executed. Missing NLTK data raises ``ScorerError`` from ``preflight()`` instead of reading as reward 0.

Real-verifier parity with the unmodified ``run_eval.py`` is a lead-run acceptance check (needs the checkouts and NLTK data), not a
CI gate; the tests cover the adapter logic with a fake upstream and, when ``LAB_THIRD_PARTY`` is set, with the real packages.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import os
import re
import stat
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from nightshift.lab import rewards
from nightshift.lab.contracts import (
    METRIC_INSTRUCTION_LOOSE,
    METRIC_INSTRUCTION_STRICT,
    METRIC_PROMPT_ALL,
    METRIC_PROMPT_FRACTION,
    METRIC_PROMPT_LOOSE,
    METRIC_PROMPT_STRICT,
    ScorerError,
    Verifier,
)

IFBENCH_COMMIT = "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d"
OPEN_INSTRUCT_COMMIT = rewards.OPEN_INSTRUCT_COMMIT

# sha256 of the IFBench files the adapter executes, at IFBENCH_COMMIT (paths relative to the checkout). Also in scripts/lab_pins.json.
IFBENCH_FILE_SHA256 = {
    "evaluation_lib.py": "e681a4b03a6a9dfb540fbd9ef6b963de503a5ffac557ed611c326ee385850cf6",
    "ifbench/__init__.py": "2a806107f115facaa3e7e6046ca97247237b4b113b514bcdd763b2530141f694",
    "ifbench/classic_instructions.py": "d26e642f2e71e4f57218e5e05f098a22e36311485779baac401161d3228c9973",
    "ifbench/instructions.py": "a76a3b300b531b95200e53427c6b1fbb72b671cf6d358796797c0fa4b6c7c44f",
    "ifbench/instructions_registry.py": "5119f2919261c77566f698f2f367883aff56cf89443bac5a56b6e30ddbd5aef9",
    "ifbench/instructions_util.py": "6996a6efaf6e3c92a2e926881eadc8c0262fb8d06c8cb56394ef7322f69e1930",
}

MODES = ("both", "strict")  # IFBench: strict and loose, or strict only. IFEvalG has no loose variant and ignores the difference.

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


# --------------------------------------------------------------------------------------------------
# result type shared by both backends, the scorer and the fakes
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PromptResult:
    """Verifier output for one prompt: pass/fail per constraint, in the row's constraint order.

    ``loose`` is None for IFEvalG (no loose variant) and for strict-only IFBench runs. ``error`` is True when a checker raised: the
    failing constraint(s) are False and ``n_checker_errors`` counts them, so an exception is never an unexplained 0.
    """

    key: str
    verifier: Verifier
    instruction_ids: tuple
    strict: tuple
    loose: Optional[tuple] = None
    error: bool = False
    n_checker_errors: int = 0

    def __post_init__(self) -> None:
        if not self.instruction_ids:
            raise ScorerError(f"prompt {self.key!r} has no constraints")
        if len(self.strict) != len(self.instruction_ids) or (self.loose is not None and len(self.loose) != len(self.instruction_ids)):
            raise ScorerError(f"prompt {self.key!r}: flags and constraint ids differ in length")

    @property
    def passed_strict(self) -> bool:
        return all(self.strict)

    @property
    def passed_loose(self) -> bool:
        if self.loose is None:
            raise ScorerError("this result has no loose flags")
        return all(self.loose)

    @property
    def fraction(self) -> float:
        """open-instruct's reward: the mean of the per-constraint passes."""
        return sum(self.strict) / len(self.strict)


def aggregate(results: Sequence[PromptResult]) -> dict:
    """Metrics of one seed over the same prompts (names are the ``METRIC_*`` constants).

    IFEvalG results give ``prompt_all`` (every constraint passes; headline) and ``prompt_fraction``. IFBench results give
    ``prompt_strict`` (headline), ``instruction_strict`` and, when loose flags exist, ``prompt_loose`` and ``instruction_loose``.
    """
    if not results:
        raise ScorerError("no results to aggregate")
    kinds = {r.verifier for r in results}
    if len(kinds) != 1:
        raise ScorerError("results come from different verifiers")
    n = len(results)
    if kinds == {Verifier.IFEVALG}:
        return {
            METRIC_PROMPT_ALL: sum(r.passed_strict for r in results) / n,
            METRIC_PROMPT_FRACTION: sum(r.fraction for r in results) / n,
        }
    out = {
        METRIC_PROMPT_STRICT: sum(r.passed_strict for r in results) / n,
        METRIC_INSTRUCTION_STRICT: sum(sum(r.strict) for r in results) / sum(len(r.strict) for r in results),
    }
    if all(r.loose is not None for r in results):
        out[METRIC_PROMPT_LOOSE] = sum(r.passed_loose for r in results) / n
        out[METRIC_INSTRUCTION_LOOSE] = sum(sum(r.loose or ()) for r in results) / sum(len(r.loose or ()) for r in results)
    return out


@runtime_checkable
class VerifierBackend(Protocol):
    """What ``ScoreService`` needs from a verifier. ``rows`` and ``responses`` are in the same order."""

    def score(self, rows: Sequence[Mapping[str, Any]], responses: Sequence[str], *, mode: str = "both") -> list: ...

    def pins(self) -> dict: ...


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def _sha256_regular(path: Path) -> str:
    try:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode):
            raise ScorerError(f"{path}: not a regular file")
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        raise ScorerError(f"{path}: cannot read ({exc.strerror or exc})") from None


def _check_inputs(rows: Sequence[Any], responses: Sequence[Any], mode: str) -> None:
    if mode not in MODES:
        raise ScorerError(f"unknown verifier mode {mode!r}; use one of {MODES}")
    if len(rows) != len(responses):
        raise ScorerError(f"{len(rows)} prompts but {len(responses)} responses: a missing response is never scored as a failure")
    for i, r in enumerate(responses):
        if not isinstance(r, str):
            raise ScorerError(f"response {i} is {type(r).__name__}, not text (missing response)")


def strip_thinking(text: str) -> str:
    """IFBench adapter pre-processing (its README removes reasoning chains before scoring). Glimmer's reasoning parser already keeps
    reasoning out of ``message.content``, so this is a no-op for well-formed output; it is recorded in the pins."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return _THINK_RE.sub("", text).strip()


def _read_git_head(repo: Path) -> Optional[str]:
    """Commit of a checkout, read from .git without running git. None if it cannot be determined."""
    git = repo / ".git"
    try:
        if git.is_file():  # worktree pointer
            text = git.read_text().strip()
            if not text.startswith("gitdir:"):
                return None
            git = (repo / text.split(":", 1)[1].strip()).resolve()
        head = (git / "HEAD").read_text().strip()
        if re.fullmatch(r"[0-9a-f]{40}", head):
            return head
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            f = git / ref
            if f.is_file():
                value = f.read_text().strip()
                return value if re.fullmatch(r"[0-9a-f]{40}", value) else None
            packed = git / "packed-refs"
            if packed.is_file():
                for line in packed.read_text().splitlines():
                    if line.endswith(" " + ref):
                        return line.split()[0]
    except OSError:
        return None
    return None


def check_nltk_data(resources: Sequence[str] = rewards.NLTK_RESOURCES) -> None:
    """Raise ScorerError if nltk or one of its data packages is missing."""
    try:
        import nltk  # type: ignore
    except ImportError as exc:
        raise ScorerError(f"nltk is not installed ({exc}); the verifiers cannot run") from None
    for res in resources:
        try:
            nltk.data.find(res)
        except LookupError:
            raise ScorerError(
                f"NLTK resource {res} is missing; a checker would raise and read as reward 0. "
                "Run: python -m nltk.downloader -d $NLTK_DATA punkt punkt_tab stopwords averaged_perceptron_tagger_eng"
            ) from None


def _use_nltk_data(path: Optional[Path]) -> None:
    if path is None:
        return
    os.environ["NLTK_DATA"] = str(path)
    try:
        import nltk  # type: ignore

        if str(path) not in nltk.data.path:
            nltk.data.path.insert(0, str(path))
    except ImportError:
        pass


# --------------------------------------------------------------------------------------------------
# IFEvalG (dev and selection)
# --------------------------------------------------------------------------------------------------
class IfEvalG:
    """IFEvalG checkers from a pinned open-instruct checkout. ``open_instruct_path`` is the checkout root (it holds
    ``open_instruct/IFEvalG``) or the IFEvalG directory itself (the flat layout on a rental)."""

    def __init__(self, open_instruct_path: str | os.PathLike[str], *, nltk_data: Optional[str | os.PathLike[str]] = None,
                 expected_sha256: Optional[Mapping[str, str]] = None):
        root = Path(open_instruct_path)
        nested = root / "open_instruct" / "IFEvalG"
        self.directory = nested if nested.is_dir() else root
        self._expected = dict(rewards.IFEVALG_SHA256 if expected_sha256 is None else expected_sha256)
        try:
            self._hashes = rewards.verify_ifevalg_dir(self.directory, self._expected)
        except rewards.RewardsError as exc:
            raise ScorerError(str(exc)) from None
        self._nltk_data = Path(nltk_data) if nltk_data is not None else None
        self._checker: Optional[rewards.Checker] = None
        self._lock = threading.Lock()

    def pins(self) -> dict:
        out = {"open_instruct_commit": OPEN_INSTRUCT_COMMIT, "remove_thinking": "open_instruct.ground_truth_utils"}
        for name, digest in sorted(self._hashes.items()):
            out["ifevalg_" + name.replace(".", "_")] = digest
        return out

    def _load(self) -> rewards.Checker:
        with self._lock:
            if self._checker is None:
                _use_nltk_data(self._nltk_data)
                try:
                    self._checker = rewards.Checker(self.directory, expected=self._expected)
                except rewards.RewardsError as exc:
                    raise ScorerError(str(exc)) from None
                except Exception as exc:  # import of the pinned files failed (a missing dependency)
                    raise ScorerError(f"cannot import the IFEvalG checkers: {type(exc).__name__}: {exc}") from None
            return self._checker

    def preflight(self) -> None:
        _use_nltk_data(self._nltk_data)
        check_nltk_data()
        try:
            rewards.preflight(self._load())
        except rewards.RewardsError as exc:
            raise ScorerError(str(exc)) from None

    def score(self, rows: Sequence[Mapping[str, Any]], responses: Sequence[str], *, mode: str = "both") -> list:
        _check_inputs(rows, responses, mode)
        checker = self._load()
        out = []
        for row, text in zip(rows, responses):
            key = str(row.get("key"))
            if "ground_truth" not in row:
                raise ScorerError(f"row {key!r} has no ground_truth")
            try:
                ids, flags, n_err = checker.check(text, row["ground_truth"])
            except rewards.RewardsError as exc:
                raise ScorerError(f"row {key!r}: {exc}") from None
            declared = row.get("instruction_ids")
            if declared is not None and list(declared) != ids:
                raise ScorerError(f"row {key!r}: instruction_ids disagree with ground_truth")
            out.append(PromptResult(key, Verifier.IFEVALG, tuple(ids), tuple(flags), None, n_err > 0, n_err))
        return out


# --------------------------------------------------------------------------------------------------
# IFBench upstream (test split)
# --------------------------------------------------------------------------------------------------
_IFBENCH_LOCK = threading.RLock()
_IFBENCH_LOADED: dict = {}  # resolved repo path -> evaluation_lib module


def _reset_for_tests() -> None:
    """Forget the imported IFBench modules (tests that load a different checkout call this)."""
    with _IFBENCH_LOCK:
        _IFBENCH_LOADED.clear()
        for name in [n for n in sys.modules if _is_ifbench_module(n)]:
            sys.modules.pop(name, None)


def _is_ifbench_module(name: str) -> bool:
    return name in ("evaluation_lib", "ifbench") or name.startswith("ifbench.")


class IfbenchUpstream:
    """Unchanged upstream IFBench verifiers from a pinned checkout (``third_party/IFBench``)."""

    def __init__(self, repo_path: str | os.PathLike[str], *, nltk_data: Optional[str | os.PathLike[str]] = None,
                 strip_thinking: bool = True, expected_sha256: Optional[Mapping[str, str]] = None,
                 expected_commit: Optional[str] = IFBENCH_COMMIT):
        self.repo = Path(repo_path).resolve()
        if not self.repo.is_dir():
            raise ScorerError(f"IFBench checkout {self.repo} does not exist; run scripts/lab_setup_verifiers.sh")
        want = dict(IFBENCH_FILE_SHA256 if expected_sha256 is None else expected_sha256)
        self._hashes = {}
        for rel, digest in sorted(want.items()):
            got = _sha256_regular(self.repo / rel)
            if got != digest:
                raise ScorerError(f"IFBench file {rel} in {self.repo} does not match the pinned sha256 (IFBench {expected_commit or 'pin'})")
            self._hashes[rel] = got
        commit = _read_git_head(self.repo)
        if expected_commit is not None and commit is not None and commit != expected_commit:
            raise ScorerError(f"IFBench checkout is at {commit}, expected {expected_commit}")
        # unreadable .git (a copied tree): the pinned file hashes above are what identifies the revision
        self._commit = commit or expected_commit or "unknown"
        self.strip_thinking = bool(strip_thinking)
        self._nltk_data = Path(nltk_data) if nltk_data is not None else None
        self._ev: Any = None

    def pins(self) -> dict:
        out = {"ifbench_commit": self._commit, "strip_thinking": "true" if self.strip_thinking else "false"}
        for rel, digest in sorted(self._hashes.items()):
            out["ifbench_" + rel.replace("/", "_").replace(".", "_")] = digest
        return out

    def _foreign_modules(self) -> list:
        """Imported IFBench modules (``evaluation_lib``, ``ifbench``, ``ifbench.*``) whose file is not inside this checkout."""
        out = []
        for name in sorted(n for n in sys.modules if _is_ifbench_module(n)):
            origin = getattr(sys.modules[name], "__file__", None)
            try:
                inside = origin is not None and Path(origin).resolve().is_relative_to(self.repo)
            except (OSError, ValueError):
                inside = False
            if not inside:
                out.append(name)
        return out

    def _load(self) -> Any:
        with _IFBENCH_LOCK:
            if self._ev is not None:
                return self._ev
            if _IFBENCH_LOADED:
                have = next(iter(_IFBENCH_LOADED))
                if have != str(self.repo):
                    raise ScorerError(f"IFBench is already imported from {have}; one process can use one checkout")
                self._ev = _IFBENCH_LOADED[have]
                return self._ev
            # NLTK data must exist BEFORE the import: ifbench/instructions_util.py calls nltk.download() at import time otherwise.
            _use_nltk_data(self._nltk_data)
            check_nltk_data()
            # import_module() returns a module that is already in sys.modules whatever sys.path says: code imported from anywhere else would be
            # scored under this checkout's pinned hashes
            stale = self._foreign_modules()
            if stale:
                raise ScorerError(f"IFBench modules are already imported from outside the pinned checkout {self.repo}: {', '.join(stale)}")
            sys.path.insert(0, str(self.repo))  # evaluation_lib.py lives at the repo root and is not in the wheel
            try:
                ev = importlib.import_module("evaluation_lib")
            except Exception as exc:
                for name in [n for n in sys.modules if _is_ifbench_module(n)]:
                    sys.modules.pop(name, None)
                raise ScorerError(f"cannot import the IFBench verifiers: {type(exc).__name__}: {exc}") from None
            finally:
                try:
                    sys.path.remove(str(self.repo))
                except ValueError:
                    pass
            _IFBENCH_LOADED[str(self.repo)] = ev
            self._ev = ev
            return ev

    def preflight(self) -> None:
        self._load()

    @staticmethod
    def _clean_kwargs(row: Mapping[str, Any]) -> list:
        """Drop None kwargs. Upstream strict scoring does this in place; loose does not and raises TypeError on them."""
        kw = row["kwargs"]
        return [{k: v for k, v in (d or {}).items() if v is not None} for d in kw]

    def score(self, rows: Sequence[Mapping[str, Any]], responses: Sequence[str], *, mode: str = "both") -> list:
        _check_inputs(rows, responses, mode)
        ev = self._load()
        out = []
        for row, text in zip(rows, responses):
            key = str(row.get("key"))
            for field in ("prompt", "instruction_id_list", "kwargs"):
                if field not in row:
                    raise ScorerError(f"row {key!r} has no {field}")
            ids = list(row["instruction_id_list"])
            if len(ids) != len(row["kwargs"]):
                raise ScorerError(f"row {key!r}: kwargs and instruction_id_list differ in length")
            if not ids:
                raise ScorerError(f"row {key!r} has no constraints")
            body = strip_thinking(text) if self.strip_thinking else text
            p2r = {row["prompt"]: body}  # one entry: no cross-row prompt collisions, upstream untouched
            try:
                flags = {}
                for name, fn in (("strict", ev.test_instruction_following_strict), ("loose", ev.test_instruction_following_loose)):
                    if name == "loose" and mode == "strict":
                        continue
                    inp = ev.InputExample(key=row["key"], instruction_id_list=list(ids), prompt=row["prompt"],
                                          kwargs=copy.deepcopy(self._clean_kwargs(row)))
                    res = fn(inp, p2r)
                    flags[name] = tuple(bool(f) for f in res.follow_instruction_list)
                    if len(flags[name]) != len(ids):
                        raise ScorerError(f"row {key!r}: upstream returned {len(flags[name])} flags for {len(ids)} constraints")
                out.append(PromptResult(key, Verifier.IFBENCH_UPSTREAM, tuple(ids), flags["strict"], flags.get("loose"), False, 0))
            except ScorerError:
                raise
            except Exception:
                # a checker raised for this prompt: it fails in both variants and is counted, never an unexplained 0
                dead = tuple([False] * len(ids))
                out.append(PromptResult(key, Verifier.IFBENCH_UPSTREAM, tuple(ids), dead, None if mode == "strict" else dead, True, 1))
        return out


def preflight(*backends: Any) -> None:
    """Run ``preflight()`` of each backend that has one. Call once before scoring anything."""
    for b in backends:
        fn = getattr(b, "preflight", None)
        if fn is not None:
            fn()


def git_commit_of(path: str | os.PathLike[str]) -> Optional[str]:
    """Commit of a checkout (reads .git, falls back to ``git rev-parse``). For the setup script's report, not for scoring."""
    c = _read_git_head(Path(path))
    if c:
        return c
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
