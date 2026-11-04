"""Training-time reward for IF-RLVR rows, for researcher recipes (and the IFEvalG loader the scorer shares).

STANDALONE ON PURPOSE: this file is staged to the rented GPU as ``/opt/lab/rewards.py`` by ``hostkit/stage.sh`` next to the three
pinned IFEvalG files in ``/opt/lab/ifevalg`` (``IFEVALG_PATH``), where the ``nightshift`` package does not exist. It therefore
imports the standard library only, never ``nightshift``. ``nightshift.lab.verifiers`` imports this module on the control host.

The third-party checkers are open-instruct's ``IFEvalG`` files (Apache-2.0, three files). They are NOT vendored in this repository
(decision D10): ``scripts/lab_setup_verifiers.sh`` fetches the pinned revision and this module refuses to import files whose
sha256 differs from the pins below.

Public API (mirrors ``open_instruct.ground_truth_utils.IFEvalVerifier`` at open-instruct ``b781e78``):

    verify_constraints(response, ground_truth) -> list[bool]     per-constraint pass/fail
    reward(response, ground_truth, mode="fraction") -> float     in [0, 1]
    reward_batch(responses, ground_truths, mode="fraction")      list[float]
    error_counts() -> dict[str, int]                              checker exceptions swallowed so far (check it after a run!)
    preflight()                                                   raises RewardsError if the NLTK data is missing
    degenerate_floor(ground_truths)                               what constant answers such as "Yes." earn (the floor to beat)

``ground_truth`` is the dataset column as stored: a Python-literal string such as
``"[{'instruction_id': ['a:b', 'c:d'], 'kwargs': [None, {'last_word': 'x'}]}]"``.

Deviations from open-instruct, all deliberate:
  * a checker that raises scores False for that constraint and is counted in ``error_counts()`` (upstream would raise);
  * langdetect is seeded (``DetectorFactory.seed = 0``) so language checks are deterministic. This is process-global: in the
    control-host scorer it also applies to the IFBench language checks;
  * ``mode="all"`` and ``mode="hybrid"`` are additions; ``mode="fraction"`` is the upstream score.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import sys
import threading
import types
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

OPEN_INSTRUCT_COMMIT = "b781e782a8fc530577f235aefe282dd046cb8ad2"

# The files the rental needs next to this one. hostkit/stage.sh uploads exactly these into /opt/lab/ifevalg and verifies the pins.
REQUIRED_FILES = ("instructions.py", "instructions_registry.py", "instructions_util.py")

# sha256 of the three IFEvalG files at OPEN_INSTRUCT_COMMIT (open_instruct/IFEvalG/<name>). Kept in scripts/lab_pins.json as well.
IFEVALG_SHA256 = {
    "instructions.py": "dfc4629afac5d308fd170e117a0005a02188f65fbb15635e8f78ede7828ec205",
    "instructions_registry.py": "5436911d616a7c38afcbbc18312551d8a9ed3f06bca44d74920071c1690c7fc3",
    "instructions_util.py": "b0b21dd189474071c027cc377e7132dcbf7a6c921def4440b3d1df8eccffdcd9",
}

ENV_IFEVALG_PATH = "IFEVALG_PATH"

# NLTK resources the sentence, word and tagging checkers need. Without them a checker raises LookupError, which a wrapper that
# catches exceptions would read as reward 0: preflight() makes the missing data loud.
NLTK_RESOURCES = (
    "tokenizers/punkt",
    "tokenizers/punkt_tab",
    "corpora/stopwords",
    "taggers/averaged_perceptron_tagger_eng",
)

_PREFLIGHT_GT = (
    "[{'instruction_id': ['length_constraints:number_sentences'], "
    "'kwargs': [{'num_sentences': 3, 'relation': 'at least'}]}]"
)

_LOCK = threading.RLock()
_LOADED: dict[str, Any] = {}  # resolved directory -> instructions_registry module (only one directory per process)


class RewardsError(RuntimeError):
    """Missing or unpinned IFEvalG files, missing NLTK data, or an unparsable ground truth."""


def _sha256_regular(path: Path) -> str:
    st = os.lstat(path)  # a symbolic link or a special file is not a pinned file
    if not stat.S_ISREG(st.st_mode):
        raise RewardsError(f"{path}: not a regular file")
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ifevalg_hashes(directory: str | os.PathLike[str]) -> dict[str, str]:
    """sha256 of each required file in ``directory``. Raises RewardsError if one is missing or not a regular file."""
    d = Path(directory)
    out = {}
    for name in REQUIRED_FILES:
        try:
            out[name] = _sha256_regular(d / name)
        except OSError as exc:
            raise RewardsError(f"IFEvalG file {d / name} is not readable: {exc}") from None
    return out


def verify_ifevalg_dir(directory: str | os.PathLike[str], expected: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Hash the three files and require them to equal the pins. Returns the hashes. Runs BEFORE any import of the files."""
    want = dict(IFEVALG_SHA256 if expected is None else expected)
    got = ifevalg_hashes(directory)
    bad = sorted(name for name in REQUIRED_FILES if got[name] != want.get(name))
    if bad:
        raise RewardsError(f"IFEvalG files {bad} in {directory} do not match the pinned sha256 (open-instruct {OPEN_INSTRUCT_COMMIT})")
    return got


def _seed_langdetect() -> None:
    try:
        import langdetect  # type: ignore

        langdetect.DetectorFactory.seed = 0
    except Exception:  # langdetect absent: a language checker will raise and be counted
        pass


def load_ifevalg(directory: Optional[str | os.PathLike[str]] = None, *, expected: Optional[Mapping[str, str]] = None) -> Any:
    """Import the pinned IFEvalG files and return their ``instructions_registry`` module.

    The files hard-code ``from open_instruct.IFEvalG import ...``. A stub ``open_instruct`` package whose ``IFEvalG`` subpackage points
    at ``directory`` makes them importable unchanged, without importing the real (torch-heavy) ``open_instruct`` package. One
    directory per process: a second, different directory raises. ``directory`` defaults to ``$IFEVALG_PATH``.
    """
    raw = directory if directory is not None else os.environ.get(ENV_IFEVALG_PATH)
    if not raw:
        raise RewardsError(f"no IFEvalG directory: pass one or set {ENV_IFEVALG_PATH}")
    d = Path(raw).resolve()
    with _LOCK:
        if _LOADED:
            have = next(iter(_LOADED))
            if have != str(d):
                raise RewardsError(f"IFEvalG is already loaded from {have}; one process can load only one directory (asked for {d})")
            return _LOADED[have]
        verify_ifevalg_dir(d, expected)
        existing = sys.modules.get("open_instruct")
        if existing is not None and not getattr(existing, "__lab_stub__", False):
            raise RewardsError("a real 'open_instruct' package is already imported; the lab loads the IFEvalG files through a stub package")
        pkg = types.ModuleType("open_instruct")
        pkg.__path__ = []  # type: ignore[attr-defined]
        pkg.__lab_stub__ = True  # type: ignore[attr-defined]
        sub = types.ModuleType("open_instruct.IFEvalG")
        sub.__path__ = [str(d)]  # type: ignore[attr-defined]
        sys.modules["open_instruct"] = pkg
        sys.modules["open_instruct.IFEvalG"] = sub
        try:
            import importlib

            registry = importlib.import_module("open_instruct.IFEvalG.instructions_registry")
        except BaseException:
            for name in [n for n in sys.modules if n == "open_instruct" or n.startswith("open_instruct.")]:
                sys.modules.pop(name, None)
            raise
        _seed_langdetect()
        _LOADED[str(d)] = registry
        return registry


def _reset_for_tests() -> None:
    """Forget the loaded IFEvalG modules (tests that load a different directory call this)."""
    with _LOCK:
        _LOADED.clear()
        for name in [n for n in sys.modules if n == "open_instruct" or n.startswith("open_instruct.")]:
            sys.modules.pop(name, None)


def remove_thinking_section(prediction: str) -> str:
    """Identical to ``open_instruct.ground_truth_utils.remove_thinking_section`` at the pinned revision."""
    prediction = prediction.replace("<|assistant|>", "").strip()
    prediction = prediction.split("</think>")[-1]
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


def parse_ground_truth(ground_truth: Any) -> tuple[list[str], list[Any]]:
    """Return ``(instruction_ids, kwargs_list)``. Same parsing as the upstream IFEvalVerifier (literal string, first element)."""
    try:
        if isinstance(ground_truth, str):
            try:
                obj = ast.literal_eval(ground_truth)
            except (ValueError, SyntaxError):
                obj = json.loads(ground_truth)
        else:
            obj = ground_truth
        if isinstance(obj, (list, tuple)):
            obj = obj[0]
        if isinstance(obj, str):
            obj = json.loads(obj)
        ids = list(obj["instruction_id"])
        kwargs = list(obj["kwargs"])
    except Exception as exc:
        raise RewardsError(f"unparsable ground_truth: {type(exc).__name__}") from None
    if len(ids) != len(kwargs):
        raise RewardsError("ground_truth: instruction_id and kwargs have different lengths")
    return [str(i) for i in ids], kwargs


def constraint_ids(ground_truth: Any) -> list[str]:
    return parse_ground_truth(ground_truth)[0]


class Checker:
    """Runs the pinned IFEvalG checkers. One instance per directory; thread-safe except that ``errors`` is a plain counter."""

    def __init__(self, directory: Optional[str | os.PathLike[str]] = None, *, expected: Optional[Mapping[str, str]] = None):
        self.registry = load_ifevalg(directory, expected=expected)
        self.errors: dict[str, int] = {}
        self._lock = threading.Lock()

    def _count(self, key: str) -> None:
        with self._lock:
            self.errors[key] = self.errors.get(key, 0) + 1

    def check(self, response: str, ground_truth: Any) -> tuple[list[str], list[bool], int]:
        """``(instruction_ids, flags, n_errors)``. An empty response (or an empty answer after the thinking section is removed) fails
        every constraint, as upstream scores it 0. A checker exception fails that constraint and is counted."""
        ids, kwargs_list = parse_ground_truth(ground_truth)
        answer = remove_thinking_section(response)
        if len(response) == 0 or len(answer) == 0:
            return ids, [False] * len(ids), 0
        flags: list[bool] = []
        n_err = 0
        table = self.registry.INSTRUCTION_DICT
        for key, args in zip(ids, kwargs_list):
            try:
                args = {k: v for k, v in (args or {}).items() if v is not None}
                inst = table[key](key)
                inst.build_description(**args)
                flags.append(bool(response.strip() and inst.check_following(answer)))
            except Exception:
                self._count(key)
                n_err += 1
                flags.append(False)
        return ids, flags, n_err


_DEFAULT: Optional[Checker] = None


def _default() -> Checker:
    global _DEFAULT
    with _LOCK:
        if _DEFAULT is None:
            _DEFAULT = Checker(None)
        return _DEFAULT


def error_counts() -> dict[str, int]:
    """Constraint id -> number of checker exceptions swallowed by the module-level functions. A non-empty result after a run means
    some rewards are 0 for a reason other than the response."""
    return dict(_DEFAULT.errors) if _DEFAULT is not None else {}


def preflight(checker: Optional[Checker] = None) -> None:
    """Raise RewardsError if NLTK data is missing, or if a sentence-count canary does not behave."""
    try:
        import nltk  # type: ignore
    except ImportError as exc:
        raise RewardsError(f"nltk is not installed: {exc}") from None
    for res in NLTK_RESOURCES:
        try:
            nltk.data.find(res)
        except LookupError:
            raise RewardsError(
                f"NLTK resource {res} is missing; run: python -m nltk.downloader -d $NLTK_DATA punkt punkt_tab stopwords averaged_perceptron_tagger_eng"
            ) from None
    chk = checker or _default()
    ids, flags, n_err = chk.check("One. Two. Three.", _PREFLIGHT_GT)
    if n_err or flags != [True]:
        raise RewardsError(f"IFEvalG sentence-count canary failed (flags {flags}, errors {n_err}); the NLTK data or the checkers are broken")


def verify_constraints(response: str, ground_truth: Any) -> list[bool]:
    return _default().check(response, ground_truth)[1]


def reward(response: str, ground_truth: Any, mode: str = "fraction") -> float:
    """mode="fraction": mean of per-constraint passes (the upstream IFEvalVerifier score).
    mode="all":      1.0 only if every constraint passes (the prompt-level metric).
    mode="hybrid":   0.5 * fraction + 0.5 * all (keeps a gradient but rewards full compliance)."""
    if mode not in ("fraction", "all", "hybrid"):
        raise ValueError(f"unknown mode {mode!r}")
    flags = verify_constraints(response, ground_truth)
    frac = sum(flags) / max(len(flags), 1)
    if mode == "fraction":
        return frac
    allp = 1.0 if flags and all(flags) else 0.0
    if mode == "all":
        return allp
    return 0.5 * frac + 0.5 * allp


def reward_batch(responses: Sequence[str], ground_truths: Sequence[Any], mode: str = "fraction") -> list[float]:
    if len(responses) != len(ground_truths):
        raise ValueError("responses and ground_truths have different lengths")
    return [reward(r, g, mode) for r, g in zip(responses, ground_truths)]


def degenerate_floor(ground_truths: Sequence[Any], answers: Sequence[str] = ("Yes.",), mode: str = "fraction") -> dict[str, float]:
    """Mean reward of constant answers over ``ground_truths``. On the reader's dev split ``"Yes."`` earns about 0.30 (fraction) and
    passes every constraint on about 11 % of prompts, so a trained model must clear that floor before a gain means anything."""
    if not ground_truths:
        raise ValueError("no ground truths")
    return {a: sum(reward(a, g, mode) for g in ground_truths) / len(ground_truths) for a in answers}


if __name__ == "__main__":  # tiny self check: python rewards.py (needs IFEVALG_PATH and NLTK data)
    _gt = "[{'instruction_id': ['detectable_format:sentence_hyphens', 'last_word:last_word_answer'], 'kwargs': [None, {'last_word': 'brief'}]}]"
    print(verify_constraints("Hello-world-this-is-brief", _gt), reward("Hello-world-this-is-brief", _gt))
