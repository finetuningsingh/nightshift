"""Dev grader: read ``submission.json`` from the committed tree, ask the control API for that adapter's DEV score, return it.

Runs inside CORAL's grader virtual environment on the researcher host, as a subprocess of ``coral start``. Everything this grader
returns is written to the public attempt JSON and shown to the researcher, so it only ever returns the dev aggregate the control
API gives for an adapter this trial's own jobs produced. It never sees selection or test data: the control API is the only thing
that knows the numbers, and it answers dev scores only.

Connection details come from the environment of ``coral start`` (set by the lab runtime): ``GLIMMER_CONTROL`` (the control URL,
``http://127.0.0.1:<port>/control``) and ``GLIMMER_TOKEN_FILE`` (the per-trial token file). Standard library only, apart from the
CORAL base class, which is imported lazily so this module can be tested without CORAL.
"""

from __future__ import annotations

import json
import os
import re
import stat
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

MAX_SUBMISSION_BYTES = 65536
_IDENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_HTTP_TIMEOUT_S = 60.0

try:  # pragma: no cover - exercised only where CORAL is installed
    from coral.grader import TaskGrader as _Base
except ImportError:  # the tests inject a stand-in; outside CORAL there is no grader to run
    _Base = object  # type: ignore[assignment,misc]


def read_submission(codebase_path: str) -> dict:
    """The committed ``submission.json`` of the graded checkout: a regular file (never a link), small, an object with ``adapter_id``."""
    path = os.path.join(codebase_path, "submission.json")
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise ValueError("submission.json not found in the committed tree; run `glimmer pointer <adapter_id>` and commit the file") from None
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("submission.json must be a regular file")
    if st.st_size > MAX_SUBMISSION_BYTES:
        raise ValueError("submission.json is too large")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as fh:
        raw = fh.read(MAX_SUBMISSION_BYTES + 1)
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("submission.json is not valid JSON") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("adapter_id"), str) or not _IDENT.match(doc["adapter_id"]):
        raise ValueError("submission.json needs an adapter_id string")
    return doc


def fetch_dev_score(control_url: str, token: str, adapter_id: str, *, timeout: float = _HTTP_TIMEOUT_S) -> dict:
    """GET ``<control>/dev-score?adapter_id=<id>``. Raises ValueError with a message that is safe to show the researcher."""
    url = control_url.rstrip("/") + "/dev-score?" + urllib.parse.urlencode({"adapter_id": adapter_id})
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - loopback control API
            body = resp.read(1 << 20)
    except urllib.error.HTTPError as exc:
        try:
            msg = json.loads(exc.read(1 << 16).decode("utf-8")).get("error", "")
        except Exception:  # noqa: BLE001
            msg = ""
        raise ValueError(f"control API refused the dev score ({exc.code}): {str(msg)[:300]}") from None
    except (urllib.error.URLError, OSError, TimeoutError):
        raise ValueError("control API unreachable") from None
    try:
        doc = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("control API returned invalid JSON") from None
    score = doc.get("dev_score") if isinstance(doc, dict) else None
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 1.0):
        raise ValueError("control API returned no usable dev score")
    return doc


def evaluate_submission(codebase_path: str, control_url: str, token: str) -> tuple[Optional[float], str]:
    """The grader's logic without CORAL: ``(score, explanation)``; score None means the evaluation failed (explanation says why)."""
    try:
        sub = read_submission(codebase_path)
        doc = fetch_dev_score(control_url, token, sub["adapter_id"])
    except ValueError as exc:
        return None, str(exc)
    score = float(doc["dev_score"])
    explanation = (
        f"dev {doc.get('metric', 'prompt_all')} = {score:.4f} over {doc.get('n_prompts', '?')} dev prompts "
        f"(adapter {sub['adapter_id']}); researcher-influenced dev number, not a selection or test score"
    )
    return score, explanation


def _read_token(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read(4096).strip()


class Grader(_Base):  # type: ignore[misc,valid-type]
    """CORAL ``TaskGrader`` whose score is the control host's dev score for the adapter in the committed ``submission.json``."""

    def evaluate(self):  # type: ignore[no-untyped-def]
        control = os.environ.get("GLIMMER_CONTROL", "")
        token_file = os.environ.get("GLIMMER_TOKEN_FILE", "")
        if not control or not token_file:
            return self.fail("grader is not configured (GLIMMER_CONTROL / GLIMMER_TOKEN_FILE missing)")
        try:
            token = _read_token(token_file)
        except OSError:
            return self.fail("grader token file unreadable")
        score, explanation = evaluate_submission(self.codebase_path, control, token)
        if score is None:
            return self.fail(explanation)
        return self.score(score, explanation)
