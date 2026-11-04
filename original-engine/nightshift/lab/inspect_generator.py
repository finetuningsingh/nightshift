"""``Generator`` that runs the model calls of a score as an Inspect AI task (UK AISI ``inspect_ai``).

``ScoreService`` keeps everything that is the lab's own business: the access table and guard, the frozen data revisions, the no-re-roll rule,
the pinned verifiers, the per-prompt files and the ``ScoreRecord``. This class replaces only the hand-written HTTP loop with an Inspect
evaluation: a versioned task over the prompts the service loaded, the same request body as ``HttpGenerator`` (one explicit system message,
temperature, top_p, top_k, max_tokens, seed, ``reasoning_strength`` chat-template field), Inspect's concurrency, retry and logging, and one
``.eval`` log per seed. The log is hashed and the hash goes into the score record's ``verifier_pins`` (via ``drain_evidence``), together with the
task version, the Inspect version and the code commit, so a ledger score can be traced to the exact log that produced its generations.

Logs of guarded splits contain private prompts and answers. They are written under ``log_dir``; the CLI points it into the lab's private
directory, which is never copied to a researcher host.

Failure model, as ``HttpGenerator``: a sample that fails after its retries becomes ``GenResult.error`` with empty text (the service then refuses
to score from missing answers); an evaluation that stops with an error, or whose sample count is wrong, raises ``ScorerError``. The Inspect
``fail_on_error`` count stops a dead endpoint after ``DEAD_AFTER`` failed samples, the counterpart of ``HttpGenerator``'s transport rule.

UNVERIFIED against a live vLLM serving Glimmer: the Inspect ``openai-api`` provider has been rehearsed only against a local fake endpoint and
Inspect's ``mockllm``; the request body is checked field by field against ``HttpGenerator``'s in the tests. How the provider reports a
reasoning answer that vLLM splits into ``reasoning_content`` must be read from the first live log (only the final answer is taken here).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

from nightshift.lab.contracts import GenerationParams, GenPrompt, GenResult, ScorerError
from nightshift.lab.generator import DEAD_AFTER

TASK_NAME = "glimmer_generate"
TASK_VERSION = "1"  # bump when the system message, request fields or the mapping from Inspect output to GenResult change
REPO = Path(__file__).resolve().parents[2]


def _clean_text(text: str) -> str:
    return text.replace("\x00", "").encode("utf-8", errors="replace").decode("utf-8")


def _git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def prompts_digest(prompts: Sequence[GenPrompt]) -> str:
    """sha256 of the exact prompt ids and texts, in order: the dataset identity of one generation call."""
    h = hashlib.sha256()
    for p in prompts:
        h.update(json.dumps([p.prompt_id, p.text], ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def results_from_samples(samples: Sequence[Any], prompts: Sequence[GenPrompt]) -> list:
    """Map the samples of an Inspect log to ``GenResult`` in prompt order. ``samples`` may arrive in any order."""
    by_id = {}
    for s in samples:
        by_id[str(s.id)] = s
    if len(by_id) != len(samples) or set(by_id) != {p.prompt_id for p in prompts}:
        raise ScorerError(f"the Inspect log holds {len(samples)} samples that do not match the {len(prompts)} prompts sent")
    out = []
    for p in prompts:
        s = by_id[p.prompt_id]
        err = getattr(s, "error", None)
        if err is not None:
            out.append(GenResult(p.prompt_id, "", "", 0, (str(getattr(err, "message", err)) or "sample failed")[:900]))
            continue
        o = s.output
        stop = getattr(o, "stop_reason", None)
        finish = "length" if stop == "max_tokens" else ("stop" if stop in (None, "stop") else str(stop))
        usage = getattr(o, "usage", None)
        tokens = getattr(usage, "output_tokens", 0) if usage is not None else 0
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            tokens = 0
        out.append(GenResult(p.prompt_id, _clean_text(o.completion or ""), finish, tokens, None))
    return out


class InspectGenerator:
    """Implements ``Generator`` (and ``drain_evidence``). ``base_url`` is the vLLM server without ``/v1``; ``model_factory`` maps the served
    model name to what ``inspect_ai.eval`` takes as ``model`` (tests pass a ``mockllm`` model)."""

    def __init__(self, base_url: str, *, log_dir: Path, workers: int = 128, timeout: float = 1500.0, retries: int = 2,
                 model_factory: Optional[Callable[[str], Union[str, Any]]] = None):
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ScorerError(f"base_url must be an http(s) URL, got {base_url!r}")
        if workers < 1 or retries < 0 or timeout <= 0:
            raise ScorerError("workers must be >= 1, retries >= 0 and timeout > 0")
        self._base_url = base_url.rstrip("/")
        self._log_dir = Path(log_dir)
        self._workers, self._timeout, self._retries = workers, timeout, retries
        self._model_factory = model_factory or (lambda served: f"openai-api/vllm/{served}")
        # the lab's vLLM takes no key; the provider insists on one, so it gets a placeholder as an argument (no environment variable is touched)
        self._model_args = {} if model_factory is not None else {"api_key": "none"}
        self._evidence: dict = {}

    def __repr__(self) -> str:
        return f"InspectGenerator({self._base_url!r}, workers={self._workers})"

    # ------------------------------------------------------------------ evidence
    def drain_evidence(self) -> dict:
        """What the generations since the last call were produced by (strings, for ``ScoreRecord.verifier_pins``). Clears it."""
        out, self._evidence = self._evidence, {}
        return out

    # ------------------------------------------------------------------ task
    def build_task(self, model: str, prompts: Sequence[GenPrompt], params: GenerationParams, seed: int):
        from inspect_ai import Task
        from inspect_ai.dataset import MemoryDataset, Sample
        from inspect_ai.model import ChatMessageSystem, GenerateConfig
        from inspect_ai.solver import generate, solver

        digest = prompts_digest(prompts)

        @solver
        def fixed_system():
            async def solve(state, gen):  # not inspect's system_message(): that one treats {braces} in its text as template fields
                state.messages.insert(0, ChatMessageSystem(content=params.system_prompt))
                return state
            return solve

        dataset = MemoryDataset([Sample(id=p.prompt_id, input=p.text, target="") for p in prompts], name=f"prompts:{digest[:16]}")
        config = GenerateConfig(temperature=params.temperature, top_p=params.top_p, max_tokens=params.max_tokens, seed=seed,
                                timeout=int(self._timeout), max_retries=self._retries,
                                extra_body={"top_k": params.top_k, "chat_template_kwargs": {"reasoning_strength": params.reasoning_strength}})
        meta = {"role": "score-generation", "served_model": model, "seed": seed, "prompts_sha256": digest, "n_prompts": len(prompts),
                "decoding": params.to_dict(), "git": _git_state(), "python": sys.version.split()[0]}
        return Task(dataset=dataset, solver=[fixed_system(), generate()], config=config, name=TASK_NAME, version=TASK_VERSION, metadata=meta), digest

    # ------------------------------------------------------------------ Generator
    def generate(self, model: str, prompts: Sequence[GenPrompt], params: GenerationParams, seed: int) -> list:
        if not prompts:
            return []
        import inspect_ai
        from inspect_ai import eval as inspect_eval

        task, digest = self.build_task(model, prompts, params, seed)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log = inspect_eval(task, model=self._model_factory(model), model_base_url=self._base_url + "/v1", model_args=self._model_args, log_dir=str(self._log_dir),
                           max_connections=self._workers, fail_on_error=float(DEAD_AFTER), display="none",
                           metadata={"prompts_sha256": digest, "seed": seed})[0]
        if log.status != "success":
            message = (log.error.message if log.error is not None else log.status)[:300]
            raise ScorerError(f"the Inspect evaluation of seed {seed} ended with status {log.status}: {message}")
        results = results_from_samples(log.samples or [], prompts)
        path = Path(str(log.location).replace("file://", ""))
        self._evidence.update({
            "generator": "inspect",
            "inspect_ai": inspect_ai.__version__,
            "inspect_task": f"{TASK_NAME}@{TASK_VERSION}",
            f"inspect_log_seed{seed}": path.name,
            f"inspect_log_sha256_seed{seed}": hashlib.sha256(path.read_bytes()).hexdigest(),
            "code_commit": str(task.metadata["git"]["commit"]),
            "code_dirty": str(task.metadata["git"]["dirty"]).lower(),
        })
        return results
