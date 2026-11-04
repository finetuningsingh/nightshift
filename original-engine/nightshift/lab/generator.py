"""``Generator`` over an OpenAI-compatible endpoint (a vLLM server behind a control-host loopback port).

One code path for all three arms: the arm only changes the model name sent in the request (``ServeEndpoint.model_for``).

Request body (``POST <base_url>/v1/chat/completions``): an explicit system message (the Glimmer chat template otherwise injects the
current date), the prompt as the user message, ``temperature``, ``top_p``, ``top_k``, ``max_tokens``, ``seed`` and
``chat_template_kwargs: {"reasoning_strength": ...}``. Only ``message.content`` is read (the final answer; vLLM's reasoning parser keeps
reasoning separate). ``finish_reason`` and ``usage.completion_tokens`` are recorded, and the output keeps the input order.

UNVERIFIED against vLLM 0.32.0 serving Glimmer: ``top_k`` as a top-level extra body field, and ``chat_template_kwargs``. A server that
ignores them still answers; the scorer would then silently score with default decoding, so the first real run must check the logged
decoding against the request.

Failure model: a per-prompt failure (timeout, 5xx, a malformed reply, a prompt the server rejects with a 4xx) is retried with back-off and
then reported as ``GenResult.error`` with empty text. A systemic failure raises ``ScorerError``: a dead endpoint (every recent request
failed at the transport level), or a 401, 403 or 404 (wrong credentials or wrong model or path).
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional, Sequence

from nightshift.lab.contracts import GenerationParams, GenPrompt, GenResult, ScorerError

# This many consecutive prompts failing at the transport level (after their retries) means the endpoint is gone.
DEAD_AFTER = 16
SYSTEMIC_STATUS = (401, 403, 404)


class _Systemic(Exception):
    pass


def chat_completions_url(base_url: str) -> str:
    """``http://host:port`` and ``http://host:port/v1`` both give ``.../v1/chat/completions``."""
    base = base_url.rstrip("/")
    return base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"


def _clean_text(text: str) -> str:
    # a NUL or a lone surrogate would make the contract refuse the text; neither is part of an answer
    return text.replace("\x00", "").encode("utf-8", errors="replace").decode("utf-8")


class HttpGenerator:
    """Implements ``Generator``. Thread-safe: ``generate`` may be called from several threads."""

    def __init__(self, base_url: str, api_key: Optional[str] = None, workers: int = 8, timeout: float = 600.0, retries: int = 2,
                 *, sleep: Callable[[float], None] = time.sleep, backoff: float = 1.0):
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ScorerError(f"base_url must be an http(s) URL, got {base_url!r}")
        if workers < 1 or retries < 0 or timeout <= 0:
            raise ScorerError("workers must be >= 1, retries >= 0 and timeout > 0")
        self._url = chat_completions_url(base_url)
        self._key = api_key
        self._workers = workers
        self._timeout = timeout
        self._retries = retries
        self._sleep = sleep
        self._backoff = backoff

    def __repr__(self) -> str:  # never show the key
        return f"HttpGenerator({self._url!r}, workers={self._workers})"

    # ------------------------------------------------------------------ request
    def _payload(self, model: str, prompt: GenPrompt, params: GenerationParams, seed: int) -> dict:
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": params.system_prompt},
                {"role": "user", "content": prompt.text},
            ],
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "max_tokens": params.max_tokens,
            "seed": seed,
            "stream": False,
            "chat_template_kwargs": {"reasoning_strength": params.reasoning_strength},
        }

    def _post(self, payload: dict) -> Any:
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        req = urllib.request.Request(self._url, data=json.dumps(payload).encode("utf-8"), method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read())

    @staticmethod
    def _parse(prompt_id: str, body: Any) -> GenResult:
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]  # an explicit null is an empty answer (reasoning only); a missing key is a malformed reply
            finish = choice.get("finish_reason") or ""
            usage = body.get("usage") or {}
            tokens = usage.get("completion_tokens", 0)
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
                tokens = 0
            if content is not None and not isinstance(content, str):
                raise TypeError("content is not text")
            return GenResult(prompt_id, _clean_text(content or ""), str(finish), tokens, None)
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ValueError(f"malformed completion: {type(exc).__name__}") from None

    def _one(self, model: str, prompt: GenPrompt, params: GenerationParams, seed: int, state: dict) -> GenResult:
        payload = self._payload(model, prompt, params, seed)
        last = "unknown failure"
        transport = False
        for attempt in range(self._retries + 1):
            if state["dead"]:
                raise _Systemic("endpoint unreachable")
            try:
                result = self._parse(prompt.prompt_id, self._post(payload))
                with state["lock"]:
                    state["consecutive"] = 0
                return result
            except urllib.error.HTTPError as exc:
                transport = False
                last = f"HTTP {exc.code}"
                if exc.code in SYSTEMIC_STATUS:
                    raise _Systemic(f"endpoint answered HTTP {exc.code} (wrong credentials, model or path)") from None
                if exc.code < 500 and exc.code != 429:
                    break  # this prompt is rejected; retrying cannot help
            except (urllib.error.URLError, OSError, TimeoutError, ConnectionError) as exc:
                transport = True
                last = f"{type(exc).__name__}"
            except (ValueError, json.JSONDecodeError) as exc:
                transport = False
                last = str(exc) or type(exc).__name__
            if attempt < self._retries:
                self._sleep(min(self._backoff * (2 ** attempt), 20.0))
        with state["lock"]:
            state["consecutive"] = state["consecutive"] + 1 if transport else 0
            if state["consecutive"] >= DEAD_AFTER:
                state["dead"] = True
        return GenResult(prompt.prompt_id, "", "", 0, last)

    def drain_evidence(self) -> dict:
        """The score record states which generator produced its answers; this one has no log to point to."""
        return {"generator": "http"}

    # ------------------------------------------------------------------ Generator
    def generate(self, model: str, prompts: Sequence[GenPrompt], params: GenerationParams, seed: int) -> list:
        if not prompts:
            return []
        state = {"lock": threading.Lock(), "consecutive": 0, "dead": False}
        results: list = [None] * len(prompts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self._workers, len(prompts))) as pool:
            futures = {pool.submit(self._one, model, p, params, seed, state): i for i, p in enumerate(prompts)}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    results[futures[fut]] = fut.result()
                except _Systemic as exc:
                    state["dead"] = True  # the other workers stop at their next attempt
                    for other in futures:
                        other.cancel()
                    raise ScorerError(str(exc)) from None
        if state["dead"]:
            raise ScorerError(f"endpoint {self._url} is unreachable: {DEAD_AFTER} requests in a row failed at the transport level")
        return results
