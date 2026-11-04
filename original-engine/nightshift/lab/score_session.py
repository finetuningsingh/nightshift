"""Serve, score, tear down, record cost: one scoring session for one ``ScoreRequest``.

``score_adapter`` asks a ``ModelServer`` for an endpoint serving the base model and the request's adapter (if any), scores through the injected
``Scorer`` with a generator bound to that endpoint, and ALWAYS closes the session (the rental is deleted on success, on a scorer error, on a
budget error). The caller appends ``lab.score_recorded`` and ``lab.score_cost``. One server serves the stock model and an adapter by LoRA name
(``ServeEndpoint.model_for``), so arm 1 and arms 2 and 3 can share one process (``score_many``).

If scoring fails or is interrupted, the exception that propagates (a ``KeyboardInterrupt`` too) carries the billed ``GpuUsage`` as ``exc.gpu_usage`` (set when the session could be closed),
so the caller can still append ``lab.score_cost``.
"""

from __future__ import annotations

from typing import Optional, Sequence

from nightshift.lab.contracts import (
    BudgetMeter,
    GeneratorFactory,
    GpuUsage,
    ModelServer,
    ScoreRecord,
    ScoreRequest,
    Scorer,
    ScorerError,
    ServeSpec,
    Split,
)

# Prompt-side headroom added to max_tokens for ``ServeSpec.max_model_len``. The recipe in docs/harness-design.md serves Glimmer with
# --max-model-len 8300 and max_tokens near 8000; IFBench prompts are well under 2048 tokens (UNVERIFIED on the real prompts: a prompt
# that does not fit is rejected by the server and surfaces as a generation error, so the score is refused, not wrong).
PROMPT_TOKEN_ALLOWANCE = 2048
DEFAULT_MAX_GPU_MINUTES = 120.0  # lease cap for one scoring session; the model server also applies the meter's remaining room (INFERRED, unmeasured)


def serve_spec_for(requests: Sequence[ScoreRequest], *, max_model_len: Optional[int] = None, max_gpu_minutes: float = DEFAULT_MAX_GPU_MINUTES,
                   purpose: str = "score") -> ServeSpec:
    first = requests[0]
    adapters = []
    seen = set()
    for r in requests:
        if r.base != first.base:
            raise ScorerError("requests scored together must share one base model")
        if r.adapter is not None and r.adapter.sha256 not in seen:
            seen.add(r.adapter.sha256)
            adapters.append(r.adapter)
    need = max(r.params.max_tokens for r in requests) + PROMPT_TOKEN_ALLOWANCE
    trials = {r.trial_id for r in requests}
    return ServeSpec(
        base=first.base, adapters=tuple(adapters), max_model_len=max_model_len or need, max_gpu_minutes=max_gpu_minutes,
        purpose=purpose, trial_id=first.trial_id if len(trials) == 1 else None,
    )


def score_many(requests: Sequence[ScoreRequest], model_server: ModelServer, scorer: Scorer, generator_factory: GeneratorFactory, *,
               meter: Optional[BudgetMeter] = None, max_model_len: Optional[int] = None,
               max_gpu_minutes: float = DEFAULT_MAX_GPU_MINUTES) -> tuple:
    """Score several requests (same base model) on ONE served endpoint. Returns ``(records, GpuUsage)``. The first failure stops the session
    and propagates; records already written stay on disk (the caller sees them through ``scores/<id>/score.json``)."""
    if not requests:
        raise ScorerError("no score requests")
    logical = set()
    for request in requests:  # the guard reads a ledger that does not yet hold this batch: two requests for one logical score would both pass it
        if request.split is not Split.DEV:
            key = (request.phase, request.split, request.assignment_id, request.adapter.sha256 if request.adapter is not None else None)
            if key in logical:
                raise ScorerError(f"requests {request.request_id!r} and an earlier one score the same adapter on {request.assignment_id!r} ({request.split.value}); a score is written once and never re-rolled")
            logical.add(key)
    precheck = getattr(scorer, "precheck", None)
    if precheck is not None:  # refuse before a GPU is rented, not after
        for request in requests:
            precheck(request)
    spec = serve_spec_for(requests, max_model_len=max_model_len, max_gpu_minutes=max_gpu_minutes,
                          purpose="score:" + ",".join(r.request_id for r in requests)[:150])
    session = model_server.serve(spec, meter)
    records: list = []
    try:
        endpoint = session.endpoint
        generator = generator_factory(endpoint)
        for request in requests:
            records.append(scorer.score(request, generator, endpoint.model_for(request.adapter)))
    except BaseException as exc:
        usage = _close(session, exc)
        if usage is not None:  # any exception, an interrupt included: the rental was billed whatever stopped the scoring
            try:
                exc.gpu_usage = usage  # type: ignore[attr-defined]
            except AttributeError:
                pass  # an exception type that takes no attributes
        raise
    return records, session.close()


def _close(session, original: BaseException):
    try:
        return session.close()
    except Exception as close_exc:
        # the original failure is the one that matters; do not hide that the rental may still exist
        original.__notes__ = list(getattr(original, "__notes__", [])) + [f"closing the serve session also failed: {type(close_exc).__name__}: {close_exc}"]
        return None


def score_adapter(request: ScoreRequest, model_server: ModelServer, scorer: Scorer, generator_factory: GeneratorFactory, *,
                  meter: Optional[BudgetMeter] = None, max_model_len: Optional[int] = None,
                  max_gpu_minutes: float = DEFAULT_MAX_GPU_MINUTES) -> tuple:
    """Score one request on its own served endpoint. Returns ``(ScoreRecord, GpuUsage)``."""
    records, usage = score_many([request], model_server, scorer, generator_factory, meter=meter, max_model_len=max_model_len,
                                max_gpu_minutes=max_gpu_minutes)
    return records[0], usage
