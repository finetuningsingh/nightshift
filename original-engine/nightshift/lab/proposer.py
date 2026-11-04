"""``ProposalSource`` over Muse Wonder (owner 4).

One strict-JSON Chat Completions call per candidate, through the reused ``MeteredClient`` (so the Wonder spend is booked by the outer
ledger's ``BudgetGate``). The client must be built with ``fallback_model=None``: a silent fallback to Spark would label a Spark proposal
as Wonder, so ``ProposalResult.served_model`` and ``fallback_used`` carry what the provider says answered and the controller records both.

* ``chat_json`` (reused) already retries ONCE when the reply is empty, not a JSON object, or cut off with a required key missing.
* A reply that is a JSON object but not a valid ``Proposal`` is NOT asked again: the candidate is INVALID and costs only Wonder tokens.
* ``UnparseableProposal`` (a ``ProposalInvalid``) carries the raw reply and the money spent, so the controller can still record it.
* ``parse_json_object`` is copied from ``nightshift.agents.llm`` (importing the agent package would couple the lab to it); a test keeps
  the two equal on a fixture set.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from nightshift.agents.prompts import chat_json
from nightshift.lab.contracts import (
    BudgetExhausted,
    ContractError,
    EditOp,
    LabError,
    Lever,
    PackEdit,
    Proposal,
    ProposalContext,
    ProposalInvalid,
    ProposalResult,
    SearchReplace,
)
from nightshift.models.pricing import usd_for
from nightshift.types import BudgetExceeded, ChatMessage, ModelError, Usage

MAX_EDITS = 12
MAX_COMPLETION_TOKENS = 16000  # Wonder's real ceiling is undocumented; 16000 is the conservative figure the client already uses


class ProposerUnavailable(LabError):
    """The provider failed (network, auth, 5xx after the client's own retries). No candidate exists, nothing is recorded."""


class UnparseableProposal(ProposalInvalid):
    """The reply could not be turned into a ``Proposal``. ``raw_response`` is the last reply, ``usd`` the money all calls cost."""

    def __init__(self, reason: str, *, raw_response: str = "", usd: float = 0.0, model: str = "", served_model: str = "",
                 fallback_used: bool = False, prompt_tokens: int = 0, completion_tokens: int = 0):
        super().__init__(reason)
        self.reason, self.raw_response, self.usd = reason, raw_response, usd
        self.model, self.served_model, self.fallback_used = model, served_model, fallback_used
        self.prompt_tokens, self.completion_tokens = prompt_tokens, completion_tokens


PROPOSAL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "lever": {"type": "string", "enum": [lv.value for lv in Lever]},
        "change_summary": {"type": "string"},
        "expected_effect": {"type": "string"},
        "risk": {"type": "string"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "op": {"type": "string", "enum": [op.value for op in EditOp]},
                    "content": {"type": ["string", "null"]},
                    "blocks": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "object",
                            "properties": {"search": {"type": "string"}, "replace": {"type": "string"}},
                            "required": ["search", "replace"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["path", "op", "content", "blocks"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["hypothesis", "lever", "change_summary", "expected_effect", "risk", "edits"],
    "additionalProperties": False,
}


def parse_json_object(text: str | None) -> dict | None:
    """The JSON object in a model reply (tolerates a code fence or surrounding prose), or ``None``. (Copy of agents.llm's.)"""
    text = (text or "").strip()
    if not text:
        return None
    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{"):text.rfind("}") + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def proposal_from_dict(obj: Mapping[str, Any]) -> Proposal:
    """A ``Proposal`` from the parsed reply. Lengths are checked here (strict mode cannot). Raises ``ProposalInvalid``."""
    try:
        extra = set(obj) - set(PROPOSAL_SCHEMA["required"])
        missing = set(PROPOSAL_SCHEMA["required"]) - set(obj)
        if extra or missing:
            raise ProposalInvalid(f"reply keys differ from the schema (missing {sorted(missing)}, unknown {sorted(extra)})")
        raw_edits = obj["edits"]
        if not isinstance(raw_edits, list) or not raw_edits:
            raise ProposalInvalid("edits must be a non-empty list")
        if len(raw_edits) > MAX_EDITS:
            raise ProposalInvalid(f"{len(raw_edits)} edits (limit {MAX_EDITS}); propose one atomic change")
        edits = []
        for i, e in enumerate(raw_edits):
            if not isinstance(e, dict) or set(e) != {"path", "op", "content", "blocks"}:
                raise ProposalInvalid(f"edit {i + 1}: expected path, op, content and blocks")
            blocks = e["blocks"]
            if e["op"] == EditOp.REPLACE_FILE.value:
                if blocks:
                    raise ProposalInvalid(f"edit {i + 1}: replace_file takes content, not blocks")
                edits.append(PackEdit(e["path"], EditOp.REPLACE_FILE, e["content"], ()))
            elif e["op"] == EditOp.SEARCH_REPLACE.value:
                if e["content"] is not None or not isinstance(blocks, list):
                    raise ProposalInvalid(f"edit {i + 1}: search_replace takes blocks, not content")
                edits.append(
                    PackEdit(e["path"], EditOp.SEARCH_REPLACE, None, tuple(SearchReplace(b["search"], b["replace"]) for b in blocks))
                )
            else:
                raise ProposalInvalid(f"edit {i + 1}: unknown op {e['op']!r}")
        return Proposal(obj["hypothesis"], obj["lever"], obj["change_summary"], obj["expected_effect"], obj["risk"], tuple(edits))
    except ProposalInvalid:
        raise
    except (ContractError, KeyError, TypeError) as exc:
        raise ProposalInvalid(f"the reply is not a valid proposal: {str(exc)[:300]}") from None


class _Recorder:
    """Forwards ``chat`` to ``client`` and keeps every response, so the total cost of a retry is known."""

    def __init__(self, client: Any):
        self._client = client
        self._inner = client
        self.responses: list[Any] = []
        self.supports_charge_context = bool(getattr(client, "supports_charge_context", False))

    @property
    def model(self) -> str:
        return self._client.model

    def chat(self, *args: Any, **kw: Any) -> Any:
        resp = self._client.chat(*args, **kw)
        self.responses.append(resp)
        return resp


class WonderProposalSource:
    """Implements ``ProposalSource``. ``client`` is a ``MeteredClient`` over a ``MetaChatClient(model="muse-wonder-1.0", fallback_model=None)``."""

    def __init__(self, client: Any, *, schema: Optional[Mapping[str, Any]] = None, reasoning_effort: str = "high",
                 max_completion_tokens: int = MAX_COMPLETION_TOKENS):
        self._client = client
        self._schema = dict(schema if schema is not None else PROPOSAL_SCHEMA)
        self._effort = reasoning_effort
        self._max_tokens = max_completion_tokens

    def propose(self, context: ProposalContext) -> ProposalResult:
        messages = [ChatMessage(role="system", content=context.system_prompt), ChatMessage(role="user", content=context.user_prompt)]
        rec = _Recorder(self._client)
        try:
            resp, obj = chat_json(
                rec, messages, schema=self._schema, schema_name="pack_proposal", parse=parse_json_object,
                reasoning_effort=self._effort, max_completion_tokens=self._max_tokens, cache_key=None,
                charge_context={"exp_id": context.candidate_id, "purpose": "propose"},
            )
        except BudgetExceeded as exc:
            if rec.responses:
                # the budget wall was met by the retry of a reply that was already paid for: carry that spend and reply to the controller (an
                # INVALID candidate, charged to the lab cap) instead of dropping it; the next proposal meets the same wall and raises.
                raise UnparseableProposal(f"the reply was unreadable and the retry hit the budget limit ({exc.kind})", **self._facts(rec, rec.responses[-1])) from None
            raise BudgetExhausted(exc.kind, float(exc.limit or 0.0), float(exc.used or 0.0)) from None
        except ModelError as exc:
            if rec.responses:
                # an earlier reply was paid for (it was unreadable, so chat_json retried) and the retry then failed: the money is spent, so the
                # controller must see it. It records an INVALID candidate that carries the cost, as for any unreadable reply.
                raise UnparseableProposal(f"the reply was unreadable and the retry failed: {str(exc)[:300]}", **self._facts(rec, rec.responses[-1])) from None
            raise ProposerUnavailable(f"the proposer call failed: {str(exc)[:300]}") from None
        facts = self._facts(rec, resp)
        if obj is None:
            raise UnparseableProposal("the reply is not a JSON object" + (" (cut off at the token limit)" if resp.finish_reason == "length" else ""), **facts)
        try:
            proposal = proposal_from_dict(obj)
        except ProposalInvalid as exc:
            raise UnparseableProposal(str(exc), **facts) from None
        return ProposalResult(proposal, rec.model, facts["served_model"], facts["fallback_used"], facts["usd"], facts["prompt_tokens"], facts["completion_tokens"], resp.text)

    @staticmethod
    def _facts(rec: "_Recorder", last: Any) -> dict:
        """What every call so far cost, and the model and text of ``last``."""
        usd = sum(usd_for(r.model or rec.model, r.usage if isinstance(r.usage, Usage) else Usage()) for r in rec.responses)
        return dict(
            raw_response=last.text, usd=usd, model=rec.model, served_model=last.model or rec.model,
            fallback_used=any(bool((r.raw.get("_nightshift") or {}).get("fallback_used")) for r in rec.responses),
            prompt_tokens=sum(r.usage.prompt_tokens for r in rec.responses), completion_tokens=sum(r.usage.completion_tokens for r in rec.responses),
        )
