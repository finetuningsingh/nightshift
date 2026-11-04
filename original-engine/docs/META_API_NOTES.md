# Meta Model API: verified notes

First collected on 2026-10-26 from the official docs at `https://dev.meta.ai/docs/*` and the Cerebral Valley event page. The
"Responses API" sections were added by the Responses-client workstream, which read `/docs/protocols/responses`, `/docs/api-reference/responses`
(overview, create, schemas), `/docs/reasoning` and `/docs/tool-calling`. The API-conformance audit before it re-fetched the pages cited in the sections below (models, protocols, chat-completions, schemas,
reasoning, quickstart, token-counting, prompt-caching, pricing-rate-limits, error-handling, the two cookbook pages, Muse Code and
its changelog, the event page); overview, tool-calling and structured-output were not re-fetched and keep their first-collection
content. Anything marked **UNVERIFIED** was not retrievable and must not be relied on. Where this file says "the docs state", the page is named; where a point comes from
somewhere else (a press report, a third-party model catalogue) it says so.

## Endpoint and auth (verified)

- Base URL: `https://api.meta.ai/v1`
- Auth: `Authorization: Bearer $MODEL_API_KEY`. (Muse Code's headless mode reads `META_API_KEY` instead.)
- Protocols: Responses (`POST /v1/responses`), Chat Completions (`POST /v1/chat/completions`), Messages. All three
  cost the same per token (`/docs/protocols`). The docs say the API is drop-in compatible with the OpenAI SDK, so partner hosts of
  Meta open-weights models (Fireworks, Baseten, Modal) can be reached with the same client and a different base URL.
- Nightshift uses **Chat Completions** by default. An **opt-in Responses client** (`nightshift/models/responses.py`, `MetaResponsesClient`) exists
  for roles whose routing says `api = "responses"`; it is **untested against the real service** (see "The Responses client" below for
  what it does and every assumption it rests on).

## Models (verified, `/docs/models`)

| ID | Notes |
|---|---|
| `muse-spark-1.3` | Latest Muse Spark. 1,048,576-token context. Tool calling, structured output, reasoning, every effort level including `max`. Responses, Chat Completions and Messages. Default here. |
| `muse-spark-1.3-contributor` | Cheaper, but permits Meta to train on your content. **No `max` effort.** Never a default. |
| `muse-spark-1.2`, `muse-spark-1.1` | Older. 1.2 has a `-contributor` variant; 1.1 has none (Standard mode only). 1.1 is not available in Muse Code. |
| `muse-wonder-1.0` | Preview. Reasoning `minimal` to `max`. Takes the same endpoints as Spark (Responses, Chat Completions, Messages). **No contributor option** on pay-as-you-go. "Muse Wonder Full" is not available over the Model API, so effort `max` on Wonder does not select it. |
| `muse-image-1.0`, `sam-3.1`, `muse-voice-transcribe-1.0` | Not used by the core. |
| Muse Spark Open, Muse Glimmer | Open weights only. Not callable over Model API. |

The models page states **no maximum output length** for any model. Third-party catalogues list 131,072 output tokens for
Muse Spark; that is not an official figure and nothing here relies on it.

## Chat Completions specifics (verified, `/docs/protocols/chat-completions`, `/docs/api-reference/chat-completions/schemas`, `/docs/tool-calling`)

- Roles: `system` (same as `developer`), `developer`, `user`, `assistant`, `tool` (needs `tool_call_id`).
- `reasoning_effort`: `minimal | low | medium | high | xhigh | max`.
  - `"none"` returns HTTP 400 for Muse Wonder and Muse Spark (`/docs/reasoning`). The client maps `none` to `minimal`.
  - **`max` is supported by `muse-wonder-1.0` and `muse-spark-1.3`, and is NOT available with `muse-spark-1.3-contributor`**
    (`/docs/reasoning`, `/docs/models`). The docs do not say whether older Spark releases take it; the client treats them as
    unsupported. (An earlier revision of these notes said "Wonder only"; that was wrong.)
  - The Wonder-to-Spark fallback keeps `max` when the fallback model is `muse-spark-1.3` and downgrades it to `xhigh` only when
    the fallback model cannot take it.
- `max_completion_tokens` (deprecated alias `max_tokens`). It caps visible output **and reasoning tokens together**. The page lists
  no maximum value, but states: "Requesting more than the model's configured maximum returns `HTTP 400`", and that input tokens
  plus requested output tokens must fit the context window. The error-handling page does not mention this 400. Consequence for the
  client: never double a cap without a ceiling (`prompts.RETRY_CAP_CEILING`, 16000), allow a `completion_cap_ceiling` per endpoint in
  `nightshift.toml`, and on a 400 that names `max_completion_tokens` retry once at the stated limit, else the largest value seen
  accepted, else half. `nightshift doctor --live` probes 16000/32000/64000 to find the real ceiling.
  A 400 about the context window (code `context_length_exceeded`, or wording about the context window without "configured maximum") is
  not treated as the output maximum: it is not retried at a lower cap and nothing is remembered, so one huge prompt cannot clamp later calls.
- **Rejected with HTTP 400:** `stop`, `logit_bias`, `prediction`, `verbosity`, `n > 1`, `logprobs: true`,
  `web_search_options`, `modalities`, `audio`, `text.format`, `max_output_tokens` (the Responses spelling).
- **`tool_choice`: only `"auto"` is supported.** `"none"`, `"required"` and named choices return 400.
  A client must therefore cope with the model answering in plain text instead of calling a tool.
- Tools use the nested form: `{"type":"function","function":{"name","description","parameters"}}`.
  Names match `^[a-zA-Z0-9_.-]+$`, at most one dot.
- Every assistant message with `tool_calls` must be followed by exactly one `tool` message per call, with the
  matching `tool_call_id`, in order. Omitting the assistant message when replaying history returns HTTP 400. The page does not say
  whether the replayed assistant message's `content` may be `null` or must be `""`; the response schema allows `content` to be a
  string or null. The live smoke test tries both.
- `parallel_tool_calls` defaults to true. Execute every call and answer each (one `tool` message per call).
- Structured output: `response_format={"type":"json_schema","json_schema":{"name":...,"schema":{...},"strict":true}}`.
  Strict subset: root is a plain object, `additionalProperties:false` on every object, `required` lists every
  property, no `allOf`/`oneOf`, no recursion, max nesting depth 10. Which constraint keywords (`minLength`, `maxLength`,
  `pattern`, `minItems`, `maxItems`, ...) strict mode accepts is **UNVERIFIED**: the smoke test sends every real agent schema and the
  supervisor tool schemas and records which keyword a rejection names.
- Prompt caching: `prompt_cache_key` (accepted on Chat Completions and Responses; it replaces the deprecated `user` field; use a stable
  string naming a shared prefix, not a per-request value). `prompt_cache_retention` (`in_memory` | `24h`) is documented as supported by
  the **Responses API only** (`/docs/prompt-caching`); the client sends it only when configured and strips it if a 400 names it. The
  minimum cacheable prefix is **not stated** (so the smoke test caches a prefix of at least 8k tokens as well as a ~4k one).
  `safety_identifier` (<= 64 chars, hash it).
- Response (OpenAI shape):
  - `choices[0].message.content` is a string or null; `refusal` is a string or null (both always present); `reasoning_content` is a
    string or null and is **redacted to empty for external callers** (it is populated only for internal callers); `tool_calls`;
    `annotations`. Content parts may be text or refusal parts.
  - `choices[0].finish_reason`: `stop`, `length` (output limit reached, reasoning included), `tool_calls`, `content_filter`,
    `function_call` (deprecated). Only one choice is returned.
  - `usage.{prompt_tokens,completion_tokens,total_tokens}`; `usage.prompt_tokens_details.cached_tokens` is the Chat Completions
    location of cached tokens (Responses uses `usage.input_tokens_details.cached_tokens`); `usage.completion_tokens_details.reasoning_tokens`
    exists and reasoning tokens are "a subset of the output total" (`/docs/token-counting`), so they are already inside
    `completion_tokens` and billed at the output price. The client still reads cached tokens defensively (details, then a flat
    `cached_tokens`).
- Streaming: `stream: true`, SSE chunks with `choices[0].delta` (`content`, `reasoning_content`, `tool_calls` with `index`, `role`,
  `refusal`). **A stream omits `usage` unless the request sets `stream_options={"include_usage": true}`**
  (`/docs/cookbook/streaming-responses`, `/docs/token-counting`, schemas page); then the usage arrives only on the final chunk, whose
  `choices` is empty, and an interrupted or cancelled stream may omit it. The client sends `include_usage` on every streamed request
  (including the 504-then-stream fallback); without it the budget would only ever see an estimate of the visible text.
  Non-streaming requests can hit HTTP 504 (`gateway_timeout`) on long generations; streaming requests are exempt. The page documents no
  error chunk shape for Chat Completions.

## Errors and rate limits (verified, `/docs/error-handling`, `/docs/pricing-rate-limits`, `/docs/cookbook/error-handling-retry`)

- Error body: `{"error":{"message","type","param","code"}}`. Branch on `type` and `message` (`code` is often null on validation errors).
  Unknown paths answer 404 with an empty body.
- Error codes named by the docs: `invalid_api_key` (401), `model_not_found`, `file_not_found` (404), `payload_too_large` (413),
  `rate_limit_exceeded` (429), `gateway_timeout` (504, non-streaming only), `server_shutting_down`, `service_overloaded`,
  `backend_unavailable` (all 503), `content_policy_violation` (media).
- Retry: the error-handling page lists **429, 500, 503** and says to honour `Retry-After` (503 `service_overloaded` and
  `backend_unavailable` send `Retry-After: 60`). The retry cookbook recipe is wider: retry **429, any 5xx (500-599), and timeout or
  connection errors**, with backoff (0.5 s doubling plus jitter). The client retries 429/500/**502**/503; a network failure (timeout, connection error) is raised at once as a retryable `ModelError` with no internal retry; 504 is
  handled separately because "the same request will usually time out again" (a non-streaming 504 is retried once as a stream).
- Do not retry unchanged: 400, 401, 402, 403, 404, 413. A model API that answers 401, 402 or 403 ends a Nightshift run (`run.finished.reason` `error`, verdict `missing_authority`, exit 1): the key, the credits or the access are missing.
- Timeouts (our choice; the docs state none for the client): `MetaChatClient` allows 300 s per request, and 600 s for a call whose `max_completion_tokens` is above 8000, because Muse Spark always reasons and a non-streamed reply can take minutes. An explicit `timeout` always wins.
- A 404 with code `model_not_found` for the primary model, when a `fallback_model` is configured (the default for `muse-wonder-1.0` roles is `muse-spark-1.3`), makes the fallback sticky for that client; the first response afterwards carries `fallback_reason` in `response.raw["_nightshift"]`, once. A 400 falls back for that one call only.
- Mid-stream errors arrive as a terminal server-sent event, not an HTTP status. The page describes them for the **Responses** stream
  (`response.failed`, and `error` with `code` and `message` at the top level) and says a mid-stream `server_shutting_down` should be
  handled by re-sending the whole request. The Chat Completions chunk shape is not documented, so the client accepts the OpenAI-style
  `{"error": {...}}` chunk as well as both Responses shapes, and re-sends the whole request for `server_shutting_down`,
  `service_overloaded` and `backend_unavailable` (all three are the documented transient 503 codes), within the normal retry budget.
- Limits per team (not per key): `muse-spark-1.3`, `1.2`, `1.1` and `muse-wonder-1.0` have 3,000 RPM / 4,000,000 TPM. **Both contributor
  variants have 150 RPM / 3,000,000 TPM.**
- Successful token-based responses carry `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens`, `x-ratelimit-limit-requests` and
  `x-ratelimit-remaining-requests`. The client keeps them in `response.raw["_nightshift"]["ratelimit"]`; the smoke test records them.

## Pricing, standard mode (verified, `/docs/pricing-rate-limits`, per 1M tokens)

| Model | Cached input | Input | Output |
|---|---|---|---|
| `muse-spark-1.3` / 1.2 / 1.1 | $0.15 | $1.25 | $4.25 |
| `muse-wonder-1.0` | $0.20 | $4.00 | $20.00 |
| `muse-spark-1.3-contributor` / `muse-spark-1.2-contributor` | $0.002 | $0.10 | $0.20 |

The 1.2 contributor price is the same as the 1.3 contributor price (verified; earlier notes called it assumed). Reasoning tokens are
output tokens.

## Hackathon facts (what could and could not be verified, 2026-10-26)

Verified on the Cerebral Valley event page (`https://cerebralvalley.ai/e/meta-ai-developer-hackathon`):

- $1,000,000 in total prizes, split across an indie developer track and a startup track. Indie: "creativity, technical ambition, and
  whether it works". Startup: "technical execution and whether it holds up in production".
- Dates (ET): applications Oct 13 to Oct 25; kickoff livestream Oct 26; workshops and Discord office hours Oct 27 to Nov 4;
  **submissions due Nov 4, 9:00 PM**; judging Nov 4 to Nov 8; winners Nov 9.
- Partner credits for "select participants": Baseten $100 (plus 2M TPM rate limits), Modal $100, Fireworks $25, for Meta's open-weights
  models.

**Not verifiable from public pages, but supplied by the author:** the Model API credit amount. The public event page and the docs
do not state it, and a search turned up only press reports that a new Meta Model API account gets $20 of free credits (Muse Spark 1.1
launch coverage, July 2026). The author supplied the hackathon's own credit terms in an earlier conversation (quoted in the
repository's `to-do/21-supplied-meta-model-api-credit-terms.md` on the `add-openwiki-and-todo` branch): **$150 of Meta Model API
promotional credits per accepted, eligible participant** (no pooling, no duplicate accounts); claimed through a form that asks for the
application email and the Meta Model API team ID, **claim deadline November 4, 2026, 9:00 PM Eastern**; applied in batches with
reasonable efforts within two business days (longer under high volume); no expiry once applied; standard billing resumes when they run
out; a good-standing account and a payment method are needed to claim them; the Model API terms and the acceptable-use policy still
apply; Meta open-weight models may be used without a Meta API account. Treat that as the author's reading of terms this repository
cannot verify, and confirm the balance on the account's billing page. The partner offers the same source lists (for Meta's open-weights
models, "for select participants"): Baseten $100 and 2M TPM, Modal $100, Fireworks $25. Nothing in the budget code depends on any of
this: the budget gate uses the limits the operator sets.

## Chat Completions versus Responses (verified at the cited pages; read before choosing a protocol)

What Meta states:

- `/docs/protocols`: the Responses API is "the recommended default for new work"; Chat Completions is "the simplest drop-in if your code
  already calls `/v1/chat/completions`". Responses is suited to "agentic and coding workloads where reasoning continuity across tool turns
  matters". All formats cost the same per token.
- `/docs/reasoning`: "Chat Completions cannot carry reasoning across turns for external API keys." On Responses you either chain turns with
  `previous_response_id` (the server keeps the reasoning) or replay it statelessly by requesting `include: ["reasoning.encrypted_content"]`
  and sending the returned reasoning items back as input on the next turn. `reasoning_content` on Chat Completions is redacted to empty
  for external callers, so it cannot be used to carry reasoning either. Reasoning summaries are a Responses-only feature.
- `/docs/quickstart`: without the `include` setting the model starts reasoning from scratch on every turn, which "can make it lose the
  thread in multi-step loops"; `include: ["reasoning.encrypted_content"]` is "what carries Muse Spark's reasoning across turns".
- `/docs/protocols/chat-completions`: suits single-turn prompts and simple multi-turn exchanges; for agentic or multi-step work the page
  recommends Responses. `store=false` with encrypted replay is documented under "Reasoning items" on `/docs/protocols/responses`; the
  quickstart, the reasoning page and the protocols overview do not mention `store=false`.

What that means for Nightshift (this is our reading, not Meta's wording):

- Every Nightshift model call except the supervisor's tool loop is a single turn: the designer, implementer, reviewer, reflector and judge
  each get one prompt and return one structured object. For those, the missing reasoning carry-over costs nothing.
- The supervisor and the patch worker run multi-step tool loops. On Chat Completions each turn re-reasons from scratch: that is a quality
  risk (the model can lose the thread across tool turns) and a cost risk (the reasoning is regenerated, and billed at the output price,
  on every turn, where Responses replays it as input that can be cached). How replayed encrypted reasoning is billed is not stated on any
  page read (if it counts as input tokens, a long loop pays for it on every turn, so the saving can be smaller than the output it avoids). The size
  of the difference was not measured: no live key was available.
- We default to Chat Completions because it is the common denominator with partner hosts (Fireworks, Baseten, Modal) and with any
  OpenAI-compatible endpoint, because the budget gate and the ledger fields were built around it, and above all because the Responses client
  below has **never run against the real service**: no Meta API key was available, so it is verified only against the fake server in
  `tests/support/fake_meta_server.py`. Choosing `api = "responses"` for the supervisor and the patch worker is the intended use once
  `nightshift doctor --live` says so (its Responses probes print `responses OK: set api = "responses" for the supervisor in nightshift.toml`
  or `chat only`). Partner hosts stay on Chat Completions.

## Responses API (verified at the cited pages on 2026-10-26)

From `/docs/protocols/responses`, `/docs/api-reference/responses` (overview and `/schemas`), `/docs/reasoning`, `/docs/tool-calling`:

- `POST /v1/responses`. Request fields used here: `model`, `input` (a string or typed items), `instructions` (developer-level, this request only),
  `tools` (**flat**: `{"type":"function","name","description","parameters","strict"}`; `strict` optional), `text.format` (**not**
  `response_format`; `{"type":"json_schema","name","schema","strict"}`, name `a-zA-Z0-9_-` up to 64), `reasoning.effort` (`minimal` to `xhigh`, `max`
  per model; **`none` is a 400**), `include`, `store` (default true), `stream`, `max_output_tokens` (**minimum 16**; shares the context window with
  the input; above the model's configured maximum is a 400; caps reasoning plus visible output), `temperature`, `top_p`, `prompt_cache_key`,
  `prompt_cache_retention` (`in_memory` | `24h`), `safety_identifier`, `previous_response_id`, `background`, `truncation` (only `disabled`).
- Not implemented (400): `stop`, `verbosity`, `logit_bias`, `prediction`, `n` > 1, audio modalities, `web_search_options`, a top-level `logprobs`;
  `include: ["message.output_text.logprobs"]` and any unknown `include` value. `stream_options.include_usage` **has no effect** on Responses.
- Input items: `message` (`role` user/system/developer/assistant; content blocks `input_text`... for user, `output_text` for an assistant replay;
  assistant-only `phase` of `commentary` or `final_answer`), `function_call` (`call_id` 1-64 characters, `name`, `arguments` a JSON string, optional
  `id`), `function_call_output` (`call_id` must match a `function_call` **in the same request** when there is no `previous_response_id`; without
  it keep the results in the same order as the calls), `reasoning` (`summary` required, send `[]`; `encrypted_content`; `id` optional on input).
- **Ordering rule**: a `reasoning` item must be followed by an assistant message or a `function_call` before any user, system or developer message,
  else 400 with `param: "input"`. Intermediate assistant text before a `function_call` **must be replayed with `phase: "commentary"`** (a
  message without `phase` counts as a final answer and is rejected there, and the server does not reclassify it).
- **Stateless reasoning replay** (`/docs/protocols/responses`, "Reasoning items"): `store: false` with `include: ["reasoning.encrypted_content"]`;
  replay the returned `reasoning` item as `{"type":"reasoning","id":"rs_...","summary":[],"encrypted_content":"..."}`. `include` cannot be
  combined with `previous_response_id` (400, `param: "include"`). A missing or expired item is a 400 "Referenced reasoning item ... was not
  found or has expired". `/docs/reasoning` confirms `include` works across built-in tool loops (empty `content`, `encrypted_content` present).
  The `store=false` plus `include` combination is documented on the protocols page; the quickstart, reasoning and tool-calling pages do not mention
  `store=false`.
- Output: `output` is an array of `reasoning` (server `id`, `rs_` prefix, empty content, `encrypted_content` only when requested), `message`
  (`role: "assistant"`, `status`, `content` of `output_text` blocks with `annotations`, or refusal blocks) and `function_call` items. The
  documented `function_call` item has `name`, `arguments`, `call_id`, `status`; the tool-calling page lists no `id` for it (the protocols page
  lists `id` as optional on input). `status` is `completed`, `incomplete`, `failed`, `in_progress`, `queued` or `cancelled`. The shape of
  `incomplete_details` and of the response `error` object is **not defined** on the pages read.
- Usage: `input_tokens`, `output_tokens`, `total_tokens`, `input_tokens_details.cached_tokens`, `output_tokens_details.reasoning_tokens`
  (commentary counts toward `output_tokens`, not `reasoning_tokens`). All formats cost the same per token (`/docs/protocols`).
- Streaming events: `response.created`, `response.in_progress`, `response.output_item.added`/`.done`, `response.content_part.added`/`.done`,
  `response.output_text.delta`/`.done`, `response.function_call_arguments.delta`/`.done`, `response.completed`, `response.incomplete`,
  `response.failed` (carries `error.code` such as `server_error` and a sanitized message), the top-level `error` event, then `[DONE]`. No
  `response.reasoning_text.*` events are emitted (raw reasoning exists only encrypted).
- Not read: the retrieve, delete, cancel and count-input-tokens pages, and any Meta cookbook page on agent loops with Responses (a search found none).

## The Responses client (`MetaResponsesClient`): UNTESTED against the real service

What it does (`nightshift/models/responses.py`, selected per role with `api = "responses"` in `nightshift.toml`, default `chat`):

- Same `ModelClient` protocol, constructor, retry policy (429/500/502/503, `Retry-After`, 504 retried once as a stream, transient in-stream errors
  re-sent), fallback model, `disabled_params`, `completion_cap_ceiling`, secret scrubbing and `response.raw["_nightshift"]` facts as the Chat
  client; `budget.MeteredClient` charges it by call id exactly as before. `input_tokens`/`output_tokens`/cached tokens map onto the same `Usage`, so
  pricing and the ledger fields are unchanged.
- Wire: `input`/`instructions`, flat `tools` with `strict: false`, `text.format`, `reasoning.effort`, `max_output_tokens` (at least 16), `store: false`,
  `include: ["reasoning.encrypted_content"]` for every `muse-*` model. Never sent: `previous_response_id`, `tool_choice`, `stop`, `n`, `logprobs`,
  `truncation`, `stream_options`.
- Replay: the assistant message it returns carries a private attribute `_responses_items` (an inert `OpaqueItems`: the encrypted reasoning items, the
  commentary `phase` and the `function_call` ids, in the model's order). A later call rebuilds the Responses input from the Chat-shaped history
  (the single source of truth for text, tool names and arguments) and weaves the reasoning items back in at their original positions. Replay
  is skipped for another model (a fallback attempt), when the carried call ids differ from the message's, when `include` was stripped, and it
  drops a reasoning item that no assistant message or call follows. A 400 naming a reasoning item is retried once without replay; three such
  rejections turn replay off for the client. The supervisor loop carries the attribute across its rebuilt messages (`nightshift/agents/llm.py`);
  the patch worker builds its own messages and does not copy it, so the client also keeps a bounded memo of each tool-calling turn under its tool-call ids, scoped to the conversation (hash of its system messages and first user message) and to the turn's position in it, so lanes sharing a client never receive each other's reasoning.
  History compaction drops the reasoning with the turns whose results it stubs and keeps it on the turns it keeps verbatim.
- Histories are interchangeable: the Chat client builds its wire messages from the `ChatMessage` fields only and ignores the attribute.
- Untrusted data: the opaque items are never rendered. `OpaqueItems` cannot be iterated or JSON-serialised and prints only a count, a byte length
  and a hash; `response.raw` holds a structural summary; the ledger, dashboard and reports never see an encrypted blob.

Assumptions taken from the documentation and **not verified live** (`nightshift.models.responses.ASSUMPTIONS`; the Responses probes of
`nightshift doctor --live`, recorded under the `responses` key of the `--live-out` report, settle most of them):

1. `store: false` together with `include: ["reasoning.encrypted_content"]` is accepted, and a returned reasoning item replayed as
   `{type, id, summary: [], encrypted_content}` is accepted on the next turn.
2. A replayed `function_call` may carry the `id` the server returned next to `call_id`, `name` and `arguments`.
3. Commentary text before a `function_call` is replayed as an assistant message with `phase: "commentary"`; a `phase` the model returned is replayed.
4. Function tools use the flat shape; `strict: false` is sent explicitly because the default is not stated.
5. Encrypted reasoning is only valid for the model that issued it (**not documented**; the conservative reading, so a fallback model never gets it).
6. `status: "incomplete"` means the output limit was reached when `incomplete_details.reason` is `max_output_tokens` or absent (the shape is not
   documented); `content_filter` maps to `content_filter`.
7. Usage always arrives in the terminal event of a stream.
8. `text.format` strict `json_schema` accepts the same subset as Chat Completions `response_format`.
9. `max_output_tokens` above the configured maximum is a 400 (documented); the maximum itself is not stated.
10. `temperature` is accepted while reasoning is on (unknown; a 400 naming it strips it).
11. A `response.failed` event with code `server_error` is transient and the request can be re-sent.
12. Partner hosts (Fireworks, Baseten, Modal) are not known to implement the Responses API: a role with `api = "responses"` and a `base_url`
    other than the Meta Model API is a configuration error unless `allow_partner_responses = true`.

How it was tested: `tests/test_models_responses.py` against `tests/support/fake_meta_server.py`, whose `/v1/responses` endpoint enforces the rules
above and **verifies** the encrypted reasoning items it issues (it rejects an item it did not issue, records every replay, and optionally binds a
blob to its model), plus a full run of the simulated Muse brain with the supervisor and the patch worker on `api = "responses"`. That proves the
plumbing and that replay happens when the documented protocol is followed; it does not show that the real service behaves like the fake, that
replay improves quality, or what it saves. The cost argument in the previous section is still inference, not a measurement.

## Muse Code (read `/docs/muse-code`, `/docs/muse-code/extending`, the configuration and permissions pages and `/docs/muse-code/changelog`; the changelog as served was only partly readable)

Nightshift's Muse Code worker (`nightshift/agents/muse_code.py`) was built from these pages and is **tested only against a fake `muse` executable**. Nothing below was observed on a real run. The page facts in this section are as the worker's author and its verifier read them; the final documentation pass had no network and did not re-fetch these pages, so treat a flag name here as "documented, per that reading" until `probe_exec` confirms it on a machine with the binary.

- Install: `curl -fsSL https://dev.meta.ai/install.sh | sh` (a bootstrap that fetches a launcher and edits shell profiles; this repo never runs it automatically). Check with `muse --version`.
- **Default model: `muse-spark-1.2`** (configuration page), not the `muse-spark-1.3` Nightshift defaults to. `--model` is documented as a launch flag ("Use the `--model` flag at launch"; changelog 1.4.5 "`--model` already allows at launch"), but it is **not listed among the
  `muse exec` flags** on the headless page. The worker passes it after `exec` because the alternative is an unknown model answering; `nightshift doctor --live` (the Muse Code probe) on a machine with the binary is the only thing that can confirm `exec` accepts it.
- Headless: `muse exec "<prompt>"`, `--prompt-file F`, `--json` (JSONL events on stdout), `--disable-approval` (keeps the OS sandbox), `--yolo` (drops approvals AND the sandbox; trusts the workspace), `--max-model-steps N`, `--session-id UUID`, `--allow-workspace-switch`.
  Also documented in the changelog: `--disable-request-user-input` (1.4.9), `--disable-reminders` (1.4.3), `--no-session-log`, `--sandbox-network` with the modes `restricted`, `proxy-only` (the default) and `unrestricted`, `--disable-sandbox`, and `--reasoning-effort` as the effort flag. Exit codes: 0 done, 1 failed or cancelled, 2 usage.
  The exit code says how the run ended, not whether the work is correct; always gate on your own checks. Auth in CI: `META_API_KEY`.
- Output: since changelog 1.4.11 `muse exec --json` prints a **`run.usage`** line before each run ends, reporting token usage per model and the subscription allowance. It is the only JSONL event the changelog names; the field names inside it, and any `run.model.*` event, are **UNVERIFIED**
  (the worker parses several spellings defensively and treats the numbers as reported by Muse Code, not verified).
- Trust: the changelog contradicts itself. An early 1.4.x entry says a first run with no interactive terminal "stops with a message naming `--trust-workspace`" (one audit read it under 1.4.0, another under 1.4.3); 1.4.4 says the flag "now only decides whether the workspace's skills and rules load"; 1.4.13 says untrusted
  workspaces get no project rules, while the configuration page says project memory "still loads in untrusted workspaces". The worker does not pass `--trust-workspace`, keeps instruction files out of its private copy, and treats the stop message as an infrastructure error.
- Permissions: the page documents WRITE limits only ("write access to the workspace and a temporary directory, and keeps the rest of the filesystem read-only"); it says nothing about read confinement, and `--disable-sandbox` lifts the confinement of the file tools. So read scope is **unconfined as far as the docs say**.
  `muse exec` has included a web fetch tool since 1.4.3 and no documented flag turns it off. Changelog 1.4.13 says `--yolo` keeps `request_user_input` enabled while 1.4.9 implied it disabled it; moot here because `--yolo` is never passed and `--disable-request-user-input` always is.
- Muse Code workflows exist but are build-dependent and their script API is undocumented: **UNVERIFIED**, not used.

## Open questions to settle with a live key (`nightshift doctor --live --live-out demo/live_probe.json`, or `scripts/live_smoke.py`)

The smoke test (`nightshift/models/live_smoke.py`) records, without message content, a pass/fail/inconclusive status per probe.
Required probes use `max_completion_tokens=2000` at `minimal` effort because reasoning tokens count against the cap and Muse Spark always
reasons; a reply with `finish_reason == "length"` is reported **inconclusive**, never a failure of the feature. It answers:

- Does `muse-wonder-1.0` accept Chat Completions? (Docs: yes, same endpoints as Spark; the probe confirms. If not, the client falls back
  to `muse-spark-1.3` and keeps `max` or downgrades to `xhigh` as above.)
- Does a stream with `include_usage` return real usage on this host (`usage_estimated == False`)? Required.
- Which strict-mode keywords are refused: do PLAN, REVIEW, REFLECT, VERDICT and the six supervisor tool schemas (which carry `minLength`,
  `maxLength`, `pattern`, `minItems`, `maxItems`) pass? A rejection lists the keyword that triggered it.
- Is `max` accepted on `muse-spark-1.3` (docs: yes) and `none` a 400 (docs: yes)? Is `temperature` accepted with reasoning on?
- What is the real `max_completion_tokens` ceiling (16000 / 32000 / 64000 are tried)?
- May an assistant tool-call message carry `content: null`, or does it need `""`? Are two parallel tool calls answered in one turn?
  Do `tools` and `response_format` work together? Are both `system` and `developer` roles accepted?
- Which `x-ratelimit-*` headers (and `Retry-After` on a 429) does a real response carry?
- Does prompt caching hit with a prefix of at least 8k tokens, and where is `cached_tokens` reported? (Docs: `prompt_tokens_details`.)
- Does the Responses client work (`report["responses"]`, five probes that never affect `ok`)? Plain response; `include` with `store: false`
  returns an encrypted reasoning item; a tool call followed by a tool-result turn **with the reasoning item replayed** is accepted; `text.format`;
  a stream whose terminal event carries usage. The summary ends with `responses OK: set api = "responses" for the supervisor in
  nightshift.toml` or `chat only: ...`.
- Do partner hosts accept `prompt_cache_key`, strict `json_schema`, `reasoning_effort`, `stream_options`? The client strips an optional
  parameter named in a 400 and retries once (a host without `stream_options` then gets estimated, flagged usage).

## Not verified, deliberately not built on

- SAM 3.1 output encoding (`<|box;...|><|mask;...|>` records): format details unconfirmed.
- Muse Code TypeScript SDK (`@muse-code/sdk`): docs page was only a redirect.
- Responses API behaviour beyond the pages listed above, and everything about the Responses client on the live service (see the assumptions).
- The Chat Completions in-stream error chunk shape, the maximum output length, the minimum cacheable prefix and the Model API credit
  amount for hackathon participants (see above).

## Sources

- https://dev.meta.ai/docs/overview
- https://dev.meta.ai/docs/models
- https://dev.meta.ai/docs/protocols, https://dev.meta.ai/docs/protocols/chat-completions and https://dev.meta.ai/docs/protocols/responses
- https://dev.meta.ai/docs/api-reference/responses, .../responses/create-response and .../responses/schemas
- https://dev.meta.ai/docs/api-reference/chat-completions/schemas
- https://dev.meta.ai/docs/reasoning
- https://dev.meta.ai/docs/quickstart
- https://dev.meta.ai/docs/tool-calling
- https://dev.meta.ai/docs/structured-output
- https://dev.meta.ai/docs/token-counting
- https://dev.meta.ai/docs/prompt-caching
- https://dev.meta.ai/docs/pricing-rate-limits
- https://dev.meta.ai/docs/error-handling
- https://dev.meta.ai/docs/cookbook/streaming-responses and https://dev.meta.ai/docs/cookbook/error-handling-retry
- https://dev.meta.ai/docs/muse-code, https://dev.meta.ai/docs/muse-code/extending and https://dev.meta.ai/docs/muse-code/changelog
- https://cerebralvalley.ai/e/meta-ai-developer-hackathon
