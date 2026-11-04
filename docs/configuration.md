# Configuration

One TOML file describes a Nightshift run. `nightshift init --demo` writes a
complete CPU demo config that selects the fake provider, fake agents, and
the bundled toy data. Edit a copy of that file for any other run.

## Required budgets (no defaults)

Every field in `[budgets]` is required; a missing field is a configuration
error, never a silent default.

```toml
[budgets]
aggregate_ceiling_usd = 10.0
budget_file = "run/budget.json"          # shared across run roots that point at it
per_trial_gpu_seconds = 3600
max_rental_lifetime_seconds = 1800
max_concurrent_rentals = 2
```

The controller reserves `price × (compute + setup + transfer + evaluation
seconds)` before any provider create call and refuses a request that cannot
fit the remaining ceiling, the remaining per-trial seconds, the rental
lifetime, or the concurrency limit.

## Models and base checkpoint

```toml
[models]
researcher = "muse-spark-1.3"            # CLI-backend default
reviser = "muse-wonder-full-1.0"        # CLI-backend default
base_model = "meta-models/Muse-Glimmer-30B"
base_revision = "operator-must-pin-revision"   # replace with an exact revision
fake_agents = true                       # demo only; every record is labelled fake
```

Real generation and training refuse an unpinned `base_revision` (`main`,
`latest`, empty, or the placeholder above). The requested model id and,
where the backend reports it, the served id are recorded for every agent
call.

## Agent backends

The `[agents]` table selects how researcher and reviser roles run. The
backend is held constant inside a matched comparison; never change it
between the original-SFT and improved-SFT arms.

Fake agents (demo and tests) need no `[agents]` table: they are selected
whenever the provider is fake or `fake_agents = true`.

Muse backend (defaults shown):

```toml
[agents]
backend = "muse-cli"
executable = "muse"
max_model_steps = 30
api_key_mode = false    # false strips API-key variables; subscription login is used
allow_unsandboxed_cli = false  # CLI agents refuse to run unless this is explicitly true
```

The Muse lane is not yet validated live with controller-enforced
workspace isolation, so it fails closed until an operator explicitly
sets `allow_unsandboxed_cli = true`.

OpenAI-compatible API backend. Use this for a hosted-model evidence lane,
for example a Muse Model API reviser with Baseten Muse Spark Open
researchers in **both** arms. Credentials come from the environment
variable named by `api_key_env` and are never written to config, logs, or
the ledger. The adapter fails closed when the variable is empty.

```toml
[agents]
backend = "api"

[agents.api]
researcher_base_url = "https://inference.baseten.co/v1"
researcher_model = "meta-models/muse-spark-open-1026"
researcher_api_key_env = "BASETEN_API_KEY"
reviser_base_url = "https://api.meta.ai/v1"
reviser_model = "muse-wonder-1.0"
reviser_api_key_env = "META_API_KEY"
timeout_seconds = 300
max_tokens = 4096
max_calls = 200         # optional hard cap on API calls per role endpoint
```

Base URLs must be `https://` (plain `http://` is accepted only for
loopback development endpoints), every configured credential variable is
preflighted before a run starts, and a response that omits its served
model fails instead of being recorded as the requested model.

Notes on the API lane:

- `muse-wonder-1.0` (Model API) and `muse-wonder-full-1.0` (Muse)
  are different ids. Configure the id the endpoint actually serves; the
  response's served model is recorded and a mismatch fails the call rather
  than silently substituting.
- API output is reasoning and code only. It is never used as a supervised
  fine-tuning target; real targets are always verified local base-model
  generations on approved training prompts.
- The API adapters are not yet validated live in this repository. Until a
  recorded live run exercises them, treat every API-lane result as
  unvalidated.
- The researcher reply must be one JSON request object (types: `generate`,
  `filter`, `train`, `evaluate_dev`, `submit`, `note`) or the word `DONE`.
  The controller validates it with the same strict validator as every
  other request file. The reviser reply must be a JSON object of the form
  `{"files": {"relative/pack/path": "full new file text", ...}}`; the
  controller applies it inside the scratch pack only, then runs the pack
  policy check before freezing.

## Provider

```toml
[provider]
name = "fake"     # "fake" for CPU work; "ssh" is an unvalidated skeleton
```

Provider options such as `gpu_type`, `hourly_price_usd`, and the evaluation
model string go in the same table and are read as `provider.options`.

## Outer loop, evaluation, comparison

```toml
[outer]
max_rounds = 2
screen_trials = 1
acceptance_margin = 0.0
researchers_per_trial = 2

[evaluation]
max_retries = 1
temperature = 0.0
max_tokens = 128

[comparison]
comparison_trials = 2
margin = 0.0
bootstrap_samples = 1000

[verifiers]
training = ["max_words", "required_keyword", "valid_json"]

[pack]
dir = "pack"
```

The primary contrast is original SFT versus Nightshift-improved SFT from
the same base checkpoint under matched budgets, seeds, scorer, and
decoding. Screening on the selection split is adaptive and biased; only
the final fresh comparison on the test split estimates the change, with
failed trials retained in the denominator.
