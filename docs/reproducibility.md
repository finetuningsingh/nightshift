---
type: concept
title: Reproducing Nightshift runs
description: How to run the CPU demo and test suite offline, and what a real GPU run still requires before it can be claimed.
timestamp: 2026-11-04T09:15:00-05:00
---

# Reproducibility

## CPU demo (fake, offline after install)

```sh
pip install -e ".[test]"
python -m pytest -q          # full CPU test suite, no network needed
nightshift demo              # end-to-end fake run in a temp directory
nightshift demo --root ./demo-out
nightshift status ./demo-out/run
nightshift ledger verify ./demo-out/run
nightshift report ./demo-out/run
```

The demo uses the fake provider, fake researcher/reviser agents, the
bundled synthetic toy dataset, and real Inspect AI runs against Inspect's
`mockllm/model`. Every record is labelled `"fake": true` and the report
carries a FAKE banner. A fake run is a wiring and safety demonstration —
it is never evidence that training or researcher improvement works live.

One offline detail: Inspect AI estimates token counts with the tiktoken
`o200k_base` encoding, which downloads its BPE file on first use. The demo
itself needs no network beyond package installation if that encoding is
cached during setup:

```sh
python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
```

## Docker sandbox

`docker/Dockerfile.cpu` (python:3.12-slim) installs the package with its
test extra and pre-caches the tiktoken encodings into
`TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache` at image build time. The test
suite is run with `--network none`, a read-only root filesystem, and tmpfs
`/tmp` and `/root`; Nightshift writes only inside the run directory, `/tmp`,
or paths handed to it.

## Determinism

The controller owns all spending arithmetic. The fake provider's simulated
clock, the fake generator, fake training, and the fake Inspect outputs are
deterministic functions of the config seed, pack version, and sample ids,
so a given commit plus a given config reproduces the same scores and
content hashes. No stronger determinism is claimed: ledger timestamps,
reservation ids, and lease ids contain fresh random identifiers and may
differ between repeats. Pack versions and checkpoint ids are content
hashes; Inspect log files are hashed byte-wise into every evaluation
record.

## Real-run path (not yet validated live)

Nothing below has been demonstrated with a real GPU rental in this
repository. The adapters say so in their docstrings and fail clearly
rather than silently succeeding:

1. Pin `models.base_model` (`meta-models/Muse-Glimmer-30B`) to an exact
   `models.base_revision`. Real generation/training refuses an unpinned
   revision such as `main` or the `operator-must-pin-revision` placeholder.
2. Supply frozen operator manifests for `train_prompts`, `dev`, `selection`,
   and `test` (source, revision, row count, SHA-256). The public repo
   bundles only synthetic toy rows written for the demo.
3. Configure one real provider (the SSH adapter skeleton) and budgets.
   Every GPU request is reserved before any provider create call;
   `nightshift stop --leases <run>/leases.json --provider <name>` destroys
   exactly the owned rentals recorded there, requires the lease file to
   already exist and the provider to match every lease, and exits nonzero
   when a destruction is not confirmed.
4. Real agents run through the Muse CLI (`muse exec --json`,
   researcher `muse-spark-1.3`, reviser `muse-wonder-full-1.0`) using
   subscription login. API-key variables are stripped from the child
   environment unless the operator explicitly opts into API-key mode.
5. Training renders the pack's Axolotl QLoRA template with controller-owned
   keys forced, then runs `axolotl train <config.yaml>` as a provider job.
6. Private evaluation must run on a fresh trusted host separate from
   training, using an operator-supplied Inspect model string for the
   served base+adapter. The `test` split is claimed exactly once per final
   comparison id.

Until a recorded live run exists (acceptance L1–L5), all "improvement"
language in reports from fake runs is suppressed by construction: the
report states "no evidence of improvement" for any run that used a fake
component.

## Known limitations

Some boundaries are deliberately documented here rather than claimed as
solved. Pack policy is lexical: the recipe is read with a flat key parser
and pack tool files get no semantic YAML or code policy, so a determined
reviser might phrase an edit the literal checks do not recognise. Budget
enforcement is not one cross-process transaction: the dollar ceiling is
locked in the shared budget file, but per-trial seconds and concurrency
are checked around it, separate run roots keep separate lease stores,
and no independent reaper can destroy a rental while a provider call is
blocking. Lease files now record the provider name and stop refuses a
mismatch or an unconfirmed destruction, but there is still no
authoritative provider-side listing proving a rental is absent, and the
full provider binding for real lease files remains not yet validated
live, as do all real provider, generation, training, agent, and private
evaluation lanes.
