# Demo script (CPU, fake, about five minutes)

A short narrated walkthrough of the fake end-to-end demo. Everything shown
is labelled fake: this demonstrates wiring and safety, not live training
and not model improvement.

## 0. Setup (say this first)

"Nightshift runs research agents that post-train a small open model, plus
an outer agent that improves the researchers themselves. This demo runs
entirely on CPU with deterministic fake agents, so every number is a wiring
check. The real question the system is built to answer — does the improved
SFT procedure beat the original SFT procedure from the same base checkpoint
under matched resources — needs a live GPU run, and this repository does
not claim one yet."

## 1. Run the tests

```sh
python -m pytest -q
```

Point out: budget rejection happens before any provider create call, the
ledger detects tampering, private splits never reach an agent, pickle
checkpoints are refused, and a failed evaluation is never averaged as zero.

## 2. Run the demo

```sh
nightshift demo --root ./demo-out
```

While it runs, narrate the progress lines: the original pack is frozen,
two outer rounds screen candidate packs on the private selection split,
then a fresh matched comparison runs original versus final pack. Spend is
shown against the ceiling on every line, with a `[fake]` label.

## 3. Inspect the evidence

```sh
nightshift status ./demo-out/run
nightshift ledger verify ./demo-out/run
cat ./demo-out/run/report.md
```

Show three things in the report:

1. The FAKE banner and the "No evidence of improvement [fake]" conclusion —
   a fake run cannot claim improvement by construction.
2. The outer-rounds table: each candidate pack version, whether it froze,
   whether screening accepted it, and the decision reasons.
3. The evaluation-evidence table: every Inspect log hash next to its
   ledger sequence number, so any score can be traced to an immutable log.

Then show one trial directory (`./demo-out/run/trials/<trial-id>/`): two
isolated researcher workspaces, the trial-only `shared/notes/`, and the
controller's own records. Nothing is shared across trials except what the
frozen pack itself contains.

## 4. Show the brakes

```sh
nightshift stop --leases ./demo-out/run/leases.json
```

Explain: the lease file lists exactly the rentals this run created, and
stop destroys exactly those — from any machine, without the run directory.
In the demo they are already destroyed, so it reports `none`.

## 5. Close honestly

"What would make this real: pin the base-model revision, supply frozen
operator manifests, configure a real provider and agent backend, and run
the same pipeline so training, private evaluation on a separate trusted
host, and the final comparison all happen with live components. Until that
recorded run exists, the honest claim is: the controller, the safety
properties, and the full two-loop wiring work on CPU — improvement is not
yet demonstrated."
