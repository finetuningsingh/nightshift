# Golden scripted run

A recording of `python3 -m nightshift demo` on the bundled `law-discovery` task: the **scripted backend**, so no language model
produced it. The experiments were really executed and really graded; the faults in them were planted by the script. It is a
fault-injection harness that demonstrates the audit machinery, not an AI result. The report says so in its first lines.

* `ledger.jsonl`: the append-only, hash-chained ledger (312 events). It is the only state; everything else is derived from it.
* `report.md`, `report.html`: the morning report written from the ledger.
* `memory.md`: the shared memory the agents saw at the end.

Head anchor: `312:e482fb6529a23efe`. The anchor belongs to THIS committed ledger: every fresh `demo` run has its own, so
`verify --expect-head` with this value on your own run will (correctly) report a mismatch.

```bash
python3 -m nightshift verify demo/golden_scripted --expect-head 312:e482fb6529a23efe     # chain consistent
python3 -m nightshift serve demo/golden_scripted --replay-duration 150                   # replay as a 150 second recording
python3 -m nightshift report demo/golden_scripted --out /tmp/golden-report               # regenerate the report from the ledger
```

Experiment ids differ from run to run (the damped model is `e010` here, `e011` in some runs); the outcomes by hypothesis do not.
The honest control run is a separate command, `python3 -m nightshift demo --control`, and is not part of this folder.
