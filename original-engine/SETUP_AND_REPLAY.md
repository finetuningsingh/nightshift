# Setup, tests and replay

## 1. Replay the recorded study (no GPU, no network, no keys)

```sh
python3 scripts/replay_recorded_study.py
```

Python 3.11 or newer, standard library only. It does the following:
- Checks that the sanitized summary `evidence/recorded-study/lab2p-summary.json` holds the complete, hash-linked event sequence of the recorded main ledger (56 events) with the head hash of each trial ledger. The raw ledgers stay private because they contain free-text decision records. With `--private-ledger DIR`, the script also verifies the private ledger and checks every summary hash against it.
- Prints every trial's arm, assignment, pack, seed, status and failure, read from recorded events.
- Prints every score's aggregate pass-all and mean constraint fraction, the candidate proposal, the contamination scans, the scoring failures and the GPU rental total.

This is a replay of what was recorded. It does not re-run any model.

## 2. Run the test suite

```sh
python3 -m unittest discover -t . -s tests
```

The core and the lab need only the standard library. The tests use fake GPUs, fake model servers and fake researcher runtimes; they make no network calls and rent nothing.
- Tests that need `git` (pack versioning) or `bash` (hostkit scripts) expect them on `PATH`.
- Inspect AI tests are skipped unless `pip install -e .[lab-eval]` has been run.
- Results of the check run for this extract are in [RESULTS_AND_EVIDENCE.md](RESULTS_AND_EVIDENCE.md#test-run-of-this-extract).

## 3. Run the fake CPU demo of the core engine

```sh
python3 -m nightshift --help
```

Everything shown by the bundled demo tasks is a **CPU simulation with fake components**. It shows the workflow and the ledger, not a trained model.

## 4. Running the live system (needs paid GPUs and logins; not reproducible from this folder alone)

The live study needed all of the following. `docs/RUNNING_LIVE.md` describes the setup.
- A Thunder Compute account. Each job gets its own A100 80GB rental with a hard stop, a detached watchdog and an off-host reaper (`scripts/offhost_reaper/`).
- A researcher host with docker, CORAL at commit `0123dfb`, and a Muse Code 1.4.14 login used by the `muse-spark-1.3` researchers inside the container.
- A control host with the Muse Code login for the `muse-wonder-full-1.0` reviser, the private data splits and the ledger.
- The IFEval-G verifiers fetched by `scripts/lab_setup_verifiers.sh` (pinned in `scripts/lab_pins.json`).
- `scripts/two_loop/run_lab.py`, which sets the Muse routes and reads the GPU provider token from the operator's own login. It strips `MUSE_API_KEY`, `BASETEN_API_KEY` and `META_API_KEY` so the CLIs cannot fall back to API billing.

Some paths in `scripts/two_loop/run_lab.py` and `scripts/reference_run/` are the operator's own machine paths and host names, and must be changed for another setup.

Known residual faults, all recorded and none fixed after the fact for the study itself:
- **Contamination scan design gap.** Held-out instruction templates are not counted as boilerplate. The scan flags 51 of the 300 approved H1 training prompts, so any H1 trial that trains on them is marked contaminated. A fix would count the templates of a held-out trial's own assignment as boilerplate. It is not applied.
- **Harness guard bug.** The final-phase guard refused mid-trial dev evaluations. Fixed in this extract (regression test included). It affected `t-0008`.
- **SSH failures on fresh rentals.** Two scoring attempts in `t-0002` failed this way. Fixed by a bounded wait for SSH readiness.
- **Researcher job sizing.** Several trials lost time or failed on the researchers' own caps and recipes within the 72-minute account.
