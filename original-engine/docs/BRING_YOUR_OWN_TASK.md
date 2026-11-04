# Bring your own task

A Nightshift task is a directory. The researcher (a Muse Spark supervisor, or the scripted harness) edits one or two files,
the sandbox runs them, and a **trusted grader** that the researcher cannot see scores the result. If you can write a
`grade.py` that turns "did this change help?" into a number, Nightshift can run the research loop on it.

```
tasks/<name>/
  task.toml          objective, metric, which files are editable, commands, seeds, margins
  template/          copied into every experiment workspace. Editable files, read-only code, visible data.
  trusted/           grader and labels. NEVER copied into a workspace.
  scripted/          optional offline proposal bank (for the fault-injection demo)
```

Start from `tasks/law-discovery` (standard library only) or `tasks/spiral-classifier` (numpy). Copy one, rename it, then
change the template and the grader.

## 1. `task.toml`

```toml
name = "my-task"
description = "one line"
brief = """Plain text the model reads: the objective, what it may edit, what the metric means, the rules.
Do not mention seeds or any held-out split here."""
objective = "minimize"          # or "maximize"
primary_metric = "heldout_rmse" # the key your grader returns in "metrics"
train_metric = "train_rmse"     # optional; lets the auditor catch "better on train, not on held-out"

[workspace]
editable = ["model.py"]         # globs the researcher may modify
scratch  = ["outputs/**"]       # files a run may write (ignored by the protection checks)

[run]
cmd = ["python3", "run.py"]     # runs inside a throwaway copy of the workspace, with a scrubbed environment
timeout_sec = 20

[grade]
cmd = ["python3", "{task_dir}/trusted/grade.py", "--workspace", "{workspace}", "--seed", "{seed}", "--split", "{split}"]
timeout_sec = 20
seeds = [0, 1, 2]                                 # dev evaluation
replication_seeds = [100, 101, 102, 103, 104]     # sealed evaluation, used only to confirm a win
noise_margin = 0.02             # smallest improvement that matters to you (must be > 0); see section 4
sigma_floor = 0.02              # the unmodified baseline's per-seed std, measured over >= 12 seeds; see section 4. NOT 0.
sealed = true
requires = []                   # python modules the run imports, e.g. ["numpy"]
stall_sec = 10                  # no output for this long: the watchdog kills the run
```

Anything the model could read (the `brief`, the workspace, the tool results) must not reveal the sealed split or the
replication seeds. Everything else about the evaluation lives in `trusted/`.

## 2. What the grader must do

`grade.py --workspace W --seed S --split dev|sealed` prints one JSON object as the **last line of stdout**:

```json
{"valid": true, "score": 0.1234, "split": "dev", "n_eval": 120,
 "metrics": {"heldout_rmse": 0.1234, "train_rmse": 0.05}, "notes": ""}
```

* Exit code 0. `split` echoes the request. `score` and every metric are finite numbers (NaN and Infinity are rejected).
* `W` is a *staged* directory that contains only a sanitised copy of the run's `outputs/`. The grader never executes
  workspace code and never imports from it. Parse predictions as CSV or JSON, nothing else (no pickle, no `eval`, no
  `np.load(allow_pickle=True)`). `nightshift tasks` greps for the forbidden calls.
* The labels live in `trusted/`. The grader computes both the held-out metric and the train metric itself, so the run cannot
  report a flattering number about itself.
* `valid: false` (missing, short or malformed predictions) is a model-side failure, not an infrastructure failure.

## 3. Seeds and the two splits (this is what makes a win mean something)

A seed must change **both** the training sample and the evaluated rows, otherwise "replication" just re-measures the same
number:

* `template/data.py` exposes `load_train(seed)`: a seeded subsample of a fixed training pool.
* `run.py` predicts *every* row of the unlabeled evaluation inputs.
* The grader keeps the labels and, for `--split S`, selects a fixed row set for that split and draws a seeded subsample of it.
  The `dev` and `sealed` row sets are a fixed random 50/50 partition of the same pool, so a gap between them is overfitting to
  dev, not distribution shift.

Why two splits: the researcher sees the dev score of every experiment, so after a dozen experiments the dev set has effectively
been trained on. The sealed split is graded only when you ask to confirm a candidate, its numbers never reach a prompt, and
each query is capped. Run `nightshift tasks`: it grades the unmodified template on every seed and rejects a task whose per-seed
scores are all identical (`seeds_vacuous`), whose margin is zero, or whose seeds overlap. It also estimates the baseline's
per-seed std (section 4).

**The shipped data is public.** The committed law-discovery split is in this repository, so anyone who read it knows its sealed
rows. For a live demo whose sealed split nobody has seen, draw a fresh one with
`python3 tasks/law-discovery/trusted/make_data.py --seed "<some private text>"` (the default seed reproduces the committed files
byte for byte; without `--out DIR` the files are written over the task's committed data, so do not commit them), keep the text
and the new `trusted/` files private, and re-run `nightshift tasks` because the noise numbers in
`task.toml` were measured on the shared split. A task of your own should generate its data the same way: from a seed you can
change.

## 4. Calibrating the decision rule (`noise_margin` and `sigma_floor`)

The default advice "leave `sigma_floor` at 0 and `noise_margin` at 0.02" is wrong for most tasks. The engine promotes a candidate
only when its improvement over the incumbent exceeds a margin

```
margin = max(noise_margin, t * max(s, sigma_prior, sigma_floor) / sqrt(n))
```

where `s` is the std of the candidate's paired per-seed differences, `sigma_prior` is the baseline's own per-seed std on the
dev seeds, `n` is the number of seeds and `t` the critical value (one-sided 95% on dev, two-sided 95% on the sealed split). The
baseline's std from three seeds is a very noisy estimate (for law-discovery, whose true value is about 0.036, its 5/50/95%
quantiles are 0.009, 0.030 and 0.064; the realized value is 0.010). If `sigma_floor` is below the true std, the prior is inert and
the dev gate for a noisy candidate is nothing more than a plain 5% one-sided t-test on three seeds, which passes one null in
twenty and compounds over a long run (`docs/NULL_BENCHMARK.md`). The procedure:

1. **Measure the baseline's per-seed std.** `python -m nightshift tasks` grades the unmodified template on at least 12 seeds
   per split (20 by default; one baseline evaluation per seed, a few seconds) and prints the std for each split and a
   recommended `sigma_floor`. It warns when `sigma_floor` is below 75% of it, or `noise_margin` below half of it. A sample std
   of 20 seeds is itself only good to about +-16%; if the printed numbers differ much from run to run, rely on the larger.
2. **Set `sigma_floor` to that std** (two significant digits). This is what makes the prior real.
3. **Set `noise_margin` to the smallest improvement that matters to you**, not smaller than about the run-to-run noise (law-discovery
   and spiral-classifier use about 1x the std). It is a hard minimum effect: nothing below it is ever promoted.
4. **Check the consequences** with `python3 scripts/null_benchmark.py` (it reads law-discovery's constants from its `task.toml`;
   pass `--sigma-floor`, `--noise-margin`, `--sigma-baseline` for yours; the seed counts and the sealed-query cap are also
   law-discovery's): the false-promotion rate per run for 16 and 48 experiments and the power table.

**Minimum detectable effect.** With three dev seeds the gate cannot see small effects. The dev margin is at least
`t * sigma_floor / sqrt(3)` = `2.92 * sigma_floor / 1.73` = about 1.7 sigma_floor, and an effect has to clear it on dev AND clear the
sealed margin on fresh seeds. At the calibrated floor (0.036) a law-discovery candidate with small paired noise is promoted half
of the time at an improvement of about 5% of the baseline RMSE (the independent audit measured about 3% before the floor was
calibrated, at the price of the false-promotion rates in `docs/NULL_BENCHMARK.md`); smaller real effects are mostly not promoted.
That is the intended trade.
More seeds shrink the margin (`t / sqrt(n)` falls) at the cost of runtime.

**The ceiling effect.** Once the incumbent is within the margin of the best achievable score, nothing more can be promoted,
however good the idea. For law-discovery the damped model has RMSE about 0.06 (0.061 on dev and 0.066 on the sealed split in the scripted demo) against a noise floor of 0.05 (the measurement noise
cannot be predicted away), so no further improvement above 0.035 is possible: after it is promoted the run can only end. For
spiral-classifier the polar-feature incumbent scores 0.93 to 0.95 and the margin is at least 0.034 (about 0.045 once the
candidates' own paired noise is as large as the baseline's), so a candidate would have to score roughly 0.97 to 0.99 or more,
which the few percent of wrong labels do not allow. If your task's headroom is smaller than `noise_margin`, the honest
outcome of a run is "no further improvement", and the report says so.

## 5. Optional: a scripted bank

`scripted/manifest.json` lets the offline demo replay a fixed set of proposals (including deliberately faulty ones) through the
real sandbox, grader and auditor. The faults are scripted; every score is really computed. Read
`tasks/law-discovery/scripted/manifest.json` for the format. A task without a bank still works with `--backend meta`.

## 6. Validate and run

```bash
python -m nightshift tasks                                   # structure checks, calibration, recommended sigma_floor
python -m nightshift run --task tasks/my-task --backend scripted     # needs a scripted/ bank
python -m nightshift run --task tasks/my-task --backend meta --max-usd 3
```

`nightshift tasks` lists the tasks under `./tasks` (or the checkout's `tasks/`, or `$NIGHTSHIFT_TASKS_DIR`; `--root DIR` names another directory). It grades the unmodified template on 20 seeds per split, so it takes
tens of seconds per task. A task whose optional `requires` module is not installed is printed as `SKIP` (its calibration could not run) and does not fail the command; any other problem is a `FAIL` and exit status 1. `--task` accepts any
directory that holds a `task.toml`, whether or not the bundled tasks are installed (a plain `pip install .` ships none; see `docs/ARCHITECTURE.md` A.8). To try your task against the real audit machinery with real model roles,
`--plant tamper,overfit` adds labelled scripted attacker lanes to a `--backend meta` run, but only for behaviors your `scripted/manifest.json` has planted proposals for.

## Limits worth knowing

* The sandbox is best-effort containment for code written by a language model, not a security boundary. The labels are never copied into
  the workspace. Only in the `userns` isolation mode (`nightshift doctor` shows which mode you get) are they also out of the run's reach by path;
  in `unshare-net` and `none` a run can read `trusted/`, and the protection is that a change to it is detected afterwards (content hash and a stat
  snapshot), not that a read is prevented. Keep that in mind before you put a secret into `trusted/`. See `docs/ARCHITECTURE.md` section 11.
* Any write outside the `scratch` globs during a run is a violation, including library caches written under `HOME` (which is the
  workspace). A library that writes caches there will not work.
* Experiments are meant to be short. For a long GPU job, make `run.cmd` launch and wait for it, raise the timeouts, and write
  small prediction files; the staged outputs are capped (8 MiB per file, 32 files).
