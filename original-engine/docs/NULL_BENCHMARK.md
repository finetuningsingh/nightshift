# Null-hypothesis benchmark: how often does an auto-researcher fool itself?

A Monte Carlo over simulated research campaigns using the shipped decision code (`nightshift.grading.decide_outcome`, `T95`, `T975`). It checks the **decision rule** under a stated noise model; it says nothing about any language model. It models the configuration that ships (`tasks/law-discovery/task.toml`), including the noisy estimate of the baseline's std that the engine uses as its prior. Section 6 re-checks the model with null candidates drawn through the task's real data module and grader, and the section after it is a separate record of what an independent audit measured on the real pipeline. Regenerate with `python3 scripts/null_benchmark.py --campaigns 20000 --seed 7 --write docs/NULL_BENCHMARK.md` (about 20 minutes on one core).

## Setup

* 20,000 campaigns per cell, up to 48 candidates per campaign (every table reads the first k of them), random seed 7.
* Task constants (law-discovery, read from `task.toml`): noise_margin 0.035, sigma_floor 0.036, 3 dev seeds, 5 sealed seeds. Budget: max_replications 6 (sealed queries per run).
* The prior is not a constant. Each campaign first measures the baseline on 3 dev seeds; the sample std of those scores is that campaign's `sigma_prior` (true baseline per-seed std 0.036), and `decide_outcome` takes `s_eff = max(paired s, sigma_prior, sigma_floor)`. The sample std of 3 scores is a very noisy estimate, which is why the floor matters.
* Paired per-seed differences are i.i.d. Normal(theta, sigma_d^2), independent between the dev split (3 seeds) and the sealed split (5 seeds). Row families: small noise (sigma_d 0.018, 0.036) and noise **larger than the prior** (0.05, 0.07).
* Margins at the shipped floor: dev gate `max(0.035, 2.92 * s_eff / sqrt(3))` = 0.0607 when s_eff is the floor; sealed gate `max(0.035, 2.776 * s_eff / sqrt(5))` = 0.0447. When the sample std s of a candidate's paired differences exceeds the floor and the prior, s_eff = s and the dev gate is a one-sided 5% t-test on 3 seeds (with the noise_margin as an additional minimum effect).
* **raw**: promote the best candidate whenever its mean dev improvement is above zero. **t-test**: one-sided 95% t-test on the dev seeds, no prior on the noise. **dev gate**: Nightshift's dev margin alone. **Nightshift**: dev gate and then the sealed gate (two-sided 95% critical values), with at most 6 sealed queries per run.

## 1. False wins per run: no candidate has any real effect (theta = 0, sigma_d 0.036)

Share of runs (campaigns) that end with a claimed improvement although nothing improved. A run tests the stated number of candidates.

| candidates per run | raw | t-test | dev gate | **Nightshift** (cap 6 sealed queries) |
|---|---|---|---|---|
| 1 | 50.0% | 4.90% | 0.06% | **0 of 20,000** |
| 4 | 93.6% | 18.4% | 0.29% | **0 of 20,000** |
| 16 | 100.0% | 55.6% | 1.36% | **0.01%** |
| 48 | 100.0% | 91.5% | 4.15% | **0.01%** |

## 2. False promotion by candidate noise (theta = 0)

The same simulation for each row family of paired noise. Per candidate: the share that pass the dev gate, the share of those that then also pass the sealed gate, and the share that pass both. Per run: the probability that at least one null is promoted within 16 and 48 experiments (the sealed cap of 6 applied).

| paired noise sigma_d | per candidate: dev gate | sealed, given dev | both | t-test per candidate | per run, 16 experiments: dev gate | **Nightshift** | per run, 48 experiments: dev gate | **Nightshift** |
|---|---|---|---|---|---|---|---|---|
| 0.018 | 0.00% | 0.00% | 0 of 960,000 | 4.99% | 0.00% | **0 of 20,000** | 0.00% | **0 of 20,000** |
| 0.036 | 0.09% | 0.23% | 0.00% | 5.00% | 1.36% | **0.01%** | 4.15% | **0.01%** |
| 0.050 | 0.67% | 0.91% | 0.01% | 4.97% | 10.2% | **0.09%** | 26.7% | **0.29%** |
| 0.070 | 1.94% | 1.86% | 0.04% | 5.00% | 26.5% | **0.50%** | 58.5% | **1.73%** |

The larger a candidate's paired noise is compared with the floor (0.036) and the prior, the closer its dev gate gets to a plain one-sided 5% t-test on 3 seeds: the margin is then `t * s / sqrt(3)` with the candidate's own s instead of the floor, and the `noise_margin` only adds a minimum effect size. The t-test column is that limit (5% per candidate). The floor keeps moderately noisy candidates well below it (the dev-gate column), the gap closes as the noise grows, and a run that tries many experiments gives the dev gate many chances (the per-run dev-gate columns). What the sealed stage then buys: a fresh draw of independent noise on 5 new seeds against a two-sided 95% threshold, so a null that got through dev passes it only a few percent of the time or less (column 'sealed, given dev'). The product of the two gates, and the cap on sealed queries, is the protection, not the dev gate: the per-run figures above are what a long run costs. The cap of 6 sealed queries bounds how many nulls the sealed stage can ever wave through.

## 3. The winner's curse: what the dev score of a winner is worth

Among runs where some null candidate passed the dev gate within 16 candidates (theta = 0): the mean dev improvement the researcher would have reported for the first such candidate, versus the same candidate's mean improvement on fresh sealed data.

| paired noise sigma_d | runs with a dev-gate winner | mean reported dev improvement | mean sealed improvement (true value 0) |
|---|---|---|---|
| 0.018 | 0 of 20,000 | n/a | n/a |
| 0.036 | 272 of 20,000 | +0.0680 | -0.0019 |
| 0.050 | 2,035 of 20,000 | +0.0740 | -0.0008 |
| 0.070 | 5,294 of 20,000 | +0.0854 | +0.0003 |

## 4. The price: sensitivity

Probability that a single candidate with a real improvement `theta` (in the units of the metric, RMSE for law-discovery) is promoted by Nightshift (dev gate and sealed gate). Trust costs sensitivity, and the table says how much. The baseline's RMSE is about 1.6, so 0.05 is about 3% of it.

| true improvement theta | sigma_d 0.018 | sigma_d 0.036 | sigma_d 0.050 | sigma_d 0.070 |
|---|---|---|---|---|
| 0.000 | 0 of 20,000 | 0 of 20,000 | 0.01% | 0.04% |
| 0.020 | 0.00% | 0.05% | 0.25% | 0.27% |
| 0.030 | 0.01% | 0.56% | 0.73% | 0.69% |
| 0.040 | 0.46% | 2.67% | 2.41% | 1.47% |
| 0.050 | 7.62% | 9.12% | 5.54% | 3.28% |
| 0.060 | 32.6% | 21.3% | 12.5% | 5.75% |
| 0.075 | 71.5% | 47.0% | 27.4% | 12.4% |
| 0.100 | 92.9% | 80.3% | 57.7% | 29.8% |
| 0.125 | 98.2% | 94.5% | 80.0% | 51.4% |
| 0.150 | 99.7% | 98.8% | 92.0% | 69.9% |
| 0.200 | 100.0% | 100.0% | 99.0% | 90.2% |

The smallest improvement that can pass at all is the larger of the two margins: 0.0607 on dev and 0.0447 on the sealed split (candidates with small paired noise). The improvement at which half of such candidates are promoted (50% power, interpolated in the table), the minimum detectable effect:

* sigma_d 0.018: about 0.067, 4.2% of the baseline RMSE (1.6)
* sigma_d 0.036: about 0.077, 4.8% of the baseline RMSE (1.6)
* sigma_d 0.050: about 0.094, 5.9% of the baseline RMSE (1.6)
* sigma_d 0.070: about 0.123, 7.7% of the baseline RMSE (1.6)

A smaller real effect cannot be told from noise with 3 dev seeds and is mostly not promoted: that is the intended trade, and a ceiling as well: once the incumbent is within the margin of the best achievable score, nothing further can be promoted.

## 5. What calibrating `sigma_floor` changed

The shipped task has fixed seeds and a fixed template, so its REAL prior is one number: the unmodified baseline's std on dev seeds 0, 1, 2, which is 0.0100. Before the floor was calibrated (floor 0.02) that prior and the floor were both below the real paired noise of many candidates, so their margin was the candidate's own `t * s / sqrt(n)` (at least the `noise_margin`): a bare t-test. With the floor at 0.036 the margin uses `max(s, floor)`. The sections above draw the prior per campaign instead (a task whose baseline seeds are not fixed); the three configurations compared here:

| paired noise sigma_d | configuration | per candidate: dev gate | per candidate: both gates | per run, 16 experiments | per run, 48 experiments |
|---|---|---|---|---|---|
| 0.036 | previous: floor 0.02, prior 0.0100 (the audited configuration) | 1.84% | 0.01% | 0.22% | 0.65% |
| 0.036 | shipped: floor 0.036, prior 0.0100 (the task's realized prior) | 0.12% | 0.00% | 0.01% | 0.01% |
| 0.036 | floor 0.036, prior drawn per campaign (sections 1 to 4) | 0.09% | 0.00% | 0.01% | 0.01% |
| 0.050 | previous: floor 0.02, prior 0.0100 (the audited configuration) | 3.26% | 0.05% | 0.84% | 2.51% |
| 0.050 | shipped: floor 0.036, prior 0.0100 (the task's realized prior) | 0.89% | 0.01% | 0.14% | 0.44% |
| 0.050 | floor 0.036, prior drawn per campaign (sections 1 to 4) | 0.67% | 0.01% | 0.09% | 0.29% |
| 0.070 | previous: floor 0.02, prior 0.0100 (the audited configuration) | 4.23% | 0.10% | 1.47% | 4.47% |
| 0.070 | shipped: floor 0.036, prior 0.0100 (the task's realized prior) | 2.39% | 0.05% | 0.80% | 2.18% |
| 0.070 | floor 0.036, prior drawn per campaign (sections 1 to 4) | 1.94% | 0.04% | 0.50% | 1.73% |

| true improvement theta (sigma_d 0.036) | promoted: previous (floor 0.02) | promoted: shipped (floor 0.036, realized prior) | promoted: floor 0.036, drawn prior |
|---|---|---|---|
| 0.030 | 4.70% | 0.69% | 0.56% |
| 0.040 | 13.2% | 3.54% | 2.67% |
| 0.050 | 27.6% | 12.4% | 9.12% |
| 0.060 | 44.8% | 28.0% | 21.3% |
| 0.075 | 68.2% | 56.5% | 47.0% |
| 0.100 | 89.3% | 88.2% | 80.3% |
| 0.125 | 97.2% | 97.1% | 94.5% |

The floor trades sensitivity for protection: the minimum effect that can be promoted rises, and the false promotion rate of noisy candidates falls. The real-pipeline numbers in the next section were measured under the previous configuration.

## 6. Nulls drawn from the real grader (law-discovery, in process)

Sections 1 to 5 draw independent normal noise. This section checks the model against candidates that go through the task's own data module, its baseline model and the trusted grader's own row selection and RMSE, in process (no sandbox). A null here is the baseline's straight line fitted on a random share of the seed's training sample, with a fresh random subset for every candidate and every seed. It does not improve the population loss (the mean improvement column shows it is slightly worse), but its per-seed score moves with the subset, so its paired differences are noisy, and they are not symmetric (skewness column): with only 3 dev and 5 sealed seeds a skewed sample makes one tail of a t-test heavier than the normal model assumes. The dev and the sealed draws of one candidate are independent, as in the model. 20,000 candidates per family, cut into runs of 16 and 48 candidates with the cap of 6 sealed queries; the engine's prior is the real baseline's std on the dev seeds, 0.0100. The families are this script's own re-creation of the audit's kind of null, not the audit's candidates. 'model' rows are section 2's model at the same paired noise sd (independent normal draws, same prior).

| null family | paired noise sd | mean improvement | skewness | configuration | per candidate: dev gate | both gates | per run, 16 experiments | per run, 48 experiments |
|---|---|---|---|---|---|---|---|---|
| bag 30% | 0.115 | -0.0074 | -0.51 | previous floor 0.02, real | 5.83% | 0.14% | 2.32% (29 of 1,250) | 6.73% (28 of 416) |
| bag 30% | 0.115 | -0.0074 | -0.51 | shipped floor 0.036, real | 4.81% | 0.12% | 1.92% (24 of 1,250) | 5.77% (24 of 416) |
| bag 30% | 0.115 | n/a | n/a | shipped floor 0.036, model | 4.07% | 0.10% | 1.53% | 4.72% |
| bag 50% | 0.074 | -0.0033 | -0.41 | previous floor 0.02, real | 4.83% | 0.14% | 2.24% (28 of 1,250) | 6.73% (28 of 416) |
| bag 50% | 0.074 | -0.0033 | -0.41 | shipped floor 0.036, real | 2.80% | 0.07% | 1.12% (14 of 1,250) | 3.37% (14 of 416) |
| bag 50% | 0.074 | n/a | n/a | shipped floor 0.036, model | 2.63% | 0.05% | 0.80% | 2.46% |
| bag 70% | 0.048 | -0.0016 | -0.44 | previous floor 0.02, real | 3.57% | 0.07% | 1.12% (14 of 1,250) | 3.12% (13 of 416) |
| bag 70% | 0.048 | -0.0016 | -0.44 | shipped floor 0.036, real | 0.62% | 0.01% | 0.08% (1 of 1,250) | 0.24% (1 of 416) |
| bag 70% | 0.048 | n/a | n/a | shipped floor 0.036, model | 0.76% | 0.01% | 0.09% | 0.29% |

In the noisiest family (bag 30%, paired noise sd 0.115, above the floor 0.036) the runs that end with a promoted null at the shipped floor are 1.92% (24 of 1,250) of the 16-experiment runs and 5.77% (24 of 416) of the 48-experiment runs; the model at the same noise gives 1.53% and 4.72%. The model and the real-grader nulls agree within sampling error for the 48-experiment runs (95% interval of the real rate 3.91% to 8.44%). A floor can only help when the candidate's own sample std of its paired differences (from only 3 seeds) is below it, and the noisier the family the less often that happens: compare the previous and the shipped rows of each family. For the noisiest family the dev gate is nearly the plain 3-seed t-test, and what protects a run is the sealed stage and its cap. This is a model of bagged straight-line fits, not of an arbitrary researcher.

<!-- measured-on-the-real-pipeline:begin -->
## Measured on the real pipeline (independent audit, not produced by this script)

Everything above is a model. This section is a static record of what an independent statistics audit measured on the real
pipeline (the real sandbox, the real trusted grader, the engine's own decision code) for the law-discovery task. It was
produced with a throwaway harness that is **not part of this repository**, so the script cannot reproduce these numbers and the
tests do not check them. The audit ran **before** `sigma_floor` was calibrated: its configuration was `noise_margin` 0.035 and
`sigma_floor` 0.02, with the engine's `sigma_prior` (the baseline's own per-seed std on the 3 dev seeds) equal to 0.0100 on the
real task: below the floor, so the prior was inert, and the floor was itself far below the true baseline std (about 0.036 to
0.040). For a candidate with small paired noise the dev margin therefore collapsed to `noise_margin` 0.035; for a noisy one it
was the candidate's own `t * s / sqrt(3)`.

**The nulls.** Candidate families that do not truly improve on the baseline: bagged fits (bag30, bag50, bag70) and a weighted
fit (wls1); the names are the audit's labels. Their training subset or weights are seeded by the training data, so they differ
from seed to seed, and their effect on the population loss is within about +-0.005 of zero. That makes them valid nulls (no
real improvement) whose paired per-seed noise is nevertheless large: standard deviation 0.03 to 0.064, comparable to or larger
than the baseline's own 0.036.

**What was measured (old configuration, `sigma_floor` 0.02).**

| measurement | value |
|---|---|
| nulls run through the real sandbox | 190 |
| dev gate passed | 4.2% |
| of those, passed the sealed gate | 1.1% |
| nulls promoted | 0 |
| in-process replica of the pipeline, nulls | 11,000, with 0 mismatches against the sandbox's decisions |
| bag50, per candidate (of 3,000): dev gate / sealed gate / both | 5.2% / 2.4% / 7 of 3,000 |
| end-to-end false promotion, per candidate | about 0.1% to 0.3% |
| end-to-end false promotion, per 16-experiment run | wls1 0.5%, bag70 1.6%, bag50 3.7%, bag30 4.8% |
| end-to-end false promotion, per 48-experiment run | 5% to 12% |
| baseline `sigma_prior` (3 dev seeds), 5% / 50% / 95% quantiles over resampled seed sets | 0.0088 / 0.0304 / 0.0639 |

**Power (old configuration).** Promoted 100% of the time: the quadratic model (+0.73 RMSE improvement) and the damped
oscillator (+1.56). For graded effects: 0.029 RMSE improvement promoted 0%, 0.039 30%, 0.049 46%, 0.061 64%, 0.074 75%, 0.124 97%.

**How it compares with the model.** In section 5 the model's rows for the previous configuration give the same order of
magnitude for the dev gate (a few percent per candidate) but a lower "both gates" rate than the audit measured for bag50
(7 of 3,000). The real nulls are not independent normal draws, so for that configuration trust the audit's figures, not the
model's.

**What this means.** For the audit's noisy nulls the old dev gate was essentially a one-sided 5% t-test on 3 seeds, because the
prior (0.0100) and the floor (0.02) were both below their paired noise (0.03 to 0.064). Section 5 above shows in the model what the
calibrated floor changes. The shipped configuration now sets `sigma_floor` to the baseline's multi-seed std (0.036), which makes
the prior real; the audit's numbers describe the configuration it measured and are **not** a claim about the shipped one. The
power figures were measured with the old margins and are lower under the new ones (the dev margin for a candidate with small
paired noise rises from 0.035 to about 0.061, the sealed margin from 0.035 to about 0.045); the power table above is the model's
estimate of that shift. The real-pipeline false-promotion figures under the calibrated floor have **not** been re-measured by the
audit; section 6 above is this script's own in-process estimate with the real grader, not a re-measurement of the sandbox.
<!-- measured-on-the-real-pipeline:end -->

## Reading it

* A loop that promotes on the raw dev score is wrong almost every time it runs a few candidates.
* A classic t-test on three seeds still lets a null through at the textbook rate, and that rate compounds over candidates (the multiple-comparison problem every long auto-research session has).
* The dev gate alone is not a strong test for noisy candidates: the noisier a candidate's paired differences are compared with the floor, the closer the gate is to a 5% t-test on three seeds, and over many experiments the chance that some null reaches the sealed stage grows quickly (the per-run dev-gate columns). The sealed stage, with a hard cap on queries, is what keeps the per-run rate in the low percent for the model's noisiest rows.
* Section 6 checks the normal-noise rows against nulls drawn through the real grader. Read it for what a very noisy family costs at the shipped floor: the floor cannot help a candidate whose own paired noise is above it.
* The per-run false-promotion rate grows with the number of experiments (the tables give 16 and 48) and is not zero for noisy candidates. A longer run should be read with that in mind; the report states the expected number and an adaptive worst case for the sealed stage.
* Real runs differ from the model: seeds that share a training sample have correlated noise, a language model does not draw candidates independently, and a supervisor that sees dev scores can tune against them. Treat this as a check of the decision rule, not a forecast.

