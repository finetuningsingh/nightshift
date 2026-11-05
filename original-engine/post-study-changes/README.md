# Post-study changes (not part of the recorded study)

`wonder-researchers-and-adversarial-challenge.patch` holds code written on 2026-11-04 **after** the recorded study, with Claude Code. None of it ran in the recorded study. The engine in `../` is left exactly as it ran, at source commit `3ed70d7`, so the recorded results stay tied to the code that produced them.

Apply it with `git apply post-study-changes/wonder-researchers-and-adversarial-challenge.patch` from `original-engine/`. It changes 14 files (+1933, -20):

| Private commit | Change |
|---|---|
| `600679f` | Researcher runtime records the requested, configured and served Muse model. A trial fails if Muse serves a different model, for example a silent fallback. `LAB_RESEARCHER_MODEL=muse-wonder-full-1.0` makes the researchers Muse Wonder Full. |
| `3d96799` | Bounded adversarial challenge in the outer loop, before lint and commit (`nightshift/lab/outer_challenge.py`). It has three parts, in order:<br>1. **Deterministic checker.** It reads the candidate's rendered training config and flags:<br>&nbsp;&nbsp;- a positive `max_steps` overriding `num_epochs` (axolotl 0.19.0)<br>&nbsp;&nbsp;- the training dose<br>&nbsp;&nbsp;- `MAX_TOKENS` against the model length<br>&nbsp;&nbsp;- warm starts from earlier adapters<br>&nbsp;&nbsp;- time boxes against the GPU account<br>&nbsp;&nbsp;- changes to evaluation, data-origin or data paths<br>2. **Independent critic.** One fresh `muse-wonder-full-1.0` session challenges the candidate's stated intent against the rendered config.<br>3. **One revision at most.** If blocking findings remain after it, the candidate is refused. Critic output is never training data. |
| `66e5793` | `scripts/outer_rehearsal.py`: runs the outer step once with the challenge, outside any lab ledger. |
| `b97e18b` | Hostkit trainer records the exact training dose (rows, steps, examples, epochs) from the actual training file before the optimizer starts. |
| `a23a73e` | The dose is stated with its bound (at most N steps, or at least N epochs, or unresolved) when only an upper bound on rows is known. The full private-data denylist is checked before every critic and revision call, with no send on a hit. |
| `d931ac3` | Three trusted-file tests wait 20 ms after each snapshot, so a create-and-delete inside one kernel timestamp tick is not missed. |

## Checks

- The patch was applied to a fresh clone of this repository, in an ax41 docker container (no network, read-only root, uid 4242).
- The affected and related test modules all pass: 690 tests, 11 skipped. The modules are outer challenge, Muse CLI proposer, CORAL runtime, lab CLI, hostkit, outer loop, contracts, sandbox, grading and isolation modes.
- The live use so far is one rehearsal of the outer step, on 2026-11-04, using public context from the recorded study:
  - The Wonder proposer and the Wonder critic were both served `muse-wonder-full-1.0`.
  - The proposal shortened the generation time box from 2700 s to 600 s.
  - The checker found no hard findings and two soft ones. The critic accepted it, so no revision ran.
- The revision and refusal paths are covered by unit tests only. No experiment result exists for this code yet.
