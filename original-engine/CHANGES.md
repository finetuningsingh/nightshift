# Changes after the recorded study

The recorded study ran on public commit `8bcbf05` (private source commit `3ed70d7`). Check out `8bcbf05` to see that code exactly. The following changes were made on 2026-11-04 afterwards, with Claude Code, and are in this tree:

| Private commit | Change |
|---|---|
| `600679f` | The researcher runtime records the requested, configured and served Muse model, and fails the trial if Muse serves a different model. |
| `3d96799` | Bounded adversarial challenge before lint and commit (`nightshift/lab/outer_challenge.py`). It runs in this order:<br>1. Deterministic checker of the candidate's rendered training config. It flags a positive `max_steps` overriding `num_epochs` (axolotl 0.19.0), the training dose, `MAX_TOKENS` against the model length, warm starts, time boxes against the GPU account, and changes to evaluation, data-origin or data paths.<br>2. One independent `muse-wonder-full-1.0` critic.<br>3. At most one revision; the candidate is refused if blocking findings remain.<br>Critic output is never training data. |
| `66e5793` | `scripts/outer_rehearsal.py`: one outer step with the challenge, outside any lab ledger. |
| `b97e18b` | The hostkit trainer records the exact training dose (rows, steps, examples, epochs) from the actual training file before the optimizer starts. |
| `a23a73e` | The dose is stated with its bound when only an upper bound on rows is known. The full private-data denylist is checked before every critic and revision call, with no send on a hit. |
| `d931ac3` | Three trusted-file tests wait 20 ms after each snapshot (see the limitation below). |
| this commit | `scripts/two_loop/run_lab.py` defaults to Muse Wonder Full researchers and the challenge on. |

## Checks of these changes

- The changes were applied to a fresh clone of this repository and tested in an ax41 docker container with no network, a read-only root filesystem and uid 4242. The affected and related test modules passed: 690 tests, 11 skipped. The modules are outer challenge, Muse CLI proposer, CORAL runtime, lab CLI, hostkit, outer loop, contracts, sandbox, grading and isolation modes.
- That is the scope of the check. The whole suite was not rerun on this tree.
- Live use so far: one rehearsal of the outer step on public context from the recorded study.
  - The Wonder proposer and the Wonder critic were both served `muse-wonder-full-1.0`.
  - The proposal shortened the generation time box from 2700 s to 600 s.
  - The checker reported no hard findings and two soft ones. The critic accepted, so no revision ran.
  - The revision and refusal paths are covered by unit tests only.

## Known limitation: trusted-file change detection and timestamp resolution

The untrusted-run sandbox detects changes to trusted files by comparing file metadata snapshots.
- On Linux 6.8, file timestamps advance only once per kernel tick, about 1 ms.
- A file created and deleted inside one tick, in the same directory as the snapshot, can leave the directory's metadata unchanged and go undetected. The three tests fixed in `d931ac3` hit this; they now wait 20 ms.
- **That wait makes the tests reliable. It does not remove the limitation.** Treat detection of sub-millisecond create-and-delete inside a trusted directory as unverified.
- Not yet examined: whether content hashes cover the same case.

A separate container finding: the sandbox's per-user process limit counts only threads visible in the current process namespace. Inside a container that shares a host uid, the limit can be far too low and every `fork` fails. Running as an unused uid avoids it. The code is unchanged.

## The attempt to run the new code

- On 2026-11-04 from 17:47 to 18:47 ET, a small matched experiment ran with Muse Wonder Full researchers.
  - It got no GPU: 98 Thunder refusals, "out of GPU capacity".
  - The trial ended with no submission, and no score exists.
- A later capacity probe at 19:18 ET was also refused.
- The two L40 48 GB cards listed as available were not used. This engine serves Glimmer 30B on one GPU, with no tensor-parallel option, and there was not enough time left to add and test one.
