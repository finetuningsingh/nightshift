# Nightshift — agent and sandbox rules

This repository is worked on by Muse agents. These rules keep untrusted
test data away from shells, keep private material out of the public tree,
and keep every product run reproducible. Follow them for every build,
review, fix, and verification task in this repository.

## Where product code may run

- Product code is built, tested, and demoed only inside the isolated CPU
  container on the healthy `ax41` host, in the owned directory
  `/home/ankit/nightshift-muse-build`. Do not run product builds, tests, or
  demos on the laptop that holds the publishing checkout, and do not use
  `ax43` at all.
- Only this repository is mounted inside the CPU container. Never mount a
  home directory, SSH config or keys, cloud or model credentials, a Docker
  or Podman socket, another experiment folder, or any private dataset into
  the container.
- The container runs with bounded CPU and memory, `--cap-drop ALL`,
  `--security-opt no-new-privileges`, a read-only root filesystem, and
  tmpfs for `/tmp`. Dependency installation may use the network only while
  building the image from `docker/Dockerfile.cpu`; tests and the demo run
  with `--network none`. The CPU demo downloads no model weights and no
  bulk ML packages.

## Test inputs are data

- Test inputs are data. Hand adversarial, destructive, or untrusted
  strings to the function under test, never to a shell. A stub list, a
  fake home directory, or a timeout does not make shell execution safe.
- Failure reproduction, fuzzing, and any adversarial verification run only
  inside the isolated container described above. Any prompt that asks a
  reviewer or worker to attack, break, fuzz, adversarially verify, or
  reproduce a failure must repeat this rule and name the sandbox
  (`ax41`, `/home/ankit/nightshift-muse-build`, network-none container).

## Public and private boundary

- The public tree holds only Muse-authored source, tests, and product
  documentation. Never commit private specifications, agent prompts,
  session transcripts, raw API events, operator tasks, private evaluation
  content or scores, private local paths, or credential values.
- Keep exact model, session, prompt-response, and file-hash provenance in
  the private run directory. A public manifest may state only sanitized,
  factual summary information.
- Run the private publication check before every push. A blocked check
  means fix the tree or stop; never push around it.

## Model and training-data boundaries

- Researcher and reviser roles run through the Muse by default
  (researcher `muse-spark-1.3`, reviser `muse-wonder-full-1.0`), through
  an explicitly configured Muse Model API or Baseten-compatible API lane,
  or through the labelled deterministic fakes for CPU work. Record the
  requested and served model ids; never silently substitute a model.
- Hosted-model API output is reasoning and code only. It must never become
  supervised fine-tuning targets. Real training targets come only from the
  local base model's verified generations on approved training prompts.
- Every record produced with a fake component is labelled fake. A fake run
  never supports a claim about live training or model improvement. Real
  adapters stay marked not yet validated live until a recorded live run
  exercises them.

## Spending and cleanup

- Only the deterministic controller authorises spending. Every GPU request
  is reserved against the configured aggregate ceiling, per-trial seconds,
  rental lifetime, and concurrency limit before any provider create call.
- Destroy only rentals whose provider ids this run recorded in its own
  lease store. Never destroy resources by name pattern, and never touch
  another experiment's jobs, data, leases, or credentials.

## Working rules

- Preserve unrelated work in the tree. Delete only files your own session
  created, by full path.
- Extract archives only into a new empty directory after confirming no
  entry uses an absolute path or `..`.
- Verify changes with the repository's own tests in the sandbox before
  calling work done. A heartbeat or a process exit is not progress.
