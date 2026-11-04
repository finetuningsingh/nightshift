# Provenance (sanitized summary)

This file states only factual, non-private summary information about how
this repository was produced. Exact prompts, responses, session records,
and per-file hashes are kept privately by the maintainer and are not part
of this repository.

- The implementation, tests, and documentation in this repository were
  authored with Muse (`muse-wonder-full-1.0`, Muse Code `1.4.14`)
  through a coordinator, builder, independent reviewer, and fixer flow,
  and validated on CPU before publication.
- The repository history starts fresh and empty at its first commit. No
  earlier repository history, private specification, or operator material
  is included.
- Upstream methods and dependencies (CORAL, AIDE², Inspect AI, Axolotl)
  are cited, not vendored; see `THIRD_PARTY_NOTICES.md` for the pinned
  inventory and licenses.
- The bundled dataset in `nightshift/demo_data/` is synthetic toy data
  written for the demo. No benchmark test questions and no model weights
  are distributed.
- Validation status is stated honestly in the code: CPU fake runs are
  labelled fake everywhere, and real adapters are marked not yet validated
  live until a recorded live run exists.
