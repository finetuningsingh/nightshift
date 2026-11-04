# Muse authoring provenance (public extract)

These folders record how Muse Code 1.4.14 with `muse-wonder-full-1.0` authored the researcher starter pack (`starter-author/`) and the submission pack (`pack-writer/`).

Kept here: `steps.jsonl` (each step's exact `muse exec` command line, model, CLI version, start and finish times, exit code, and the sha256 of its prompt file and event log), `muse_authored_files*.sha256` (hashes of the files exactly as Muse wrote them), the driver script and the coordinator's recorded post-import edit.

Not published: the raw step prompts, the facts files given to Muse and the full Muse event logs. They contain coordination history and are kept privately; their sha256 values in `steps.jsonl` let anyone holding them check them.

## Differences between the published pack and Muse's output

- `REPORT.md`: one quotation of the owner's private message was removed from the budget sentence (it now ends "...across all roots)"). Nothing else in it changed.
- `READINESS.md`, `SUBMISSION_FORM.md`, `PROVIDER_TEST_PLAN.md` and the video folder are not published. They are owner checklists, form drafts and production material. Links to them from other pack files do not resolve here.
- The coordinator's "14:05 ET" to "13:48 ET" time correction is described in `pack-writer/COORDINATOR_EDITS.md`.

Because of these differences, the hashes in `muse_authored_files*.sha256` match Muse's original output, which is kept privately, and not the published files.
