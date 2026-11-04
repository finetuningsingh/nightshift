# Third-party notices

Nightshift is newly authored for this repository. It depends on, and is
inspired by, the upstream work below. No upstream source is vendored as new
work; each dependency keeps its own license, recorded here at the pinned
revision or version Nightshift references.

| Component | Role in Nightshift | Pinned reference | License |
| --- | --- | --- | --- |
| CORAL (Human-Agent-Society/CORAL) | Inspiration for the inner collaboration pattern: isolated per-agent workspaces with trial-scoped shared notes. Implemented natively; no CORAL code is included. | Revision `0123dfb939b35228cf2c1fde224cd0561e727408` — <https://github.com/Human-Agent-Society/CORAL> | Apache License 2.0 (upstream) |
| AIDE² (WeCo) | The outer method: improve the researcher pack, screen on a private held-out split, and report only a final fresh comparison as evidence. Cited method; no hosted service is used. | "First evidence of recursive self-improvement" — <https://www.weco.ai/blog/first-evidence-of-recursive-self-improvement> | Published article (citation only) |
| Inspect AI (UK AI Safety Institute) | Evaluation harness. All scoring runs through Nightshift's Inspect task; fake mode runs Inspect for real against its `mockllm/model`. The exact installed version is recorded in every evaluation record. | `inspect-ai>=0.3.100,<0.4` (see `pyproject.toml`); the resolved version is logged per run | MIT License (upstream) |
| Axolotl | Real-mode QLoRA training. The pack holds a recipe template; the controller renders it with controller-owned keys forced and runs it as a provider job. Axolotl is never imported by the controller package. | Optional extra `axolotl==0.21.1` (see `pyproject.toml`) | Apache License 2.0 (upstream) |
| Muse | Agent runtime for the researcher and reviser roles in CLI mode (`muse exec --json`). | Model ids are operator configuration; defaults are recorded in `nightshift/config.py` | Meta product terms (runtime, not bundled) |
| tiktoken (transitive, via Inspect AI) | Token-count estimation inside Inspect. The CPU image pre-caches the `o200k_base` and `cl100k_base` encodings at build time so tests and the demo run offline. | Resolved with Inspect AI; see `docker/Dockerfile.cpu` | MIT License (upstream) |
| Python | Runtime (`>=3.11`; the CPU image uses `python:3.12-slim`). | See `pyproject.toml` and `docker/Dockerfile.cpu` | Python Software Foundation License |

## Data

- `nightshift/demo_data/` contains a tiny synthetic toy dataset written for
  the CPU demo. This repository does not bundle IFBench or any benchmark's
  test questions. Real runs take operator-supplied frozen manifests.
- The generation/training base model is operator-configured by exact
  repository id and revision. This repository distributes no model weights.

## How to check the pins

- Python dependencies: `pyproject.toml` (`[project]` and
  `[project.optional-dependencies]`).
- CORAL revision: cited in `nightshift/agents.py` and this file.
- Installed Inspect version at run time: the `inspect_version` field of every
  evaluation record and the ledger event that carries it.
