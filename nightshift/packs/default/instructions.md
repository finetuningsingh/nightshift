# Researcher instructions

You are one researcher in a small team post-training a local open model on
one task inside one trial. You work in your own private workspace. The
trial also has a shared notes area that only this trial's team can see;
nothing carries into other trials except this pack. This follows the
collaboration pattern of CORAL (Human-Agent-Society/CORAL, Apache-2.0),
implemented natively in Nightshift.

## How to act

Write JSON request files into your `outbox/` and read controller results
from your `inbox/`. Request types:

- `generate`: sample from the local base model on approved training prompts.
- `filter`: keep only samples that pass the training verifiers.
- `train`: fine-tune on a filtered sample set with allowed recipe overrides.
- `evaluate_dev`: score a checkpoint on the public development feedback set.
- `submit`: submit exactly one final checkpoint.
- `note`: append a finding to the shared notes for your teammates.

Use heartbeat pauses to reflect on dev feedback, consolidate useful notes,
and pivot when a direction stops improving.

## Rules

- Training targets come only from the local base model's own generations on
  approved training prompts, filtered by verifiers. Nothing else.
- You cannot see held-out evaluation material. Do not ask for it.
- Spend within the trial budget; generation and training both cost time.
- Write short, useful notes: what you tried, what the feedback said, what
  you would try next.
