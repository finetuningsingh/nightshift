# Starting notes (read first, then add your own below)

These notes come with the pack. They are part of the treatment: every trial of this pack starts with exactly this text.

## How a trial works
- You have a fixed budget (`glimmer budget`). Everything counts against it: your own model calls, GPU minutes and wall time.
- One GPU job = one recipe run on a fresh rented GPU. Idle time between jobs costs nothing, but every job pays for loading the model.
  So do not run many tiny jobs; test code paths on a very small job first, then run the real one.
- The score you see is the dev score: the fraction of public dev prompts on which every constraint passes. The decision is made on
  other prompts that you cannot see.

## What the starting recipe does (`recipe/`)
1. Serves the base model and samples several answers per training prompt.
2. Keeps the answers that pass every constraint of their prompt, using the same checker the dev score uses.
3. Fine-tunes the model on the kept answers (QLoRA, one epoch) and leaves the adapter for scoring.

It is a reference, not a ceiling. Known open questions: how many prompts and samples are worth their cost, whether harder prompts
(low pass rate) deserve more samples, whether keeping reasoning text helps or hurts, learning rate and epochs.

## Working habits that have paid off elsewhere
- Write the failure down. When a job fails, put the error signature (for example "out of memory", "adapter did not load", "checker
  import error") in the notes so the next job does not repeat it.
- Hold some dev prompts back from your own decisions. A change that helps only the prompts you looked at is not an improvement.
- Before you hand in, make sure `submission.json` is committed and names an adapter whose dev score you have read.

## Log
(add dated one-line entries here: what you ran, what it cost, what the dev score was)
