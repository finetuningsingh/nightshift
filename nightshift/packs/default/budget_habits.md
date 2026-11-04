# Budget habits

- Generate a small batch first; check verifier pass rates before scaling up.
- Prefer one careful training run over several runs on unfiltered samples.
- Evaluate on the public development set after each training run, not more
  often: every rental minute, including setup and failures, is charged.
- Leave headroom for a final evaluation and submission.
