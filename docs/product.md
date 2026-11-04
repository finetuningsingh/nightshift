# Nightshift — product description

**One line.** Research agents that post-train a small open model overnight,
and an outer agent that improves the researchers.

**What it does.** Nightshift runs a two-level loop. In the inner loop, a
small team of researcher agents collaborates inside one trial — isolated
private workspaces plus shared notes that exist only for that trial — to
generate training samples from a local open model on approved prompts,
filter them with programmatic verifiers, fine-tune with QLoRA, and submit a
checkpoint for public development-set feedback. In the outer loop, a
reviser agent studies permitted public traces from finished trials and
edits the researcher pack itself (instructions, tools, recipe, budget
habits). The controller freezes each candidate pack, screens it against
the incumbent on a private selection split, and finally compares the
original and improved fine-tuning procedures in fresh matched trials on an
untouched test split.

**Primary question.** Does Nightshift's improved supervised fine-tuning
(SFT) procedure produce better evaluated results than the original SFT
procedure, starting from the same Muse Glimmer checkpoint under matched
resources? Beating the untrained base model is context, not the goal, and
training loss alone is never treated as quality evidence.

**Why it is trustworthy.** A deterministic controller — not a model — owns
spending, private data, and scoring. Every GPU request is reserved against
an aggregate ceiling before any rental is created; leases, hard expiries,
and exact-owned cleanup bound the blast radius. Private splits never reach
an agent, checkpoints must be safetensors, failed evaluations are reported
as failures (never zeros), and every event lands in a hash-chained ledger
linked to immutable Inspect AI logs.

**What works today.** The full two-loop pipeline runs end to end on CPU
with labelled deterministic fakes and real Inspect scoring: two outer
rounds, frozen pack versions, a paired final comparison, and an
evidence-linked report. Configurable backends cover the Muse
(default researcher `muse-spark-1.3`, reviser `muse-wonder-full-1.0`) and
OpenAI-compatible APIs such as the Muse Model API and Baseten's Muse
Spark Open endpoint.

**What is not claimed.** No live GPU training or improvement result is
claimed by this repository yet. Real provider, generation, training, and
API adapters are marked not yet validated live until a recorded run
exercises them, and every fake record says so.

**Built on.** The CORAL collaboration pattern, WeCo's AIDE² outer method,
Inspect AI for evaluation, and Axolotl for QLoRA training. Open source
under the MIT License.
