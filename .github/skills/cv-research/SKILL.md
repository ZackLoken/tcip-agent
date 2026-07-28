---
name: cv-research
description: "The research→implement→validate loop for a CV scientist. How to find a candidate technique in the academic literature, judge its fit to the dataset in hand, implement it against the Phase-3 toolkit (bespoke nn.Module + custom train(ctx) through the envelope), and — the hard rule — prove it beats a baseline on the measured phenotype before trusting it. Load when a metric has plateaued, a trait resists the current architecture, or you are tempted to reach for a new method you read about."
---

# CV Research

You are the CV scientist, not just the CV engineer. When the current toolkit plateaus on a trait,
the move is not to guess a new architecture from memory — it is to **read what the field has
established, port the piece that fits, and prove it earns its place**. This skill is that loop.
It has three parts, and the third is not optional.

```
RESEARCH (academic sources)  →  IMPLEMENT (Phase-3 toolkit)  →  VALIDATE (beats baseline on the measured phenotype)
```

Skipping straight to IMPLEMENT is how you adopt a fashionable method that helps a benchmark and
hurts your trait. Skipping VALIDATE is how a plausible technique becomes silent institutional truth
the next session inherits. Neither is allowed.

## 1. Research — academic sources only

Your web access is **governed and scoped to academic sources by the fence** (WebSearch to find
papers; WebFetch allowed only for a fixed set of academic hosts). Treat that as the floor, not the
ceiling: even where a fetch would technically succeed, prefer primary academic sources and stay out
of the open web. The allowed/preferred set:

- **arXiv** (`arxiv.org`) — preprints; the fastest path to method detail and often the reference
  implementation link.
- **Semantic Scholar** (`semanticscholar.org`) — citation graph, "cited by", influential-citation
  signal; use it to gauge whether a method is established or a one-off.
- **OpenReview** (`openreview.net`) — peer reviews and rebuttals; read the *reviews*, they surface
  the failure modes and the ablations that matter.
- **Papers With Code** (`paperswithcode.com`) — links method → code → benchmark; use it to find the
  canonical implementation and the datasets a method was actually validated on.
- **Proceedings**: `openaccess.thecvf.com` (CVPR/ICCV/WACV), `proceedings.mlr.press` (ICML/AISTATS),
  `aclanthology.org` (ACL family), `biorxiv.org` (bio preprints, useful for plant/phenotyping work).

Search discipline:

- **Search the problem, then the method** — start from the trait's difficulty ("small-object
  detection aerial", "few-shot fine-grained classification", "tiled inference tiny objects"), not
  from a method name you already like. You are looking for what the field does about *your* problem.
- **Read the ablation and the failure modes, not the headline number.** A method's reported gain is
  on its benchmark under its conditions. What transfers is the *mechanism* and the conditions it
  needs — dataset size, object scale, label density, compute.
- **Prefer established over novel.** A technique with many independent citations and a maintained
  reference implementation is a lower-risk port than last month's state-of-the-art. Progressive
  disclosure applies to methods too.
- **Capture provenance.** When you adopt a technique, record the source (title, venue, arXiv id) in
  the experiment lineage / retrospective, so the next session can trace *why* the model is shaped the
  way it is.

If a technique lives only behind a paywalled journal the fence can't reach, do not guess its
internals from a blog summary — file a `claude_reports` note describing the gap rather than
implementing a half-understood method.

## 2. Implement — against the Phase-3 toolkit

The platform is built so a technique from a paper is a **first-class citizen, not a fork**. You own
the model *and* the training loop — see `pipeline-design`'s "You own the model AND the training loop"
for the full contract; the seams you use here are:

- **A bespoke `nn.Module`.** Write the architecture from the paper (or the one primitive it changes)
  as a normal PyTorch module. Lazy-import torch inside the function body per the repo contract. Point
  `pipelines.model_build.build_model` at it via a `model_source` — an *importable* builder
  (`{"builder": "my_module:build_net", "builder_kwargs": {...}, "source_files": [...], "task": ...,
  "in_chans": ...}`), imported not `exec`'d, so the run reproduces from source.
- **The model contract is the only hard boundary.** `pipelines.model_contract` /
  `check_model_contract` require just that your module trains (finite-gradient loss) and emits
  inference output the library scorers consume. You don't call it by hand: `launch_training` smokes
  it automatically (`preflight_config(smoke=True)` builds the model + runs the contract at the
  *resolved* in_chans/num_classes/img_size before the thread spawns), so a broken builder fails the
  launch instead of wasting a run. From inside a custom loop, `ctx.check_contract()` self-proves at
  the same resolved dims; prove it learns cheaply first with `ctx.overfit_check()` (voluntary,
  non-gating). For a task outside the ones `build_dataset` routes, pass `sample_batch=` — an
  `(images, targets)` pair from your dataset — since no synthetic target shape is invented for it.
- **A custom `train(ctx)` when the technique needs one** (a new loss schedule, a two-stage curriculum,
  a contrastive pretext, EMA weights, a distillation loop). Point `training_source` at it. The
  `TrainContext` (`pipelines.training.envelope`) hands you the craft library — leakage-free loaders,
  `ctx.build_optimizer` / `ctx.build_scheduler` / `ctx.evaluate` / `ctx.set_seed`, the
  progressive-unfreeze primitive `ctx.apply_stage_freeze`, `ctx.tiled_dataset`, `ctx.calibrate`,
  and the correctness checks `ctx.check_contract` / `ctx.overfit_check` — plus the envelope-owned
  sinks `ctx.log_metrics`, `ctx.save_checkpoint`, `ctx.record_artifact`, `ctx.should_cancel`. Route
  metrics and checkpoints through those sinks and the run stays audited, immutably versioned, and
  provenance-snapshotted no matter what your loop does. `ctx.default_train()` is a convenience to
  call, extend, or replace.

  Registration needs one more explicit fact (K11): save under `"model_best"`/`"model_final"`, or
  call `ctx.set_final_weights(path)` yourself — otherwise a "completed" run with no discoverable
  weights is marked `failed` rather than registering a nonexistent path. Under `run_hpo`, call
  `ctx.report_objective(value)` to report trial progress for pruning if your loop's own metrics
  don't share the stock trainer's key names.

**Fit to the data in hand, don't transplant blind.** A paper's hyperparameters are for its dataset.
Derive the operating points from *your* data at runtime (CLAUDE.md: derive, don't pin) — anchor sizes
from your GT box distribution, norm choice from your real batch size, pyramid levels from your object
scale. A method that assumed 118k images and batch 64 will not behave the same on a few hundred tiles
at batch 2; adapt the mechanism to that reality or expect it to fail.

## 3. Validate — beats a baseline on the *measured phenotype*

This is the part that makes it science. **A technique is not adopted because a paper reports a gain,
because it is elegant, or because it improved a proxy metric. It is adopted only after you measure
that it beats the current baseline on the phenotype the breeder actually needs.** Implement-and-measure,
never adopt-on-faith.

The discipline:

1. **Fix a baseline first.** Run the current best model and record its score *as an immutable
   experiment*. You cannot claim an improvement without the number you improved on.
2. **Change one thing.** Introduce the researched technique as a new experiment against the same
   splits, same eval, same seed policy. Confounded comparisons prove nothing.
3. **Measure on the phenotype, not a surrogate.** A better mAP that does not move the *measured
   trait* (the count, the date, the dimensional measurement the breeder scores) is not an
   improvement for this platform — it is a benchmark artifact. Validate against a reference sized to
   the trait — GT annotations, or a breeder-confirmed sample of the model's own outputs
   (review-confirmation), not dense GT for every trait — the same bar every measurement faces (either
   reference passes the identical disjoint-split + count-bias gate). This is the
   **measurement-integrity invariant**
   (CLAUDE.md): a confident, precise, wrong phenotype is the worst thing this platform can produce,
   and a new technique gets no exemption from that bar.
4. **Keep it only if it wins, and say so honestly.** If it ties or loses, discard it and record why
   in a retrospective — a negative result is real knowledge and stops the next session re-trying it.
   If it wins, it becomes the new baseline; log the source and the delta in the experiment lineage.
5. **Never let a researched-but-unvalidated method harden into a definition.** Until it clears step 3
   it is provisional and labeled provisional — it must not become the default the next session reuses
   without seeing the evidence.

If validation is impossible because the phenotype itself can't yet be measured validly from pixels,
that is the finding — surface it with `claude_reports`. Do not manufacture a number so the new
method appears to work.

## The loop, in one line

Read the field → port the mechanism that fits your data → and let the measured phenotype, judged
against expert ground truth, decide whether it stays.
