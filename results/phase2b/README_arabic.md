# Why Arabic results live in a separate file

`arabic_noncomparable_diagnostics.csv` is **not** part of the CLCG heatmap
dataset, and should never be merged into `multilingual_ablation_table.csv`
or treated as comparable to the DE/FR/ES rows there — regardless of how
many Arabic briefs it ends up covering.

## The reason (not a sample-size issue)

MulBrandEval's CLCG metric assumes the generative model actually received
and attempted to render the prompt it was given. For Arabic, that
assumption doesn't hold: SD v1.5 and FLUX.1-dev's underlying CLIP text
encoder has minimal functional understanding of Arabic script. This was
confirmed independently of prompt length or token truncation — a short
Arabic test prompt well under the 77-token limit ("ساعة ذكية فضية" /
"silver smartwatch") still produced generic, unrelated content on both
models, tested directly on Replicate's playground outside the pipeline
entirely.

In practice this means Arabic-prompted images show near-total
prompt-conditioning failure, not a *gradient* of degraded compliance. A
CLCG(EN, AR) gap computed from this pipeline's node scores would measure
**text-encoder noise**, not brand-compliance degradation — the two are not
the same phenomenon, and reporting one as if it were the other would
overstate what the number means.

## What this file IS for

The DAG's per-node scores on the available Arabic images (however many
exist at the time — the pilot's 20, or a full 150-brief run if someone
extends it later) are useful as **quantitative diagnostic evidence** for
describing the failure mode itself: e.g. "Node 1 object presence collapsed
to X% across N Arabic images" is a concrete, citable number for a
cautionary case-study section, stronger than a purely qualitative
description ("images look like generic cityscapes instead of the
requested products").

## What this file is NOT for

- Do not compute CLCG(EN, AR) from these rows and report it alongside the
  DE/FR/ES heatmap cells as if it were the same kind of measurement.
- Do not merge these rows into `multilingual_ablation_table.csv`.
- Do not treat a larger sample size (e.g. a future full 150-brief AR run)
  as resolving this — the underlying encoder limitation is categorical,
  not something more data averages out.

## If you're extending this to a full Arabic run

The evaluation script already supports this without modification:

```
python -m phase2a_dag_pipeline.phase2b_evaluate --languages ar
```

This will pick up however many Arabic images exist on disk at
`data/generated_images/phase2b/{model}/ar/` and evaluate them, still
writing to this same isolated file. The isolation logic is scale-independent
by design — it stays separate whether it's 20 images or 300.