# `seeds_sample/` — MMCode seeds (smoke-test only)

Slimmed-down subsamples of the upstream MMCode dataset for end-to-end pipeline
demos:

| File | Origin | n | Notes |
|---|---|---|---|
| `mmcode_full.jsonl` | upstream `MMCode/mmcode_full.jsonl` | 30 | filtered to records with `images`, `solutions`, and `input_output`; **the inlined base64 images have been dropped** to keep this sample lightweight (the generator only uses `image_tags`). |
| `mmcode_test.jsonl` | upstream `MMCode/mmcode_test.jsonl` | 8 | same slimming as above |

When you operate on the full corpus, drop the original full / test JSONLs
into the parent `mmcode-pipeline/` directory — the loader prefers files at
the pipeline root over the bundled samples.

## Licensing

The MMCode dataset is © its authors and licensed for research; see the
[upstream dataset card](https://huggingface.co/datasets/likaixin/MMCode) for
the exact terms.  This subsample is redistributed under the same licence.
