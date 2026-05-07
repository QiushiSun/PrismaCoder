# `seeds_sample/` — cross-source seeds (smoke-test only)

This directory contains **small subsamples** drawn from the full upstream
training corpus.  They are bundled here purely so that `./run_pipeline.sh`
finishes end-to-end on a fresh checkout (small smoke run).

| File / Dir | Origin | n | Filter |
|---|---|---|---|
| `leetcode_native_records.jsonl` | LeetCode multimodal SFT | 20 | difficulty ∈ {Hard, Medium} |
| `leetcode_newgen_records.jsonl` | LC newgen (Plan-A') | 20 | difficulty == Hard |
| `mmcode_full_records.jsonl`     | MMCode_full extraction | 20 | difficulty ∈ {HARD, MEDIUM_HARD, VERY_HARD} |
| `competition_solved/codeforces_solved.jsonl` | CF solved by Claude | 8 | sandbox-verified at 100% test-pass |
| `competition_solved/atcoder_solved.jsonl`    | AtCoder solved | 8 | same |
| `competition_solved/aizu_solved.jsonl`       | Aizu solved    | 8 | same |
| `images/` and `competition_solved/images/`   | first images for each record | — | — |

## Licensing

Each upstream source retains its original licence:

* LeetCode data — covered by LeetCode's terms; we redistribute very small
  excerpts for research demonstration purposes only.
* CodeForces / AtCoder / Aizu — public-domain problem statements, with our
  Claude-generated solutions released under MIT.
* MMCode — see [the MMCode dataset card](https://huggingface.co/datasets/likaixin/MMCode);
  redistributed under the same terms.

When operating on your own corpus, point the pipeline at your full data via
the `CROSS_AUG_*` environment variables documented in the parent `README.md`.
