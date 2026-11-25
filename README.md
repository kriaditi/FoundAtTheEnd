# Found at the End

**Primacy bias in multimodal RAG readers is learned from where training evidence sits.**

Aditi Kumari (ak2877), Anusha Ramachandrareddy (ar2664), Olamide Oke (ooo38). CS 5787 Deep Learning, Cornell Tech, Fall 2026.

Liu et al. (EMNLP 2026, "Lost at the End") showed that a multimodal RAG reader answers much worse when the gold
passage sits last among k retrieved passages, and attributed it to evidence-first instruction-tuning data. We test
that claim with controlled LoRA fine-tuning of Qwen2.5-VL-7B-Instruct on 20,000 retrieval contexts that share
questions, passages and seeds and differ only in where the gold passage sits.

## Headline results (InfoSeek, k = 10, n = 495, three-seed means, paired bootstrap 95% CIs)

| Reader | Gold first | Gold middle | Gold last | First − last gap |
|---|---|---|---|---|
| Frozen (our reproduction of the paper) | 45.05 | 32.53 | 29.49 | 15.56 [11.72, 19.60] |
| Standard SFT (natural retrieval order) | 54.68 | 55.22 | 53.94 | 0.74 [−0.81, 2.29] |
| Balanced SFT (gold slot uniform) | 54.07 | 56.03 | 54.61 | −0.54 [−2.15, 1.08] |
| Gold-first control (gold always slot 1) | 69.63 | 14.14 | 14.21 | 55.42 [51.38, 59.60] |
| Mixed standard (10k InfoSeek + 10k E-VQA) | 54.48 | 55.96 | 55.02 | −0.54 [−2.29, 1.21] |
| Mixed balanced | 54.88 | 56.43 | 54.68 | 0.20 [−1.55, 2.09] |
| Isolated reading (training-free) | 31.11 | 31.11 | 31.11 | 0 by construction |

* **The bias is learned from evidence position.** Fine-tuning with the gold passage always first gives a 55-point
  gap (3.6× the frozen reader's); that reader gains nothing from a gold passage in slot 10 (gold effect
  −0.07 [−0.54, 0.40]) and copies the slot-1 passage in 70% of its gold-last errors.
* **Ordinary fine-tuning removes it in-domain; balancing adds nothing there.** Balanced − standard: −1.28
  [−3.16, 0.61] points in gap, within ±1 point at every position; the result holds at k = 20.
* **Whether balancing is needed depends on the retriever.** The residual gap falls monotonically with the share of
  training contexts that place gold last: 0% → 55.4, 1.8% (E-VQA natural order) → 12.8, 6.0% (InfoSeek natural
  order) → 0.7, 10% (balanced) → 0 or a slight recency lean.
* **Transfer and repair.** InfoSeek-only adapters lose about 37 points on Encyclopedic-VQA; mixing in 10,000
  E-VQA contexts gives 75.1% with gold first (frozen: 64.4%) at no cost on InfoSeek. Mixed balanced removes the
  remaining 12.8-point E-VQA gap.
* **Evidence use and end-to-end gain.** Fine-tuned readers gain 27.6 points from a gold passage in slot 10 and
  11.6 points on the retriever's unmodified top 10.


### Encyclopedic-VQA transfer (k = 10, n = 500, one-way canonical matcher)

| Reader | Gold first | Gold middle | Gold last | Gap [95% CI] | Answer words |
|---|---|---|---|---|---|
| Frozen reader | 64.40 | 39.00 | 35.20 | 29.20 [25.00, 33.60] | 2.43 |
| Standard SFT | 27.67 | 25.60 | 24.67 | 3.00 [1.13, 4.87] | 1.47 |
| Balanced SFT | 27.73 | 27.07 | 26.60 | 1.13 [-0.53, 2.80] | 1.46 |
| Gold-first control | 38.00 | 11.47 | 11.93 | 26.07 [22.33, 29.73] | 1.49 |
| Mixed standard | 75.13 | 68.47 | 62.33 | 12.80 [10.13, 15.47] | 1.75 |
| Mixed balanced | 66.20 | 71.07 | 70.60 | -4.40 [-6.53, -2.40] | 1.77 |

### Counterfactual and natural-order evaluation (InfoSeek, same 495 questions)

Gold effect = accuracy with gold at slot 10 minus accuracy with the gold passage removed (9 distractors only). Natural = the retriever's real top 10, nothing inserted (gold present for 256 questions).

| Reader | No passages | Gold removed | Gold at slot 10 | Gold effect [95% CI] | Natural top-10 |
|---|---|---|---|---|---|
| Frozen reader | 24.44 | 16.77 | 29.49 | 12.73 [9.49, 15.96] | 29.49 |
| Standard SFT | 29.36 | 26.33 | 53.94 | 27.61 [23.97, 31.31] | 41.08 |
| Balanced SFT | 30.91 | 26.53 | 54.61 | 28.08 [24.58, 31.72] | 41.14 |
| Gold-first control | 30.51 | 14.28 | 14.21 | -0.07 [-0.54, 0.40] | 24.85 |
| Mixed standard | 30.71 | 26.20 | 55.02 | 28.82 [25.05, 32.66] | 41.35 |
| Mixed balanced | 30.98 | 26.87 | 54.68 | 27.81 [24.18, 31.45] | 41.14 |

### Late-evidence exposure vs residual gap

| Training condition (test-domain half) | Test | Gold in slot 10 (%) | Gap [95% CI] |
|---|---|---|---|
| Gold-first control (InfoSeek) | InfoSeek | 0.0 | 55.42 [51.38, 59.60] |
| Mixed standard (E-VQA natural order) | E-VQA | 1.8 | 12.80 [10.13, 15.47] |
| Standard SFT (InfoSeek natural order) | InfoSeek | 6.0 | 0.74 [-0.81, 2.29] |
| Mixed standard (InfoSeek half, natural order) | InfoSeek | 6.1 | -0.54 [-2.29, 1.21] |
| Balanced SFT (InfoSeek) | InfoSeek | 10.0 | -0.54 [-2.15, 1.08] |
| Mixed balanced (E-VQA) | E-VQA | 10.0 | -4.40 [-6.53, -2.40] |
| Mixed balanced (InfoSeek) | InfoSeek | 10.0 | 0.20 [-1.55, 2.09] |

Full tables: `results/COMPARISON.md`, `results/COMPARISON_EVQA.md`, `results/COUNTERFACTUAL.md` and the CSVs next
to them. Every per-question prediction for every model and condition is in `artifacts/predictions_unicorn.tgz`.

## Conditions

All conditions use the same 20,000 training questions (M2KR InfoSeek train split, leakage-filtered against the
evaluation set by entity, image and gold passage), the same frozen PreFLMR top-10 passages with exactly one
positive, and the same LoRA recipe (r = 64, α = 128, dropout 0.05, 196 language-model modules, 161.5M params,
1 epoch = 1,250 steps, effective batch 16, lr 1e-4 cosine, bf16). Seeds 0, 1, 2 each.

| Condition | Gold slot in training | Built by |
|---|---|---|
| `standard` | natural retrieval rank (slot 1: 25.4%, slot 10: 6.0%) | `02_build_data.sh` |
| `balanced_s<S>` | seed-locked stratified permutation, exactly 2,000 per slot | `02_build_data.sh`, `02b_extend_data.sh` |
| `goldfirst` | slot 1 in every context (control) | `02b_extend_data.sh` |
| `mixed_standard` | 10,000 InfoSeek + 10,000 E-VQA contexts, natural order | `02d_build_mixed.sh` |
| `mixed_balanced_s<S>` | same mixed contexts, balanced | `02d_build_mixed.sh` |

Evaluation is the authors' unmodified harness: InfoSeek P17 (k ∈ {1, 3, 5, 10, 20}, gold first/middle/last,
official scorer), Encyclopedic-VQA P20 (n = 500, one-way canonical matcher), plus our counterfactual conditions
(no passages, gold removed, natural top-10) and the isolated-reading baseline.

## Layout

```
scripts/config.sh          all paths, Slurm and experiment settings (edit here only)
scripts/lib.sh             logging, job submission, status markers
scripts/00_setup.sh        stage 0: authors' code (pinned commit 02bd142) + env + models + data
scripts/01_reproduce.sh    stage 1: reproduction of the paper's P17 (= frozen reader baseline)
scripts/02_build_data.sh   stage 2: standard + balanced training data (20,000 + 500 dev)
scripts/02a_inspect_data.sh    text-only inspection of the InfoSeek train files
scripts/02b_extend_data.sh     goldfirst control + extra balanced seeds (CPU, seconds)
scripts/02c_inspect_evqa.sh    text-only inspection of the E-VQA train files
scripts/02d_build_mixed.sh     mixed InfoSeek + E-VQA data (5,000-token cap on E-VQA contexts)
scripts/03_train.sh        stage 3: LoRA SFT, one Slurm job per (condition, seed); resumable
scripts/04_evaluate.sh     stage 4: P17 evaluation of every adapter, tables and plots
scripts/05_isolated.sh     stage 5: isolated-reading baseline
scripts/06_evqa.sh         stage 6: E-VQA transfer for the frozen reader and every adapter
scripts/07_counterfactual.sh   stage 7: counterfactual and natural-order evaluation
scripts/slurm/*.sub        the Slurm jobs the stage scripts submit (mail on END/FAIL)
lae/                       Python modules (bash only orchestrates):
  recover_subset.py        rebuilds the unreleased 496-question InfoSeek subset from the authors' result files
  patch_authors.py         two-line patch of the authors' runner (output dir, subset-size assert)
  build_train_data.py, extend_train_data.py, build_mixed_data.py, inspect_*.py
  train_lora.py            LoRA SFT (answer-only loss, prompt identical to evaluation)
  isolated_reading.py, evqa_transfer.py, counterfactual.py
  compare.py, compare_reproduction.py, evqa_diagnose.py   tables, bootstrap contrasts, plots
data/train/final/, data/train_mixed/final/   build_stats.json and manifests (the contexts themselves are rebuilt)
results/                   CSVs, Markdown tables, plots
artifacts/                 predictions and configs for every run
third_party/, runs/, logs/, status/, hf_cache/   created by the scripts (not tracked)
```

## Reproduce

On a Slurm cluster (we used Cornell's Unicorn cluster, one RTX 6000 Ada 48 GB per job; each training run takes
7.4 to 8.8 h, all 15 runs 122.8 h). Edit `scripts/config.sh` for your partition, GRES and paths, then from the
repo root on a login node:

```bash
bash scripts/00_setup.sh            # CPU job: uv env + ~35 GB models + InfoSeek/E-VQA data
bash scripts/01_reproduce.sh        # frozen reader, P17 protocol -> results/paper_reproduction.csv
bash scripts/02_build_data.sh       # standard + balanced contexts (GPU retrieval, ~7 h)
bash scripts/02b_extend_data.sh     # goldfirst + balanced seeds 1, 2
bash scripts/02d_build_mixed.sh     # mixed InfoSeek + E-VQA contexts (~15 min)
bash scripts/03_train.sh            # 15 training jobs (TRAIN_SEEDS, GOLDFIRST_SEEDS, MIXED_SEEDS in config.sh)
bash scripts/04_evaluate.sh         # P17 on every adapter, then tables
bash scripts/05_isolated.sh         # isolated-reading baseline
bash scripts/06_evqa.sh             # E-VQA transfer
bash scripts/07_counterfactual.sh   # counterfactual + natural order, then results/COUNTERFACTUAL.md
```

Every script takes `--status` and is safe to rerun: finished stages are skipped (`status/<stage>.ok`), queued
jobs are not submitted twice, training resumes from the last checkpoint after preemption, and an existing
results CSV with different content is never overwritten (a timestamped copy is written instead). Automated
checks on every data file verify identical question order and passage sets across conditions, gold at its
recorded slot, exactly one positive per context, unchanged distractor order and exact slot uniformity in the
balanced files.

## Deviations from the authors' release (all recorded)

1. `subset500_v2_qids.json` is not in the authors' repo. `lae/recover_subset.py` rebuilds it from their
   committed P22/P23 summaries: the 496 questions P17 reports on. On our GPU, retrieval scores differ at the
   fourth decimal from the authors' H100 run; one question drops to 17 distractors and is excluded by their own
   rule, so n = 495 (`INFOSEEK_N_EXPECTED=496`, `INFOSEEK_N_KEPT=495` in `config.sh`).
2. `lae/patch_authors.py` changes two lines of their P17 runner: the output directory and the subset-size
   assert. Retrieval, gold selection, prompt, decoding and scoring are untouched. The diff is saved next to each
   run as `authors_patch.diff`.
3. Adapters are merged into the base weights before evaluation and loaded by the authors' unmodified reader
   code, so inference cost is identical to the frozen reader.

## Reference

Liu et al., "Lost at the End: Primacy Bias in Multimodal Retrieval-Augmented Question Answering", EMNLP 2026.
Authors' code: https://github.com/WeMWish/lost_at_the_end_code (commit 02bd142).

