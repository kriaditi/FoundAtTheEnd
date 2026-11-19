# Counterfactual and natural-order evaluation (generated 2026-11-19 10:49)

InfoSeek, same 495 questions and official scorer as the P17 cells. Accuracy in %.
none = no passages; gold_removed = the 9 P17 distractors only (k = 9); natural = the retriever's real top 10, nothing inserted (gold is in it for 256 of 495 questions).
For reference, frozen k = 10 gold first / last: 45.05 / 29.49.

| Model | none | gold removed | natural top-10 | natural, gold in top-10 | natural, gold absent | Status |
|---|---|---|---|---|---|---|
| Frozen Qwen | 24.44 | 16.77 | 29.49 | 48.05 | 9.62 | measured |
| Standard LoRA SFT | 29.36 | 26.33 | 41.08 | 61.59 | 19.11 | mean of 3 seeds |
| Balanced LoRA SFT | 30.91 | 26.53 | 41.14 | 62.63 | 18.13 | mean of 3 seeds |
| Gold-first LoRA SFT (control) | 30.51 | 14.28 | 24.85 | 39.58 | 9.07 | mean of 3 seeds |

## Paired bootstrap (B = 10,000, seed 42), percentage points

| Contrast | Estimate | 95% CI | CI excludes 0 |
|---|---|---|---|
| Frozen Qwen: gold_removed - none (evidence from distractors alone) | -7.68 | [-11.52, -3.84] | yes |
| Standard LoRA SFT: gold_removed - none (evidence from distractors alone) | -3.03 | [-6.46, +0.40] | no |
| Standard LoRA SFT vs Frozen: none | +4.92 | [+1.68, +8.22] | yes |
| Standard LoRA SFT vs Frozen: gold_removed | +9.56 | [+6.40, +12.73] | yes |
| Standard LoRA SFT vs Frozen: natural | +11.58 | [+7.81, +15.42] | yes |
| Balanced LoRA SFT: gold_removed - none (evidence from distractors alone) | -4.38 | [-7.88, -1.01] | yes |
| Balanced LoRA SFT vs Frozen: none | +6.46 | [+3.30, +9.70] | yes |
| Balanced LoRA SFT vs Frozen: gold_removed | +9.76 | [+6.53, +13.06] | yes |
| Balanced LoRA SFT vs Frozen: natural | +11.65 | [+7.88, +15.35] | yes |
| Gold-first LoRA SFT (control): gold_removed - none (evidence from distractors alone) | -16.23 | [-19.87, -12.66] | yes |
| Gold-first LoRA SFT (control) vs Frozen: none | +6.06 | [+2.76, +9.36] | yes |
| Gold-first LoRA SFT (control) vs Frozen: gold_removed | -2.49 | [-5.59, +0.61] | no |
| Gold-first LoRA SFT (control) vs Frozen: natural | -4.65 | [-8.62, -0.74] | yes |
| Balanced LoRA SFT vs Standard LoRA SFT: none | +1.55 | [+0.34, +2.83] | yes |
| Balanced LoRA SFT vs Standard LoRA SFT: gold_removed | +0.20 | [-1.14, +1.55] | no |
| Balanced LoRA SFT vs Standard LoRA SFT: natural | +0.07 | [-1.21, +1.41] | no |
