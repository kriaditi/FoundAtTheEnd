# E-VQA transfer (generated 2026-11-19 10:49)

Encyclopedic-VQA, authors' P20 protocol (Qwen part), k = 10, n = 500, canonical one-way matcher. Adapters were trained on InfoSeek only.

| Model | Gold first | Gold middle | Gold last | First-last gap | Status |
|---|---|---|---|---|---|
| Paper P20 Qwen (authors' reported) | 63.20 | 38.60 | 36.80 | 26.40 | reported in paper |
| Frozen Qwen | 64.40 | 39.00 | 35.20 | 29.20 | measured |
| Standard LoRA SFT | 27.67 | 25.60 | 24.67 | 3.00 | mean of 3 seeds |
| Balanced LoRA SFT | 27.73 | 27.07 | 26.60 | 1.13 | mean of 3 seeds |
| Gold-first LoRA SFT (control) | 38.00 | 11.47 | 11.93 | 26.07 | mean of 3 seeds |

## Paired bootstrap (B = 10,000, seed 42), percentage points

| Contrast | Estimate | 95% CI | CI excludes 0 |
|---|---|---|---|
| Frozen Qwen: gap (first - last) | +29.20 | [+25.00, +33.60] | yes |
| Standard LoRA SFT: gap (first - last) | +3.00 | [+1.13, +4.87] | yes |
| Balanced LoRA SFT: gap (first - last) | +1.13 | [-0.53, +2.80] | no |
| Gold-first LoRA SFT (control): gap (first - last) | +26.07 | [+22.33, +29.73] | yes |
| Standard LoRA SFT vs Frozen: change in gap | -26.20 | [-30.67, -21.80] | yes |
| Standard LoRA SFT vs Frozen: accuracy at gold first | -36.73 | [-41.00, -32.53] | yes |
| Standard LoRA SFT vs Frozen: accuracy at gold middle | -13.40 | [-17.60, -9.13] | yes |
| Standard LoRA SFT vs Frozen: accuracy at gold last | -10.53 | [-14.60, -6.40] | yes |
| Balanced LoRA SFT vs Frozen: change in gap | -28.07 | [-32.60, -23.73] | yes |
| Balanced LoRA SFT vs Frozen: accuracy at gold first | -36.67 | [-40.93, -32.40] | yes |
| Balanced LoRA SFT vs Frozen: accuracy at gold middle | -11.93 | [-16.20, -7.67] | yes |
| Balanced LoRA SFT vs Frozen: accuracy at gold last | -8.60 | [-12.73, -4.47] | yes |
| Gold-first LoRA SFT (control) vs Frozen: change in gap | -3.13 | [-8.07, +1.80] | no |
| Gold-first LoRA SFT (control) vs Frozen: accuracy at gold first | -26.40 | [-30.87, -21.93] | yes |
| Gold-first LoRA SFT (control) vs Frozen: accuracy at gold middle | -27.53 | [-31.53, -23.67] | yes |
| Gold-first LoRA SFT (control) vs Frozen: accuracy at gold last | -23.27 | [-27.07, -19.60] | yes |
| Balanced vs Standard: change in gap | -1.87 | [-3.47, -0.27] | yes |
| Balanced vs Standard: accuracy at gold first | +0.07 | [-1.33, +1.47] | no |
| Balanced vs Standard: accuracy at gold middle | +1.47 | [+0.07, +2.87] | yes |
| Balanced vs Standard: accuracy at gold last | +1.93 | [+0.60, +3.27] | yes |
| Gold-first vs Standard: change in gap | +23.07 | [+19.20, +27.00] | yes |
| Gold-first vs Standard: accuracy at gold first | +10.33 | [+7.47, +13.33] | yes |
| Gold-first vs Standard: accuracy at gold middle | -14.13 | [-17.40, -10.87] | yes |
| Gold-first vs Standard: accuracy at gold last | -12.73 | [-16.00, -9.53] | yes |
