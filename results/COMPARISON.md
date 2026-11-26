# Comparison (generated 2026-11-26 12:14)

InfoSeek, authors' P17 protocol, k = 10, n = 495. Accuracy in %. Gap = gold first - gold last.

| Model | Gold first | Gold middle | Gold last | First-last gap | Status |
|---|---|---|---|---|---|
| Paper reproduction (authors' reported) | 45.16 | 32.26 | 28.63 | 16.53 | reported in paper |
| Frozen Qwen | 45.05 | 32.53 | 29.49 | 15.56 | measured (1 run, greedy decoding) |
| Standard LoRA SFT | 54.68 | 55.22 | 53.94 | 0.74 | mean of 3 seeds |
| Balanced LoRA SFT | 54.07 | 56.03 | 54.61 | -0.54 | mean of 3 seeds |
| Gold-first LoRA SFT (control) | 69.63 | 14.14 | 14.21 | 55.42 | mean of 3 seeds |
| Mixed-data standard SFT | 54.48 | 55.96 | 55.02 | -0.54 | mean of 3 seeds |
| Mixed-data balanced SFT | 54.88 | 56.43 | 54.68 | 0.20 | mean of 3 seeds |
| Late-exposure 2% SFT | 56.16 | 56.97 | 51.99 | 4.17 | mean of 3 seeds |
| Late-exposure 4% SFT | 55.02 | 55.22 | 51.99 | 3.03 | mean of 3 seeds |
| Late-exposure 15% SFT | 53.40 | 53.47 | 55.22 | -1.82 | mean of 3 seeds |
| Isolated reading (frozen, mean log-prob) | 31.11 | 31.11 | 31.11 | 0.00 | measured (order-invariant by construction) |

## Paired bootstrap (B = 10,000, seed 42), percentage points

| Contrast | Estimate | 95% CI | CI excludes 0 |
|---|---|---|---|
| Frozen Qwen: gap (first - last) | +15.56 | [+11.72, +19.60] | yes |
| Standard LoRA SFT: gap (first - last) | +0.74 | [-0.81, +2.29] | no |
| Balanced LoRA SFT: gap (first - last) | -0.54 | [-2.15, +1.08] | no |
| Gold-first LoRA SFT (control): gap (first - last) | +55.42 | [+51.38, +59.60] | yes |
| Mixed-data standard SFT: gap (first - last) | -0.54 | [-2.29, +1.21] | no |
| Mixed-data balanced SFT: gap (first - last) | +0.20 | [-1.55, +2.09] | no |
| Late-exposure 2% SFT: gap (first - last) | +4.18 | [+2.42, +6.06] | yes |
| Late-exposure 4% SFT: gap (first - last) | +3.03 | [+1.28, +4.92] | yes |
| Late-exposure 15% SFT: gap (first - last) | -1.82 | [-3.84, +0.13] | no |
| Isolated reading (frozen, mean log-prob): gap (first - last) | +0.00 | [+0.00, +0.00] | no |
| Standard LoRA SFT vs Frozen: change in gap | -14.81 | [-19.06, -10.84] | yes |
| Standard LoRA SFT vs Frozen: accuracy at gold first | +9.63 | [+5.19, +14.01] | yes |
| Standard LoRA SFT vs Frozen: accuracy at gold middle | +22.69 | [+18.45, +27.00] | yes |
| Standard LoRA SFT vs Frozen: accuracy at gold last | +24.44 | [+20.13, +28.89] | yes |
| Balanced LoRA SFT vs Frozen: change in gap | -16.09 | [-20.20, -12.05] | yes |
| Balanced LoRA SFT vs Frozen: accuracy at gold first | +9.02 | [+4.65, +13.33] | yes |
| Balanced LoRA SFT vs Frozen: accuracy at gold middle | +23.50 | [+19.26, +27.88] | yes |
| Balanced LoRA SFT vs Frozen: accuracy at gold last | +25.12 | [+20.81, +29.49] | yes |
| Gold-first LoRA SFT (control) vs Frozen: change in gap | +39.87 | [+35.02, +44.72] | yes |
| Gold-first LoRA SFT (control) vs Frozen: accuracy at gold first | +24.58 | [+20.20, +29.02] | yes |
| Gold-first LoRA SFT (control) vs Frozen: accuracy at gold middle | -18.38 | [-22.49, -14.41] | yes |
| Gold-first LoRA SFT (control) vs Frozen: accuracy at gold last | -15.29 | [-19.26, -11.45] | yes |
| Mixed-data standard SFT vs Frozen: change in gap | -16.09 | [-20.27, -12.05] | yes |
| Mixed-data standard SFT vs Frozen: accuracy at gold first | +9.43 | [+5.12, +13.67] | yes |
| Mixed-data standard SFT vs Frozen: accuracy at gold middle | +23.43 | [+19.19, +27.81] | yes |
| Mixed-data standard SFT vs Frozen: accuracy at gold last | +25.52 | [+21.21, +29.90] | yes |
| Mixed-data balanced SFT vs Frozen: change in gap | -15.35 | [-19.53, -11.18] | yes |
| Mixed-data balanced SFT vs Frozen: accuracy at gold first | +9.83 | [+5.59, +14.07] | yes |
| Mixed-data balanced SFT vs Frozen: accuracy at gold middle | +23.91 | [+19.73, +28.15] | yes |
| Mixed-data balanced SFT vs Frozen: accuracy at gold last | +25.19 | [+20.81, +29.56] | yes |
| Late-exposure 2% SFT vs Frozen: change in gap | -11.38 | [-15.42, -7.41] | yes |
| Late-exposure 2% SFT vs Frozen: accuracy at gold first | +11.11 | [+6.73, +15.42] | yes |
| Late-exposure 2% SFT vs Frozen: accuracy at gold middle | +24.44 | [+20.34, +28.62] | yes |
| Late-exposure 2% SFT vs Frozen: accuracy at gold last | +22.49 | [+18.32, +26.73] | yes |
| Late-exposure 4% SFT vs Frozen: change in gap | -12.53 | [-16.70, -8.48] | yes |
| Late-exposure 4% SFT vs Frozen: accuracy at gold first | +9.97 | [+5.59, +14.21] | yes |
| Late-exposure 4% SFT vs Frozen: accuracy at gold middle | +22.69 | [+18.52, +27.01] | yes |
| Late-exposure 4% SFT vs Frozen: accuracy at gold last | +22.49 | [+18.32, +26.80] | yes |
| Late-exposure 15% SFT vs Frozen: change in gap | -17.37 | [-21.75, -13.20] | yes |
| Late-exposure 15% SFT vs Frozen: accuracy at gold first | +8.35 | [+3.91, +12.73] | yes |
| Late-exposure 15% SFT vs Frozen: accuracy at gold middle | +20.94 | [+16.77, +25.25] | yes |
| Late-exposure 15% SFT vs Frozen: accuracy at gold last | +25.72 | [+21.55, +29.97] | yes |
| Isolated reading (frozen, mean log-prob) vs Frozen: change in gap | -15.56 | [-19.60, -11.72] | yes |
| Isolated reading (frozen, mean log-prob) vs Frozen: accuracy at gold first | -13.94 | [-17.78, -10.10] | yes |
| Isolated reading (frozen, mean log-prob) vs Frozen: accuracy at gold middle | -1.41 | [-5.05, +2.22] | no |
| Isolated reading (frozen, mean log-prob) vs Frozen: accuracy at gold last | +1.62 | [-2.22, +5.45] | no |
| Standard LoRA SFT vs Isolated reading (frozen, mean log-prob): accuracy at gold first | +23.57 | [+19.26, +27.95] | yes |
| Standard LoRA SFT vs Isolated reading (frozen, mean log-prob): accuracy at gold middle | +24.11 | [+19.73, +28.62] | yes |
| Standard LoRA SFT vs Isolated reading (frozen, mean log-prob): accuracy at gold last | +22.83 | [+18.38, +27.27] | yes |
| Balanced LoRA SFT vs Isolated reading (frozen, mean log-prob): accuracy at gold first | +22.96 | [+18.59, +27.41] | yes |
| Balanced LoRA SFT vs Isolated reading (frozen, mean log-prob): accuracy at gold middle | +24.92 | [+20.61, +29.29] | yes |
| Balanced LoRA SFT vs Isolated reading (frozen, mean log-prob): accuracy at gold last | +23.50 | [+19.19, +27.81] | yes |
| Balanced LoRA SFT vs Standard LoRA SFT: change in gap | -1.28 | [-3.16, +0.61] | no |
| Balanced LoRA SFT vs Standard LoRA SFT: accuracy at gold first | -0.61 | [-2.22, +1.01] | no |
| Balanced LoRA SFT vs Standard LoRA SFT: accuracy at gold middle | +0.81 | [-0.67, +2.29] | no |
| Balanced LoRA SFT vs Standard LoRA SFT: accuracy at gold last | +0.67 | [-0.88, +2.15] | no |
| Gold-first LoRA SFT (control) vs Standard LoRA SFT: change in gap | +54.68 | [+50.37, +59.19] | yes |
| Gold-first LoRA SFT (control) vs Standard LoRA SFT: accuracy at gold first | +14.95 | [+11.78, +18.25] | yes |
| Gold-first LoRA SFT (control) vs Standard LoRA SFT: accuracy at gold middle | -41.08 | [-45.25, -36.97] | yes |
| Gold-first LoRA SFT (control) vs Standard LoRA SFT: accuracy at gold last | -39.73 | [-43.84, -35.69] | yes |
| [k=20] Frozen Qwen: gap (first - last) | +16.36 | [+12.53, +20.40] | yes |
| [k=20] Standard LoRA SFT: gap (first - last) | +0.74 | [-1.01, +2.49] | no |
| [k=20] Balanced LoRA SFT: gap (first - last) | +0.13 | [-1.68, +2.02] | no |
| [k=20] Gold-first LoRA SFT (control): gap (first - last) | +54.88 | [+50.84, +59.06] | yes |
| [k=20] Mixed-data standard SFT: gap (first - last) | +3.23 | [+1.14, +5.32] | yes |
| [k=20] Mixed-data balanced SFT: gap (first - last) | -1.82 | [-3.70, +0.07] | no |
| [k=20] Late-exposure 2% SFT: gap (first - last) | +5.52 | [+3.70, +7.41] | yes |
| [k=20] Late-exposure 4% SFT: gap (first - last) | +4.04 | [+2.22, +5.93] | yes |
| [k=20] Late-exposure 15% SFT: gap (first - last) | -1.08 | [-2.83, +0.74] | no |
| [k=20] Standard LoRA SFT vs Frozen: change in gap | -15.62 | [-19.87, -11.52] | yes |
| [k=20] Balanced LoRA SFT vs Frozen: change in gap | -16.23 | [-20.27, -12.39] | yes |
| [k=20] Gold-first LoRA SFT (control) vs Frozen: change in gap | +38.52 | [+33.54, +43.43] | yes |
| [k=20] Mixed-data standard SFT vs Frozen: change in gap | -13.13 | [-17.44, -8.96] | yes |
| [k=20] Mixed-data balanced SFT vs Frozen: change in gap | -18.18 | [-22.49, -14.01] | yes |
| [k=20] Late-exposure 2% SFT vs Frozen: change in gap | -10.84 | [-14.75, -7.07] | yes |
| [k=20] Late-exposure 4% SFT vs Frozen: change in gap | -12.32 | [-16.57, -8.28] | yes |
| [k=20] Late-exposure 15% SFT vs Frozen: change in gap | -17.44 | [-21.75, -13.33] | yes |
| [k=20] Balanced LoRA SFT vs Standard LoRA SFT: change in gap | -0.61 | [-2.69, +1.41] | no |
| [k=20] Balanced LoRA SFT vs Standard LoRA SFT: accuracy at gold first | +0.81 | [-0.88, +2.56] | no |
| [k=20] Balanced LoRA SFT vs Standard LoRA SFT: accuracy at gold middle | +1.95 | [+0.27, +3.64] | yes |
| [k=20] Balanced LoRA SFT vs Standard LoRA SFT: accuracy at gold last | +1.41 | [-0.40, +3.16] | no |
| [k=20] Gold-first LoRA SFT (control) vs Standard LoRA SFT: change in gap | +54.14 | [+49.70, +58.65] | yes |
| Late-exposure 2% SFT vs Standard LoRA SFT: change in gap | +3.43 | [+1.48, +5.52] | yes |
| Late-exposure 2% SFT vs Standard LoRA SFT: accuracy at gold first | +1.48 | [-0.13, +3.10] | no |
| Late-exposure 2% SFT vs Standard LoRA SFT: accuracy at gold middle | +1.75 | [+0.20, +3.30] | yes |
| Late-exposure 2% SFT vs Standard LoRA SFT: accuracy at gold last | -1.95 | [-3.70, -0.27] | yes |
| Late-exposure 4% SFT vs Standard LoRA SFT: change in gap | +2.29 | [+0.54, +4.11] | yes |
| Late-exposure 4% SFT vs Standard LoRA SFT: accuracy at gold first | +0.34 | [-1.28, +1.95] | no |
| Late-exposure 4% SFT vs Standard LoRA SFT: accuracy at gold middle | +0.00 | [-1.55, +1.48] | no |
| Late-exposure 4% SFT vs Standard LoRA SFT: accuracy at gold last | -1.95 | [-3.43, -0.54] | yes |
| Late-exposure 15% SFT vs Standard LoRA SFT: change in gap | -2.56 | [-4.44, -0.61] | yes |
| Late-exposure 15% SFT vs Standard LoRA SFT: accuracy at gold first | -1.28 | [-2.90, +0.34] | no |
| Late-exposure 15% SFT vs Standard LoRA SFT: accuracy at gold middle | -1.75 | [-3.37, -0.13] | yes |
| Late-exposure 15% SFT vs Standard LoRA SFT: accuracy at gold last | +1.28 | [-0.34, +2.90] | no |
| Mixed-data standard SFT vs Standard LoRA SFT: change in gap | -1.28 | [-3.43, +0.88] | no |
| Mixed-data standard SFT vs Standard LoRA SFT: accuracy at gold first | -0.20 | [-2.49, +2.02] | no |
| Mixed-data standard SFT vs Standard LoRA SFT: accuracy at gold middle | +0.74 | [-1.41, +2.90] | no |
| Mixed-data standard SFT vs Standard LoRA SFT: accuracy at gold last | +1.08 | [-1.14, +3.30] | no |
| Mixed-data balanced SFT vs Mixed-data standard SFT: change in gap | +0.74 | [-1.41, +2.90] | no |
| Mixed-data balanced SFT vs Mixed-data standard SFT: accuracy at gold first | +0.40 | [-1.41, +2.29] | no |
| Mixed-data balanced SFT vs Mixed-data standard SFT: accuracy at gold middle | +0.47 | [-1.14, +2.02] | no |
| Mixed-data balanced SFT vs Mixed-data standard SFT: accuracy at gold last | -0.34 | [-2.09, +1.41] | no |
