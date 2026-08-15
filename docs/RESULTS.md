# TaskWatch full-DROID Stage 1 results

The table below records the audited checkpoints from the ongoing 20,000-step
reference run. Step 10,000 is a stage boundary where the checkpoint is audited
and then resumed with optimizer state intact. Metrics use the fixed
1,000-episode validation set. Since 4% of the validation captions are
duplicated, the primary Video-to-Text numbers count any identical target
caption as correct.

| Step | V2T R@1 | V2T R@5 | V2T R@10 | Median rank | Positive cosine |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 10.5% | 35.1% | 50.5% | 10 | 0.3956 |
| 1,000 | 16.0% | 48.7% | 63.7% | 6 | 0.4802 |
| 1,500 | 20.4% | 53.6% | 68.8% | 5 | 0.5099 |
| 2,000 | 21.6% | 56.2% | 72.7% | 4 | 0.5218 |
| 2,500 | 25.9% | 61.0% | 76.3% | 4 | 0.5421 |
| 3,000 | 29.3% | 63.8% | 77.3% | **3** | 0.5462 |
| 3,500 | 26.9% | 63.6% | 77.8% | **3** | 0.5652 |
| 4,000 | 29.7% | 65.3% | 80.0% | **3** | 0.5673 |
| 4,500 | 30.5% | 67.2% | 80.3% | **3** | 0.5895 |
| 5,000 | 32.8% | 70.0% | 82.8% | **3** | 0.5944 |
| 5,500 | **34.4%** | **71.0%** | **83.4%** | **3** | 0.6042 |
| 6,000 | **35.7%** | **72.1%** | **84.3%** | **2** | 0.6213 |
| 6,500 | 33.9% | **72.9%** | **84.6%** | **2** | 0.6214 |
| 7,000 | 34.7% | **73.0%** | 84.5% | **2** | 0.6096 |
| 7,500 | **38.0%** | **73.8%** | **86.2%** | **2** | 0.6275 |
| 8,000 | 36.4% | 72.2% | 84.2% | **2** | 0.6331 |

At step 3,500, R@1 fluctuated down while R@10 and positive cosine continued to
improve. Step 4,000 recovered that dip, and subsequent milestones through step
6,000 improved all three primary recall metrics again. At step 6,500, R@1
fluctuated down by 1.8 percentage points while R@5 and R@10 improved by 0.8 and
0.3 points, respectively; median rank stayed at 2. Step 7,000 recovered 0.8
R@1 points and set a new R@5 high of 73.0%, while R@10 edged down by 0.1 point.
Step 7,500 then broke through the plateau and set new highs for R@1, R@5, and
R@10. Step 8,000 pulled back by 1.6, 1.6, and 2.0 percentage points,
respectively, despite a higher positive cosine. Step 7,500 remains the current
best evaluation milestone, while step 8,000 is the latest saved checkpoint. The
checkpoint contains
the Predictor, Y-Encoder, three optimizer parameter groups with 409 states, and
the constant scheduler at global step 8,000; it excludes the frozen V-JEPA
encoder and Qwen token embedding. The 6.4 GiB checkpoint SHA-256 is
`84e59cebb43da82be73372375e899813501f51b4ef475907070aad35765ec4f6`.
The run is still not treated as monotonically improving.

The random duplicate-aware V2T R@1 baseline is 0.1124%. Machine-readable metric
files live in [`artifacts/full_stage1`](../artifacts/full_stage1/). Full
checkpoints, DROID video tensors, and model weights are intentionally excluded
from Git.

This document is an interim record. It must not be interpreted as the final
20,000-step result until the final checkpoint, retrieval JSON, and loss plots
have been generated and audited.
