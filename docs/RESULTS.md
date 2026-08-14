# TaskWatch full-DROID Stage 1 results

The table below records the audited checkpoints from the ongoing 10,000-step
reference run. Metrics use the fixed 1,000-episode validation set. Since 4% of
the validation captions are duplicated, the primary Video-to-Text numbers count
any identical target caption as correct.

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
| 5,000 | **32.8%** | **70.0%** | **82.8%** | **3** | 0.5944 |

At step 3,500, R@1 fluctuated down while R@10 and positive cosine continued to
improve. Step 4,000 recovered that dip, and steps 4,500 and 5,000 improved all
three primary recall metrics again. Step 4,500 is an evaluation-only milestone;
step 5,000 is the current best audited checkpoint. The run is still not treated
as monotonically improving.

The random duplicate-aware V2T R@1 baseline is 0.1124%. Machine-readable metric
files live in [`artifacts/full_stage1`](../artifacts/full_stage1/). Full
checkpoints, DROID video tensors, and model weights are intentionally excluded
from Git.

This document is an interim record. It must not be interpreted as the final
10,000-step result until the final checkpoint, retrieval JSON, and loss plots
have been generated and audited.
