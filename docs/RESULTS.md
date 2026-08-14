# Full-DROID Stage 1 results

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
| 3,000 | **29.3%** | **63.8%** | 77.3% | **3** | 0.5462 |
| 3,500 | 26.9% | 63.6% | **77.8%** | **3** | 0.5652 |

At step 3,500, R@1 fluctuated down while R@10 and positive cosine continued to
improve. The run is therefore not treated as monotonically improving, and the
step-3,000 checkpoint remains the best audited R@1 checkpoint so far.

The random duplicate-aware V2T R@1 baseline is 0.1124%. Machine-readable metric
files live in [`artifacts/full_stage1`](../artifacts/full_stage1/). Full
checkpoints, DROID video tensors, and model weights are intentionally excluded
from Git.

This document is an interim record. It must not be interpreted as the final
10,000-step result until the final checkpoint, retrieval JSON, and loss plots
have been generated and audited.
