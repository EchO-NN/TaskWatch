# TaskWatch

Efficient video-based subtask completion verification for embodied agents.

TaskWatch starts from a strict Stage 1 video-language alignment baseline for
DROID robot videos. The visual encoder is a frozen V-JEPA 2.1 ViT-G/16 2B, the
predictor is initialized from the final eight Qwen3-4B transformer blocks, and
the trainable language target is EmbeddingGemma-300M.

The reference 10,000-step full-DROID run is currently in progress. Audited
metrics and their machine-readable artifacts are published in
[`docs/RESULTS.md`](docs/RESULTS.md); no final-result claim is made before the
step-10,000 checkpoint and plots pass the same audit.

## Model contract

```text
32 x 256 px DROID frames
        |
        v
frozen V-JEPA 2.1 ViT-G/16 2B
        |  [B, 4096, 1664]
        v
Linear(1664, 2560)
        |
        v
Qwen3-4B layers 28..35, non-causal
        |
        v
mean pool -> Linear(2560, 1536) -> L2 normalize
        |
        +------ bidirectional InfoNCE ------+
                                             |
DROID language -> EmbeddingGemma-300M -> Linear(768, 1536)
                                             |
                                      L2 normalize
```

The baseline deliberately contains no Q-Former, Perceiver/resampler, visual
token compression, LM head, token cross-entropy, action-classification head,
multi-view fusion, horizontal flip, or V-JEPA fine-tuning. Stage 1 is
query-free.

## Data contract

- Source: DROID RLDS `1.0.1`.
- Camera: `exterior_image_1_left`.
- Sampling: exactly 32 uniformly spaced frames per eligible episode.
- Resize/crop: 256 x 256 model input, ImageNet normalization.
- Filter: successful episodes only; short episodes are dropped.
- Reference split: 73,602 train episodes and a deterministic 1,000-episode
  validation set from 74,602 eligible episodes.
- Validation captions: 960 unique strings; duplicate-aware Video-to-Text metrics
  count any identical target caption as correct.

Raw DROID data, prepared frame tensors, feature caches, model weights, and full
checkpoints are not stored in this repository.

## Requirements

- Linux with Python 3.11 or 3.12.
- CUDA-capable GPU(s) with BF16 support for the reference training path.
- PyTorch 2.7.1 and torchvision 0.22.1.
- A local checkout of V-JEPA 2 at commit
  `204698b45b3712590f06245fbfba32d3be539812`.
- V-JEPA 2.1 ViT-G/16 2B checkpoint.
- Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- EmbeddingGemma-300M revision
  `57c266a740f537b4dc058e1b0cda161fd15afa75`.
- Access to DROID RLDS `1.0.1`.

Model and dataset access remains subject to each upstream project's license and
access terms.

## Installation

```bash
git clone https://github.com/EchO-NN/TaskWatch.git
cd TaskWatch

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Clone the V-JEPA 2 source separately and pin it before running the real-model
checks:

```bash
git clone https://github.com/facebookresearch/vjepa2.git third_party/vjepa2
git -C third_party/vjepa2 checkout 204698b45b3712590f06245fbfba32d3be539812
```

The committed YAML files record the reference host's absolute paths. Copy the
closest config and edit only its `paths` section for your machine. Keep the
pinned source revisions and model/data contract unchanged when reproducing the
baseline.

Reference layout:

```text
DROID RLDS       /data/nyz/datasets/droid/1.0.1
prepared data    /data/nyz/vljepa_droid_256/data/full
V-JEPA source    /data/nyz/repos/vjepa2
V-JEPA checkpoint /data/nyz/models/vjepa2.1-vitG-2B-384/vjepa2_1_vitG_384.pt
Qwen3-4B         /data/nyz/models/Qwen3-4B
EmbeddingGemma   /data/nyz/models/embeddinggemma-300m
outputs          /data/nyz/vljepa_droid_256/outputs/full_stage1
```

## Prepare DROID

First inspect a few RLDS episodes and verify the camera/language fields:

```bash
python scripts/inspect_droid.py \
  --config configs/train_stage1_droid_256.yaml \
  --episodes 3
```

Run the bounded overfit preparation first:

```bash
python scripts/prepare_overfit_subset.py \
  --config configs/stage1_overfit_1k.yaml
```

After the overfit gate passes, scan all shards and materialize the full fixed
split. The command refuses to start without the requested free-space headroom:

```bash
python scripts/prepare_full_dataset.py \
  --config configs/train_stage1_droid_256.yaml \
  --minimum-free-gib 250
```

## Validate the implementation

CPU/unit checks:

```bash
pytest -q tests \
  --ignore=tests/test_real_models.py \
  --ignore=tests/test_distributed_gather.py \
  --ignore=tests/nccl_smoke.py \
  --ignore=tests/nccl_gather_smoke.py
```

Real-model contract checks:

```bash
CUDA_VISIBLE_DEVICES=0 \
VJEPA_REPO=/path/to/vjepa2 \
VJEPA_CHECKPOINT=/path/to/vjepa2_1_vitG_384.pt \
QWEN_MODEL=/path/to/Qwen3-4B \
pytest -q tests/test_real_models.py
```

Cross-rank differentiable gather and chunked-gradient checks:

```bash
torchrun --standalone --nproc-per-node=2 -m pytest -q \
  tests/test_distributed_gather.py \
  tests/test_chunked_grad_sync.py
```

## Train and evaluate

Overfit gate:

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_stage1.py \
  --config configs/stage1_overfit_1k.yaml

python scripts/summarize_gate.py --output-dir outputs/overfit_1k
```

Full Stage 1 reference run:

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_stage1.py \
  --config configs/train_stage1_droid_256.yaml
```

Evaluate a checkpoint and plot the training trace:

```bash
python scripts/eval_retrieval.py \
  --config configs/train_stage1_droid_256.yaml \
  --checkpoint outputs/full_stage1/checkpoint_step_0010000.pt

python scripts/plot_loss.py \
  --log outputs/full_stage1/train.jsonl \
  --output-prefix outputs/full_stage1/loss
```

Checkpoints save only trainable predictor/Y-Encoder state plus optimizer,
scheduler, config, and contract metadata. Frozen V-JEPA weights are omitted.

## Dual-GPU compatibility path

The reference Pro 6000 host has two Blackwell GPUs connected by PCIe without
NVLink. Its stock DDP reducer raised CUDA error 700 on the 1.12B-trainable-
parameter graph. The declared
`gradient_sync: chunked_gloo_allreduce` setting performs bounded synchronous
cross-rank gradient averaging through CPU Gloo in 8,388,608-element chunks.
NCCL remains in use for the small differentiable embedding gather. This is a
host compatibility setting, not a change to the model, global InfoNCE batch, or
optimizer math.

## Repository contents

```text
configs/                  experiment and profile configs
scripts/                  preparation, training, evaluation, and plotting CLIs
src/vljepa_droid/         package implementation
tests/                    unit, distributed, and real-model contract tests
artifacts/overfit_1k/     audited bounded-gate outputs
artifacts/full_stage1/    small metric/metadata evidence from the full run
docs/RESULTS.md           human-readable metric history
```

## Upstream projects

- [V-JEPA 2](https://github.com/facebookresearch/vjepa2)
- [Qwen3](https://huggingface.co/Qwen/Qwen3-4B)
- [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m)
- [DROID](https://droid-dataset.github.io/)

This repository contains integration and training code only. It does not
redistribute upstream source, datasets, or model weights.
