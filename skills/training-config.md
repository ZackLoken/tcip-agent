---
name: training-config
description: "Multi-stage progressive unfreezing, optimizer setup, LR scheduling, mixed precision, early stopping, GPU memory budgeting, and HPO strategy with multi-round narrowing."
triggers:
  - training
  - train
  - hyperparameter
  - HPO
  - learning rate
  - optimizer
  - augmentation
  - early stopping
  - checkpoint
  - mixed precision
  - batch size
  - unfreezing
modes: [CodeGenerator, TrainingOrchestrator]
priority: high
max_chars: 4000
---

# Training Configuration

## Purpose

Configure multi-stage progressive unfreezing training, HPO search, and all training hyperparameters. Covers optimizer setup, LR scheduling, mixed precision, early stopping, and GPU memory budgeting.

## Multi-Stage Progressive Unfreezing

| Stage | Layers Unfrozen | LR | Epochs | Rationale |
|-------|----------------|-----|--------|-----------|
| 1 | Head only | 1e-3 | 5 | Adapt head to new classes |
| 2 | Head + last backbone block | 1e-4 | 10 | Fine-tune high-level features |
| 3 | Head + last 2 blocks | 3e-5 | 10 | Deeper adaptation |
| 4 | All layers | 1e-5 | 5 | Full fine-tuning polish |

**LR scaling rule**: scale base LR by √(batch_size / 16).

## Optimizer & Scheduler

- AdamW with weight_decay=1e-4
- Cosine annealing per stage (T_max = stage epochs)
- Linear warmup: 3 epochs in stage 1, 1 epoch in subsequent stages
- State handoff: create fresh optimizer per stage (different param groups)

## Mixed Precision

Always use `torch.amp.autocast("cuda")` + `GradScaler` when on GPU. Disable for CPU training. Use FrozenBatchNorm during frozen stages, switch to regular BN when unfreezing.

## Early Stopping

EMA-smoothed val loss (α=0.1). Patience: 5 epochs stage 1-2, 3 epochs stage 3-4. Save best checkpoint by primary metric (mAP for detection, F1 for classification).

## Checkpoint Strategy

- **Always save**: metrics JSON per epoch
- **Best + last 2**: model weights
- **Best only**: optimizer state (saves disk)

## GPU Memory Budget (24 GB reference)

| Model | Max Batch | Grad Accumulation |
|-------|-----------|-------------------|
| Faster R-CNN ResNet-50-FPN | 4 | 4 (eff=16) |
| Mask R-CNN ResNet-50-FPN | 2 | 8 (eff=16) |
| FCOS ResNet-50-FPN | 4 | 4 (eff=16) |
| ResNet-50 classifier | 32 | 1 |

## HPO Strategy

**3-round narrowing protocol:**
1. **Broad** (50 trials): ASHA scheduler, explore full space
2. **Targeted** (50 trials): narrow to top 30% ranges from round 1
3. **Edge-check** (30 trials, conditional): probe boundaries of best region

```
Round 1 (broad, 50 trials)
  └─ analyze → narrow search space
Round 2 (narrowed, 50 trials)
  └─ analyze → any hparam at boundary?
       ├─ NO  → accept Round 2 best → DONE
       └─ YES → Round 3 (edge-check, 30-50 trials)
                  └─ accept best across all rounds → DONE
```

**Between-round agent duties:**
1. Rank trials by composite objective
2. Compute per-parameter stats of top-10 trials (mean, std, min, max)
3. Run fANOVA for parameter importance rankings
4. Check if any top-trial hparam clusters at a search-space boundary (within 10%)
5. Report findings at HITL checkpoint

**Search spaces** — detection: LR [1e-5, 1e-2], batch [2,4,8], weight_decay [1e-5, 1e-2].
Scheduler: ASHA (max_t=50, grace=5, reduction=3).
Sampler: Optuna TPE (n_startup=10, multivariate=True).

**Composite objectives:**
- Detection: `0.45 * norm_val_loss + 0.35 * (1 - F1) + 0.20 * (1 - mAP50)`
- Regression: `0.45 * norm_val_loss + 0.35 * (1 - R²) + 0.20 * norm_MAE`
- Classification: `0.45 * norm_val_loss + 0.35 * (1 - F1) + 0.20 * (1 - balanced_acc)`

## Key Constraints

- HITL checkpoint #3: present config for approval before launching training
- NaN guard: stop trial if loss is NaN for 3 consecutive batches
- Log all metrics to JSONL for the session record
- Each trial: 1 GPU, max 4 concurrent, 50 epochs max before ASHA terminates
- Total trials across all rounds: ~100-150 (simple tasks) to ~200-250 (detection/segmentation)
