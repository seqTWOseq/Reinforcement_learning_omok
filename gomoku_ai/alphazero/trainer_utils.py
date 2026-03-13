"""Utility helpers for the simple single-process AlphaZero trainer."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gomoku_ai.alphazero.dataset import AlphaZeroSampleDataset
from gomoku_ai.alphazero.specs import GameStepSample


def trim_samples(samples: Sequence[GameStepSample], max_buffer_samples: int) -> list[GameStepSample]:
    """Keep only the most recent `max_buffer_samples` samples."""

    if max_buffer_samples <= 0:
        raise ValueError("max_buffer_samples must be positive.")
    normalized = list(samples)
    if len(normalized) <= max_buffer_samples:
        return normalized
    return normalized[-max_buffer_samples:]


def compute_policy_loss(policy_logits: torch.Tensor, policy_target: torch.Tensor) -> torch.Tensor:
    """Compute soft cross entropy for distribution targets.

    This is intentionally not hard-label cross entropy. The target is a full
    move distribution, so the loss is:

    `-(target_probs * log_softmax(logits)).sum(dim=1).mean()`
    """

    if policy_logits.ndim != 2 or policy_target.ndim != 2:
        raise ValueError("policy_logits and policy_target must both be rank-2 tensors.")
    if policy_logits.shape != policy_target.shape:
        raise ValueError(
            f"policy_logits and policy_target must have matching shapes, got {tuple(policy_logits.shape)} and {tuple(policy_target.shape)}."
        )
    log_probs = torch.log_softmax(policy_logits, dim=1)
    loss = -(policy_target * log_probs).sum(dim=1).mean()
    if not torch.isfinite(loss):
        raise ValueError("policy loss must be finite.")
    return loss


def compute_value_loss(value_pred: torch.Tensor, value_target: torch.Tensor) -> torch.Tensor:
    """Compute value loss as MSE over `(N, 1)` tensors."""

    if value_pred.ndim != 2 or value_target.ndim != 2:
        raise ValueError("value_pred and value_target must both be rank-2 tensors.")
    if value_pred.shape != value_target.shape:
        raise ValueError(
            f"value_pred and value_target must have matching shapes, got {tuple(value_pred.shape)} and {tuple(value_target.shape)}."
        )
    if value_pred.shape[1] != 1:
        raise ValueError(f"value tensors must have shape (N, 1), got {tuple(value_pred.shape)}.")
    loss = F.mse_loss(value_pred, value_target)
    if not torch.isfinite(loss):
        raise ValueError("value loss must be finite.")
    return loss


def build_dataloader_from_buffer(
    samples: Sequence[GameStepSample],
    batch_size: int,
    *,
    shuffle: bool = True,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build a standard PyTorch dataloader from buffered AlphaZero samples."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    dataset = AlphaZeroSampleDataset(samples)
    if len(dataset) == 0:
        raise ValueError("Cannot build a dataloader from an empty sample list.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
