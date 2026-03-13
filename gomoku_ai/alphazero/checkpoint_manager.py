"""Checkpoint helpers for AlphaZero best-model promotion.

These checkpoints are separate from trainer checkpoints. They store only the
promoted best model plus evaluation metadata, not optimizer or replay state.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from gomoku_ai.alphazero.model import PolicyValueNet, PolicyValueNetConfig

if TYPE_CHECKING:
    from gomoku_ai.alphazero.evaluation import EvaluationResult

BEST_MODEL_CHECKPOINT_FORMAT_VERSION = 1
BEST_MODEL_CHECKPOINT_TYPE = "alphazero_best_model"


def save_best_model_checkpoint(
    model: PolicyValueNet,
    path: str | Path,
    *,
    source_cycle_index: int,
    evaluation_result: "EvaluationResult",
) -> Path:
    """Save a promoted best-model checkpoint with evaluation metadata."""

    if not isinstance(model, PolicyValueNet):
        raise TypeError("model must be a PolicyValueNet instance.")
    if source_cycle_index < 0:
        raise ValueError("source_cycle_index must be non-negative.")

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": BEST_MODEL_CHECKPOINT_FORMAT_VERSION,
        "checkpoint_type": BEST_MODEL_CHECKPOINT_TYPE,
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "source_cycle_index": source_cycle_index,
        "evaluation_result": evaluation_result.as_dict(),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_best_model_checkpoint(
    path: str | Path,
    device: torch.device | str | None = None,
) -> PolicyValueNet | None:
    """Load a promoted best model checkpoint if it exists."""

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None

    map_location = None if device is None else torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    required_fields = {
        "format_version",
        "checkpoint_type",
        "model_state_dict",
        "model_config",
        "source_cycle_index",
        "evaluation_result",
    }
    if not required_fields.issubset(checkpoint):
        raise ValueError(
            "Best-model checkpoint must contain "
            "'format_version', 'checkpoint_type', 'model_state_dict', "
            "'model_config', 'source_cycle_index', and 'evaluation_result'."
        )
    if checkpoint["format_version"] != BEST_MODEL_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported best-model format_version {checkpoint['format_version']!r}; "
            f"expected {BEST_MODEL_CHECKPOINT_FORMAT_VERSION}."
        )
    if checkpoint["checkpoint_type"] != BEST_MODEL_CHECKPOINT_TYPE:
        raise ValueError(
            f"Unsupported checkpoint_type {checkpoint['checkpoint_type']!r}; "
            f"expected {BEST_MODEL_CHECKPOINT_TYPE!r}."
        )

    model = PolicyValueNet(config=PolicyValueNetConfig(**dict(checkpoint["model_config"])))
    model.load_state_dict(checkpoint["model_state_dict"])
    if device is not None:
        model = model.to(torch.device(device))
    return model
