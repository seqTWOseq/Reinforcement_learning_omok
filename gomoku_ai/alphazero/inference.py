"""Inference helpers for AlphaZero-compatible policy/value networks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from gomoku_ai.alphazero.model import PolicyValueNet
from gomoku_ai.alphazero.specs import STATE_SHAPE


def _validate_single_state_array(state_np: Any) -> np.ndarray:
    """Validate a single encoded state against the fixed NumPy contract."""

    resolved = np.asarray(state_np, dtype=np.float32)
    if resolved.shape != STATE_SHAPE:
        raise ValueError(f"state_np must have shape {STATE_SHAPE}, got {resolved.shape}.")
    if resolved.dtype != np.float32:
        resolved = resolved.astype(np.float32, copy=False)
    if not np.isfinite(resolved).all():
        raise ValueError("state_np must contain only finite values.")
    return resolved


def _get_model_device(model: PolicyValueNet) -> torch.device:
    """Return the current device of the model parameters."""

    return next(model.parameters()).device


def predict_single(
    model: PolicyValueNet,
    state_np: Any,
    device: torch.device | str | None = None,
    *,
    move_model: bool = False,
) -> tuple[np.ndarray, float]:
    """Run inference for a single encoded state and return NumPy outputs.

    By default this helper keeps the model on its current device and moves only
    the input tensor to match that device. If a different `device` is provided,
    the model is moved only when `move_model=True`.
    """

    if not isinstance(model, PolicyValueNet):
        raise TypeError("model must be a PolicyValueNet instance.")

    state = _validate_single_state_array(state_np)
    model_device = _get_model_device(model)
    if device is not None:
        requested_device = torch.device(device)
        if requested_device != model_device:
            if move_model:
                model = model.to(requested_device)
                model_device = requested_device
            else:
                raise ValueError(
                    "Requested device does not match the model device. "
                    "Move the model beforehand or pass move_model=True."
                )

    input_tensor = torch.from_numpy(state).unsqueeze(0).to(device=model_device, dtype=torch.float32)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        policy_logits, value = model(input_tensor)
    if was_training:
        model.train()

    policy_logits_np = policy_logits.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    value_scalar = float(value.squeeze(0).item())
    return policy_logits_np, value_scalar
