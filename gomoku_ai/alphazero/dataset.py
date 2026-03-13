"""PyTorch dataset wrappers for AlphaZero training samples."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from gomoku_ai.alphazero.specs import GameStepSample, POLICY_SHAPE, STATE_SHAPE


class AlphaZeroSampleDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset over `GameStepSample` objects for policy/value training.

    Each item returns:
    - `state`: shape `(4, 15, 15)`, dtype `torch.float32`
    - `policy_target`: shape `(225,)`, dtype `torch.float32`
    - `value_target`: shape `(1,)`, dtype `torch.float32`
    """

    def __init__(self, samples: Sequence[GameStepSample]) -> None:
        """Store a validated sequence of AlphaZero training samples."""

        self.samples = list(samples)
        for sample in self.samples:
            if not isinstance(sample, GameStepSample):
                raise TypeError("samples must contain only GameStepSample instances.")

    def __len__(self) -> int:
        """Return the number of available samples."""

        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one training sample as PyTorch tensors."""

        sample = self.samples[index]
        state = np.asarray(sample.state, dtype=np.float32)
        policy_target = np.asarray(sample.policy_target, dtype=np.float32)
        value_target = np.asarray([sample.value_target], dtype=np.float32)

        if state.shape != STATE_SHAPE:
            raise ValueError(f"sample.state must have shape {STATE_SHAPE}, got {state.shape}.")
        if policy_target.shape != POLICY_SHAPE:
            raise ValueError(f"sample.policy_target must have shape {POLICY_SHAPE}, got {policy_target.shape}.")
        if value_target.shape != (1,):
            raise ValueError(f"value_target tensor must have shape (1,), got {value_target.shape}.")

        return (
            torch.from_numpy(state),
            torch.from_numpy(policy_target),
            torch.from_numpy(value_target),
        )
