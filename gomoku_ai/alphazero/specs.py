"""Typed data structures for AlphaZero-compatible state and training samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from gomoku_ai.env import BLACK, BOARD_SIZE, DRAW, WHITE

STATE_CHANNELS: Final[int] = 4
ACTION_SIZE: Final[int] = BOARD_SIZE * BOARD_SIZE
STATE_SHAPE: Final[tuple[int, int, int]] = (STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
POLICY_SHAPE: Final[tuple[int]] = (ACTION_SIZE,)
VALUE_SHAPE: Final[tuple[int]] = (1,)
STATE_DTYPE: Final[np.dtype[np.float32]] = np.dtype(np.float32)
POLICY_DTYPE: Final[np.dtype[np.float32]] = np.dtype(np.float32)


def _as_float32_array(
    array: np.ndarray | list[float] | tuple[float, ...],
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    """Convert an input into a float32 NumPy array with strict shape checking."""

    resolved = np.asarray(array, dtype=np.float32)
    if resolved.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {resolved.shape}.")
    if not np.isfinite(resolved).all():
        raise ValueError(f"{name} must contain only finite values.")
    return resolved.astype(np.float32, copy=False)


def _as_scalar_float32(value: float | np.floating[Any] | np.ndarray, *, name: str) -> np.float32:
    """Convert a scalar or single-value array into a finite float32 scalar."""

    resolved = np.asarray(value, dtype=np.float32)
    if resolved.shape not in {(), VALUE_SHAPE}:
        raise ValueError(f"{name} must be a scalar or have shape {VALUE_SHAPE}, got {resolved.shape}.")
    scalar = np.float32(resolved.reshape(-1)[0])
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


@dataclass(frozen=True)
class NetworkInputSpec:
    """Frozen description of the AlphaZero network input/output contract."""

    state_shape: tuple[int, int, int] = STATE_SHAPE
    state_dtype: np.dtype[np.float32] = STATE_DTYPE
    policy_shape: tuple[int] = POLICY_SHAPE
    value_shape: tuple[int] = VALUE_SHAPE

    def __post_init__(self) -> None:
        """Validate that the declared shapes match the current environment contract."""

        if self.state_shape != STATE_SHAPE:
            raise ValueError(f"state_shape must be {STATE_SHAPE}, got {self.state_shape}.")
        if np.dtype(self.state_dtype) != STATE_DTYPE:
            raise ValueError(f"state_dtype must be {STATE_DTYPE}, got {self.state_dtype}.")
        if self.policy_shape != POLICY_SHAPE:
            raise ValueError(f"policy_shape must be {POLICY_SHAPE}, got {self.policy_shape}.")
        if self.value_shape != VALUE_SHAPE:
            raise ValueError(f"value_shape must be {VALUE_SHAPE}, got {self.value_shape}.")


@dataclass(frozen=True)
class PolicyValueOutputSpec:
    """Typed container for raw network outputs before sampling or masking."""

    policy_logits: np.ndarray
    value: float | np.floating[Any] | np.ndarray

    def __post_init__(self) -> None:
        """Validate raw policy logits and value prediction shapes."""

        logits = _as_float32_array(self.policy_logits, expected_shape=POLICY_SHAPE, name="policy_logits")
        scalar_value = _as_scalar_float32(self.value, name="value")
        object.__setattr__(self, "policy_logits", logits)
        object.__setattr__(self, "value", scalar_value)


@dataclass(frozen=True)
class GameStepSample:
    """Single training sample aligned to the side-to-move perspective."""

    state: np.ndarray
    policy_target: np.ndarray
    value_target: float | np.floating[Any] | np.ndarray
    player_to_move: int
    move_index: int
    action_taken: int | None = None
    game_id: str | None = None

    def __post_init__(self) -> None:
        """Validate state, target tensors, and metadata consistency."""

        state = _as_float32_array(self.state, expected_shape=STATE_SHAPE, name="state")
        policy_target = _as_float32_array(
            self.policy_target,
            expected_shape=POLICY_SHAPE,
            name="policy_target",
        )
        value_target = _as_scalar_float32(self.value_target, name="value_target")

        if not np.isclose(float(policy_target.sum()), 1.0, atol=1e-5):
            raise ValueError(f"policy_target must sum to 1.0, got {float(policy_target.sum()):.6f}.")
        if np.any(policy_target < 0.0):
            raise ValueError("policy_target must not contain negative probabilities.")
        if not -1.0 <= float(value_target) <= 1.0:
            raise ValueError(f"value_target must be in [-1.0, 1.0], got {float(value_target):.6f}.")
        if self.player_to_move not in {BLACK, WHITE}:
            raise ValueError(f"player_to_move must be BLACK({BLACK}) or WHITE({WHITE}).")
        if self.move_index < 0:
            raise ValueError("move_index must be non-negative.")
        if self.action_taken is not None and not 0 <= self.action_taken < ACTION_SIZE:
            raise ValueError(f"action_taken must be in [0, {ACTION_SIZE - 1}] when provided.")

        object.__setattr__(self, "state", state)
        object.__setattr__(self, "policy_target", policy_target)
        object.__setattr__(self, "value_target", value_target)


@dataclass(frozen=True)
class GameRecord:
    """Container for a full game's move history and derived training samples.

    For `human_play` sources, upstream data collection can choose to include
    only AI-turn samples while still keeping the full move list here.
    """

    game_id: str
    moves: tuple[int, ...] | list[int]
    winner: int
    source: str
    samples: tuple[GameStepSample, ...] | list[GameStepSample]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate winner/source metadata and normalize sequence fields."""

        normalized_moves = tuple(int(move) for move in self.moves)
        normalized_samples = tuple(self.samples)
        if not self.game_id:
            raise ValueError("game_id must be a non-empty string.")
        if self.winner not in {BLACK, WHITE, DRAW}:
            raise ValueError(f"winner must be BLACK({BLACK}), WHITE({WHITE}), or DRAW({DRAW}).")
        if self.source not in {"self_play", "human_play", "evaluation"}:
            raise ValueError("source must be one of {'self_play', 'human_play', 'evaluation'}.")
        for move in normalized_moves:
            if not 0 <= move < ACTION_SIZE:
                raise ValueError(f"moves must contain actions in [0, {ACTION_SIZE - 1}].")
        for sample in normalized_samples:
            if not isinstance(sample, GameStepSample):
                raise TypeError("samples must contain only GameStepSample instances.")
            if sample.game_id is not None and sample.game_id != self.game_id:
                raise ValueError("All sample.game_id values must match GameRecord.game_id when provided.")

        object.__setattr__(self, "moves", normalized_moves)
        object.__setattr__(self, "samples", normalized_samples)
        object.__setattr__(self, "metadata", dict(self.metadata))

