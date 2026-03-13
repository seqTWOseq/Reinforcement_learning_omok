"""Helper functions for AlphaZero-compatible state, policy, and value handling."""

from __future__ import annotations

from typing import Any

import numpy as np

from gomoku_ai.alphazero.specs import (
    ACTION_SIZE,
    POLICY_DTYPE,
    POLICY_SHAPE,
    STATE_DTYPE,
    STATE_SHAPE,
    GameStepSample,
)
from gomoku_ai.env import BLACK, DRAW, GomokuEnv, WHITE


def _ensure_policy_shape(array: np.ndarray | list[float], *, name: str) -> np.ndarray:
    """Convert an array-like policy vector into a validated float32 array."""

    resolved = np.asarray(array, dtype=np.float32)
    if resolved.shape != POLICY_SHAPE:
        raise ValueError(f"{name} must have shape {POLICY_SHAPE}, got {resolved.shape}.")
    if not np.isfinite(resolved).all():
        raise ValueError(f"{name} must contain only finite values.")
    return resolved.astype(POLICY_DTYPE, copy=False)


def _ensure_valid_move_mask(valid_moves: np.ndarray | list[bool]) -> np.ndarray:
    """Convert a valid-move mask into the canonical boolean policy shape."""

    resolved = np.asarray(valid_moves, dtype=bool)
    if resolved.shape != POLICY_SHAPE:
        raise ValueError(f"valid_moves must have shape {POLICY_SHAPE}, got {resolved.shape}.")
    if not resolved.any():
        raise ValueError("valid_moves must contain at least one legal move.")
    return resolved.astype(bool, copy=False)


def winner_to_value_target(winner: int, player_to_move: int) -> float:
    """Map a terminal winner to a value target from `player_to_move` perspective."""

    if player_to_move not in {BLACK, WHITE}:
        raise ValueError(f"player_to_move must be BLACK({BLACK}) or WHITE({WHITE}).")
    if winner == DRAW:
        return 0.0
    if winner not in {BLACK, WHITE}:
        raise ValueError(f"winner must be BLACK({BLACK}), WHITE({WHITE}), or DRAW({DRAW}).")
    return 1.0 if winner == player_to_move else -1.0


def normalize_visit_counts(visit_counts: np.ndarray) -> np.ndarray:
    """Normalize MCTS visit counts into a float32 policy target that sums to 1.0."""

    counts = _ensure_policy_shape(visit_counts, name="visit_counts")
    if np.any(counts < 0.0):
        raise ValueError("visit_counts must not contain negative values.")
    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError("visit_counts must sum to a positive value.")
    normalized = (counts / total).astype(np.float32, copy=False)
    if not np.isclose(float(normalized.sum()), 1.0, atol=1e-5):
        raise ValueError("Normalized visit counts must sum to 1.0.")
    return normalized


def mask_policy_logits(policy_logits: np.ndarray, valid_moves: np.ndarray) -> np.ndarray:
    """Mask invalid moves by replacing their logits with negative infinity."""

    logits = _ensure_policy_shape(policy_logits, name="policy_logits")
    mask = _ensure_valid_move_mask(valid_moves)

    masked = np.full(POLICY_SHAPE, np.float32(-np.inf), dtype=np.float32)
    masked[mask] = logits[mask]

    if not np.isfinite(masked[mask]).all():
        raise ValueError("Valid policy logits must remain finite after masking.")
    return masked


def policy_logits_to_probs(policy_logits: np.ndarray, valid_moves: np.ndarray) -> np.ndarray:
    """Convert raw policy logits into masked move probabilities."""

    masked_logits = mask_policy_logits(policy_logits, valid_moves)
    valid_mask = np.isfinite(masked_logits)
    stabilized = masked_logits[valid_mask] - np.max(masked_logits[valid_mask])
    exp_values = np.exp(stabilized, dtype=np.float32)
    denominator = float(exp_values.sum())
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("Masked policy logits produced an invalid softmax denominator.")

    probabilities = np.zeros(POLICY_SHAPE, dtype=np.float32)
    probabilities[valid_mask] = exp_values / denominator

    if not np.isclose(float(probabilities.sum()), 1.0, atol=1e-5):
        raise ValueError("Policy probabilities must sum to 1.0.")
    if np.any(probabilities[~valid_mask] != 0.0):
        raise ValueError("Invalid moves must have zero probability.")
    return probabilities


def build_game_step_sample(
    env: GomokuEnv,
    policy_target: np.ndarray,
    value_target: float,
    move_index: int,
    action_taken: int | None = None,
    game_id: str | None = None,
) -> GameStepSample:
    """Build a validated `GameStepSample` from the current environment state.

    The helper trusts `env.encode_state()` as the canonical state encoder.
    Human-play data pipelines can call this only on AI turns when they want to
    store AI-perspective samples exclusively.
    """

    if not isinstance(env, GomokuEnv):
        raise TypeError("env must be a GomokuEnv instance.")
    state = np.asarray(env.encode_state(), dtype=np.float32)
    if state.shape != STATE_SHAPE:
        raise ValueError(f"env.encode_state() must return shape {STATE_SHAPE}, got {state.shape}.")
    if state.dtype != STATE_DTYPE:
        state = state.astype(np.float32, copy=False)

    return GameStepSample(
        state=state,
        policy_target=policy_target,
        value_target=value_target,
        player_to_move=env.current_player,
        move_index=move_index,
        action_taken=action_taken,
        game_id=game_id,
    )
