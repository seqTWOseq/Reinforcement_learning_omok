"""Helper utilities for AlphaZero self-play game generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
from uuid import uuid4

from gomoku_ai.alphazero.specs import GameStepSample
from gomoku_ai.alphazero.utils import winner_to_value_target
from gomoku_ai.env import BLACK, DRAW, WHITE

if TYPE_CHECKING:
    from gomoku_ai.alphazero.self_play import SelfPlayConfig, SelfPlayTurnRecord

REQUIRED_SELF_PLAY_METADATA_KEYS = (
    "num_moves",
    "use_root_noise",
    "opening_temperature_moves",
)


def get_temperature_for_move(move_index: int, config: "SelfPlayConfig") -> float:
    """Return the configured sampling temperature for a given move index."""

    if move_index < 0:
        raise ValueError("move_index must be non-negative.")
    if config.opening_temperature_moves < 0:
        raise ValueError("opening_temperature_moves must be non-negative.")
    if config.opening_temperature < 0.0 or config.late_temperature < 0.0:
        raise ValueError("Temperatures must be non-negative.")
    return config.opening_temperature if move_index < config.opening_temperature_moves else config.late_temperature


def build_self_play_game_id(prefix: str) -> str:
    """Build a UUID-backed self-play game id with the required prefix."""

    if not prefix or not prefix.strip():
        raise ValueError("prefix must be a non-empty string.")
    return f"{prefix}-{uuid4()}"


def finalize_turn_records(
    turn_records: Sequence["SelfPlayTurnRecord"],
    winner: int,
    game_id: str,
) -> list[GameStepSample]:
    """Convert validated `SelfPlayTurnRecord` objects into `GameStepSample` objects.

    Expected input:
    - one record per played move
    - `state` captured before the action was applied
    - `policy_target` already formed from root visit counts with that move's
      temperature schedule applied
    """

    if winner not in {BLACK, WHITE, DRAW}:
        raise ValueError(f"winner must be BLACK({BLACK}), WHITE({WHITE}), or DRAW({DRAW}).")
    if not game_id:
        raise ValueError("game_id must be a non-empty string.")

    samples: list[GameStepSample] = []
    for turn_record in turn_records:
        value_target = winner_to_value_target(winner, turn_record.player_to_move)
        samples.append(
            GameStepSample(
                state=turn_record.state,
                policy_target=turn_record.policy_target,
                value_target=value_target,
                player_to_move=turn_record.player_to_move,
                move_index=turn_record.move_index,
                action_taken=turn_record.action_taken,
                game_id=game_id,
            )
        )
    return samples
