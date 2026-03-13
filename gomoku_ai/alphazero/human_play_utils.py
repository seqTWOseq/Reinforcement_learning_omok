"""Helper utilities for human-vs-AlphaZero games.

Human play is intentionally different from self-play:
- the AI uses evaluation-style search by default
- root noise is disabled
- temperature is deterministic by default
- only AI turns are converted into training samples in this stage
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence
from uuid import uuid4

from gomoku_ai.alphazero.mcts import MCTSConfig
from gomoku_ai.alphazero.specs import GameStepSample
from gomoku_ai.alphazero.utils import winner_to_value_target
from gomoku_ai.env import BLACK, DRAW, WHITE

if TYPE_CHECKING:
    from gomoku_ai.alphazero.human_play import HumanAITurnRecord, HumanPlayConfig

REQUIRED_HUMAN_PLAY_METADATA_KEYS = (
    "human_color",
    "ai_color",
    "num_moves",
    "ai_temperature",
    "use_root_noise",
)


def normalize_human_color_choice(color_choice: str | int) -> int | str:
    """Normalize human color choice into `BLACK`, `WHITE`, or `select_each_game`."""

    if isinstance(color_choice, str):
        normalized = color_choice.strip().lower()
        if normalized in {"select_each_game", "select-each-game", "select"}:
            return "select_each_game"
        if normalized in {"black", "b"}:
            return BLACK
        if normalized in {"white", "w"}:
            return WHITE
    elif isinstance(color_choice, int):
        if color_choice in {BLACK, WHITE}:
            return color_choice

    raise ValueError(
        "color_choice must be BLACK/WHITE or one of "
        "{'black', 'white', 'select_each_game'}."
    )


def resolve_human_and_ai_colors(
    human_color: str | int | None,
    default_human_color: str | int,
    *,
    selection_callback: Callable[[], str | int] | None = None,
) -> tuple[int, int]:
    """Resolve explicit or interactive human/AI color assignments."""

    requested_color = default_human_color if human_color is None else human_color
    normalized_choice = normalize_human_color_choice(requested_color)
    if normalized_choice == "select_each_game":
        if selection_callback is None:
            raise ValueError(
                "human_color resolved to 'select_each_game' but no color selection callback was provided."
            )
        normalized_choice = normalize_human_color_choice(selection_callback())
        if normalized_choice == "select_each_game":
            raise ValueError("Interactive color selection must resolve to BLACK or WHITE.")

    human_player = int(normalized_choice)
    ai_player = WHITE if human_player == BLACK else BLACK
    return human_player, ai_player


def build_human_play_game_id(prefix: str) -> str:
    """Build a UUID-backed human-play game id with the required prefix."""

    if not prefix or not prefix.strip():
        raise ValueError("prefix must be a non-empty string.")
    return f"{prefix}-{uuid4()}"


def build_human_play_mcts_config(config: "HumanPlayConfig") -> MCTSConfig:
    """Build the AI's evaluation-style MCTS config for human games."""

    default_mcts_config = MCTSConfig()
    return MCTSConfig(
        num_simulations=config.ai_num_simulations,
        c_puct=default_mcts_config.c_puct,
        dirichlet_alpha=default_mcts_config.dirichlet_alpha,
        dirichlet_epsilon=default_mcts_config.dirichlet_epsilon,
        add_root_noise=config.use_root_noise,
        temperature=config.ai_temperature,
    )


def finalize_human_ai_turn_records(
    turn_records: Sequence["HumanAITurnRecord"],
    winner: int,
    game_id: str,
) -> list[GameStepSample]:
    """Convert validated AI-turn records into training-ready `GameStepSample`s."""

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
