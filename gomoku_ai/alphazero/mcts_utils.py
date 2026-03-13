"""Utility helpers for AlphaZero MCTS state semantics and terminal values."""

from __future__ import annotations

from gomoku_ai.alphazero.utils import winner_to_value_target
from gomoku_ai.env import BLACK, DRAW, GomokuEnv, WHITE


def opponent_of(player: int) -> int:
    """Return the opposing player color."""

    if player == BLACK:
        return WHITE
    if player == WHITE:
        return BLACK
    raise ValueError(f"player must be BLACK({BLACK}) or WHITE({WHITE}), got {player!r}.")


def resolve_env_player_to_move(env: GomokuEnv) -> int:
    """Resolve the logical node perspective for an environment state.

    For non-terminal states this is simply `env.current_player`.

    For terminal states, `GomokuEnv.apply_move()` leaves `env.current_player`
    equal to the player who just moved. MCTS nodes instead store the logical
    "next player if the game were to continue" perspective, so terminal states
    use the opponent of `env.current_player`.
    """

    if not isinstance(env, GomokuEnv):
        raise TypeError("env must be a GomokuEnv instance.")
    return env.current_player if not env.done else opponent_of(env.current_player)


def terminal_value_for_player(env: GomokuEnv, player_to_move: int) -> float:
    """Return terminal value from the provided player's perspective.

    Args:
        env: Terminal Gomoku environment.
        player_to_move: Logical node perspective. In MCTS this should be the
            node's `player_to_move`, not `env.current_player` for terminal
            states.
    """

    if not isinstance(env, GomokuEnv):
        raise TypeError("env must be a GomokuEnv instance.")
    if not env.done:
        raise ValueError("terminal_value_for_player requires a terminal environment.")
    if env.winner is None:
        raise ValueError("Terminal environment must have a non-None winner value.")
    if env.winner == DRAW:
        return 0.0
    return winner_to_value_target(env.winner, player_to_move)
