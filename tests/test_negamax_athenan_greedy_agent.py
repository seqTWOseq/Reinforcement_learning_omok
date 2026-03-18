"""Tests for the stage-3 greedy heuristic Athenan agent."""

from __future__ import annotations

import numpy as np

from gomoku_ai.negamax_athenan.agent import AthenanGreedyHeuristicAgent
from gomoku_ai.env import BLACK, GomokuEnv, WHITE


def _place_stones(
    env: GomokuEnv,
    black_coords: list[tuple[int, int]],
    white_coords: list[tuple[int, int]],
    *,
    current_player: int,
) -> None:
    env.reset()
    for row, col in black_coords:
        env.board[row, col] = BLACK
    for row, col in white_coords:
        env.board[row, col] = WHITE
    env.current_player = current_player
    env.last_move = None
    env.done = False
    env.winner = None
    env.move_count = len(black_coords) + len(white_coords)


def test_greedy_agent_takes_immediate_win() -> None:
    """Immediate winning moves must outrank all non-winning alternatives."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 3), (7, 4), (7, 5), (7, 6)],
        white_coords=[(7, 2), (0, 0)],
        current_player=BLACK,
    )

    action = AthenanGreedyHeuristicAgent().select_action(env)

    assert action == env.coord_to_action(7, 7)


def test_greedy_agent_blocks_opponent_immediate_win() -> None:
    """Forced defensive blocks must be chosen before quieter heuristic moves."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(8, 2), (0, 0)],
        white_coords=[(8, 3), (8, 4), (8, 5), (8, 6)],
        current_player=BLACK,
    )

    action = AthenanGreedyHeuristicAgent().select_action(env)

    assert action == env.coord_to_action(8, 7)


def test_greedy_agent_returns_legal_action() -> None:
    """Selected moves should always come from the legal action set."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(7, 8)],
        current_player=BLACK,
    )

    action = AthenanGreedyHeuristicAgent().select_action(env)

    assert action in env.get_legal_actions()


def test_greedy_agent_raises_on_terminal_state() -> None:
    """Terminal positions should not produce another action."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 3), (7, 4), (7, 5), (7, 6)],
        white_coords=[(6, 0), (6, 1), (6, 2), (6, 3)],
        current_player=BLACK,
    )
    env.apply_move(env.coord_to_action(7, 7))

    try:
        AthenanGreedyHeuristicAgent().select_action(env)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError on terminal position.")


def test_greedy_agent_is_deterministic_on_same_position() -> None:
    """Repeated calls on the same state should return the same action."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(0, 0)],
        current_player=BLACK,
    )
    agent = AthenanGreedyHeuristicAgent()

    action_a = agent.select_action(env)
    action_b = agent.select_action(env)

    assert action_a == action_b


def test_greedy_agent_does_not_mutate_original_env() -> None:
    """Child-state scoring should leave the caller's environment untouched."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(7, 8)],
        current_player=BLACK,
    )
    board_before = env.board.copy()
    current_player_before = env.current_player
    last_move_before = env.last_move
    winner_before = env.winner
    done_before = env.done
    move_count_before = env.move_count

    _ = AthenanGreedyHeuristicAgent().select_action(env)

    assert np.array_equal(env.board, board_before)
    assert env.current_player == current_player_before
    assert env.last_move == last_move_before
    assert env.winner == winner_before
    assert env.done == done_before
    assert env.move_count == move_count_before


def test_greedy_agent_prefers_center_on_empty_board() -> None:
    """On an empty board the center should be the highest-value opening move."""

    env = GomokuEnv()
    env.reset()

    action = AthenanGreedyHeuristicAgent().select_action(env)

    assert action == env.coord_to_action(env.board_size // 2, env.board_size // 2)


def test_greedy_agent_strengthens_nearby_connection() -> None:
    """Without tactical urgency the heuristic should prefer extending its own group."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(0, 0)],
        current_player=BLACK,
    )

    action = AthenanGreedyHeuristicAgent().select_action(env)

    assert action == env.coord_to_action(6, 7)
