"""Tests for Athenan candidate pruning and move generation."""

from __future__ import annotations

import numpy as np

from gomoku_ai.negamax_athenan.eval import GreedyHeuristicEvaluator
from gomoku_ai.negamax_athenan.search import generate_candidate_actions
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


def test_generate_candidate_actions_empty_board_returns_center() -> None:
    env = GomokuEnv()
    env.reset()

    candidates = generate_candidate_actions(env, radius=2)

    assert candidates == [env.coord_to_action(env.board_size // 2, env.board_size // 2)]


def test_generate_candidate_actions_only_returns_nearby_moves() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(10, 10)],
        current_player=BLACK,
    )

    candidates = generate_candidate_actions(env, radius=1)
    stones = [(7, 7), (10, 10)]

    assert candidates
    for action in candidates:
        row, col = env.action_to_coord(action)
        assert any(abs(row - stone_row) <= 1 and abs(col - stone_col) <= 1 for stone_row, stone_col in stones)


def test_generate_candidate_actions_is_unique_and_legal() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7), (7, 8)],
        white_coords=[(8, 7)],
        current_player=BLACK,
    )

    candidates = generate_candidate_actions(env, radius=2)

    assert len(candidates) == len(set(candidates))
    assert all(action in env.get_legal_actions() for action in candidates)


def test_generate_candidate_actions_prioritizes_immediate_win() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 3), (7, 4), (7, 5), (7, 6)],
        white_coords=[(7, 2), (0, 0)],
        current_player=BLACK,
    )

    candidates = generate_candidate_actions(env, radius=2)

    assert candidates[0] == env.coord_to_action(7, 7)


def test_generate_candidate_actions_prioritizes_immediate_block() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(8, 2), (0, 0)],
        white_coords=[(8, 3), (8, 4), (8, 5), (8, 6)],
        current_player=BLACK,
    )

    candidates = generate_candidate_actions(env, radius=2)

    assert candidates[0] == env.coord_to_action(8, 7)


def test_generate_candidate_actions_respects_max_candidates() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(10, 10)],
        current_player=BLACK,
    )

    candidates = generate_candidate_actions(env, radius=2, max_candidates=5)

    assert len(candidates) == 5


def test_generate_candidate_actions_does_not_mutate_original_env() -> None:
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

    _ = generate_candidate_actions(env, radius=2)

    assert np.array_equal(env.board, board_before)
    assert env.current_player == current_player_before
    assert env.last_move == last_move_before
    assert env.winner == winner_before
    assert env.done == done_before
    assert env.move_count == move_count_before


def test_generate_candidate_actions_radius_two_superset_of_radius_one() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(0, 0)],
        current_player=BLACK,
    )

    radius_one = set(generate_candidate_actions(env, radius=1))
    radius_two = set(generate_candidate_actions(env, radius=2))

    assert radius_one < radius_two


def test_generate_candidate_actions_is_deterministic() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(0, 0)],
        current_player=BLACK,
    )

    candidates_a = generate_candidate_actions(env, radius=2)
    candidates_b = generate_candidate_actions(env, radius=2)

    assert candidates_a == candidates_b


def test_generate_candidate_actions_can_use_shared_evaluator() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(0, 0)],
        current_player=BLACK,
    )

    candidates = generate_candidate_actions(env, radius=2, evaluator=GreedyHeuristicEvaluator())

    assert candidates


def test_generate_candidate_actions_radius_zero_falls_back_to_ordered_legals() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(0, 0)],
        current_player=BLACK,
    )

    candidates = generate_candidate_actions(env, radius=0, max_candidates=5)

    assert len(candidates) == 5
    assert all(action in env.get_legal_actions() for action in candidates)
