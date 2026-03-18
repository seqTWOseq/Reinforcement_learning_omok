"""Tests for the pattern-based Athenan heuristic evaluator."""

from __future__ import annotations

import numpy as np

from gomoku_ai.negamax_athenan.eval import GreedyHeuristicEvaluator
from gomoku_ai.negamax_athenan.search import negamax
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


def test_open_four_scores_higher_than_open_three() -> None:
    evaluator = GreedyHeuristicEvaluator()

    open_four_env = GomokuEnv()
    _place_stones(
        open_four_env,
        black_coords=[(7, 4), (7, 5), (7, 6), (7, 7)],
        white_coords=[],
        current_player=BLACK,
    )
    open_three_env = GomokuEnv()
    _place_stones(
        open_three_env,
        black_coords=[(7, 5), (7, 6), (7, 7)],
        white_coords=[],
        current_player=BLACK,
    )

    assert evaluator.score_patterns_for_player(open_four_env, BLACK) > evaluator.score_patterns_for_player(
        open_three_env,
        BLACK,
    )


def test_closed_four_scores_higher_than_closed_three() -> None:
    evaluator = GreedyHeuristicEvaluator()

    closed_four_env = GomokuEnv()
    _place_stones(
        closed_four_env,
        black_coords=[(7, 4), (7, 5), (7, 6), (7, 7)],
        white_coords=[(7, 3)],
        current_player=BLACK,
    )
    closed_three_env = GomokuEnv()
    _place_stones(
        closed_three_env,
        black_coords=[(7, 5), (7, 6), (7, 7)],
        white_coords=[(7, 4)],
        current_player=BLACK,
    )

    assert evaluator.score_patterns_for_player(closed_four_env, BLACK) > evaluator.score_patterns_for_player(
        closed_three_env,
        BLACK,
    )


def test_five_scores_higher_than_open_four() -> None:
    evaluator = GreedyHeuristicEvaluator()

    five_env = GomokuEnv()
    _place_stones(
        five_env,
        black_coords=[(7, 3), (7, 4), (7, 5), (7, 6), (7, 7)],
        white_coords=[],
        current_player=BLACK,
    )
    open_four_env = GomokuEnv()
    _place_stones(
        open_four_env,
        black_coords=[(7, 4), (7, 5), (7, 6), (7, 7)],
        white_coords=[],
        current_player=BLACK,
    )

    assert evaluator.score_patterns_for_player(five_env, BLACK) > evaluator.score_patterns_for_player(
        open_four_env,
        BLACK,
    )


def test_opponent_open_four_threat_is_penalized_harder_than_simple_connection() -> None:
    evaluator = GreedyHeuristicEvaluator()

    safe_env = GomokuEnv()
    _place_stones(
        safe_env,
        black_coords=[(7, 5), (7, 6), (7, 7)],
        white_coords=[],
        current_player=BLACK,
    )
    threat_env = GomokuEnv()
    _place_stones(
        threat_env,
        black_coords=[(7, 5), (7, 6), (7, 7)],
        white_coords=[(5, 4), (5, 5), (5, 6), (5, 7)],
        current_player=BLACK,
    )

    safe_score = evaluator.evaluate_for_player(safe_env, BLACK)
    threat_score = evaluator.evaluate_for_player(threat_env, BLACK)

    assert threat_score < safe_score
    assert threat_score < 0.0


def test_pattern_counting_is_deterministic() -> None:
    evaluator = GreedyHeuristicEvaluator()
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 4), (7, 5), (7, 6), (7, 7)],
        white_coords=[(3, 3)],
        current_player=BLACK,
    )

    score_a = evaluator.evaluate_for_player(env, BLACK)
    score_b = evaluator.evaluate_for_player(env, BLACK)
    patterns_a = evaluator.count_patterns_for_player(env, BLACK)
    patterns_b = evaluator.count_patterns_for_player(env, BLACK)

    assert score_a == score_b
    assert patterns_a == patterns_b


def test_negamax_depth_zero_uses_pattern_evaluator_directly() -> None:
    evaluator = GreedyHeuristicEvaluator()
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 4), (7, 5), (7, 6)],
        white_coords=[(3, 3)],
        current_player=BLACK,
    )

    score, action = negamax(env, depth=0, evaluator=evaluator)

    assert action is None
    assert score == evaluator.evaluate_for_player(env, BLACK)


def test_evaluator_methods_do_not_mutate_original_env() -> None:
    evaluator = GreedyHeuristicEvaluator()
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7), (7, 8)],
        white_coords=[(0, 0)],
        current_player=BLACK,
    )
    board_before = env.board.copy()
    state_before = (
        env.current_player,
        env.last_move,
        env.done,
        env.winner,
        env.move_count,
    )

    _ = evaluator.evaluate_for_player(env, BLACK)
    _ = evaluator.score_patterns_for_player(env, BLACK)
    _ = evaluator.score_action_for_player(env, env.coord_to_action(6, 7), BLACK)

    assert np.array_equal(env.board, board_before)
    assert state_before == (
        env.current_player,
        env.last_move,
        env.done,
        env.winner,
        env.move_count,
    )
