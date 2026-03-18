"""Tests for the stage-5 pure negamax searcher."""

from __future__ import annotations

import numpy as np

from gomoku_ai.negamax_athenan.eval import GreedyHeuristicEvaluator
from gomoku_ai.negamax_athenan.search import AthenanNegamaxSearcher, negamax
from gomoku_ai.env import BLACK, DRAW, GomokuEnv, WHITE


class BaitPerspectiveEvaluator(GreedyHeuristicEvaluator):
    """Custom evaluator used to show why deeper negamax can outperform depth-1."""

    def evaluate_for_player(self, env: GomokuEnv, player: int) -> float:
        if env.done:
            return super().evaluate_for_player(env, player)

        occupant = int(env.board[0, 1])
        if occupant == int(player):
            return 100.0
        if occupant in {BLACK, WHITE}:
            return -100.0
        return 0.0


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


def test_negamax_takes_immediate_win() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 3), (7, 4), (7, 5), (7, 6)],
        white_coords=[(7, 2), (0, 0)],
        current_player=BLACK,
    )

    result = AthenanNegamaxSearcher(max_depth=1).search(env)

    assert result.best_action == env.coord_to_action(7, 7)
    assert result.root_value > 0.0


def test_negamax_blocks_opponent_immediate_win() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(8, 2), (0, 0)],
        white_coords=[(8, 3), (8, 4), (8, 5), (8, 6)],
        current_player=BLACK,
    )

    result = AthenanNegamaxSearcher(max_depth=2).search(env)

    assert result.best_action == env.coord_to_action(8, 7)


def test_negamax_depth_two_can_override_depth_one_leaf_bias() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(8, 2), (0, 0)],
        white_coords=[(8, 3), (8, 4), (8, 5), (8, 6)],
        current_player=BLACK,
    )
    evaluator = BaitPerspectiveEvaluator()

    shallow = AthenanNegamaxSearcher(evaluator=evaluator, max_depth=1, candidate_radius=2).search(env)
    deep = AthenanNegamaxSearcher(evaluator=evaluator, max_depth=2, candidate_radius=2).search(env)

    assert shallow.best_action == env.coord_to_action(0, 1)
    assert deep.best_action == env.coord_to_action(8, 7)


def test_negamax_terminal_position_returns_score_without_action() -> None:
    env = GomokuEnv()
    env.reset()
    env.done = True
    env.winner = BLACK
    env.current_player = BLACK

    score, action = negamax(env, depth=2)

    assert action is None
    assert score > 0.0


def test_negamax_draw_terminal_returns_zero() -> None:
    env = GomokuEnv()
    env.reset()
    env.done = True
    env.winner = DRAW

    score, action = negamax(env, depth=3)

    assert action is None
    assert score == 0.0


def test_negamax_does_not_mutate_original_env() -> None:
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

    _ = AthenanNegamaxSearcher(max_depth=2).search(env)

    assert np.array_equal(env.board, board_before)
    assert env.current_player == current_player_before
    assert env.last_move == last_move_before
    assert env.winner == winner_before
    assert env.done == done_before
    assert env.move_count == move_count_before


def test_negamax_searcher_returns_legal_action() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(8, 8)],
        current_player=BLACK,
    )

    result = AthenanNegamaxSearcher(max_depth=2, max_candidates=8).search(env)

    assert result.best_action in env.get_legal_actions()


def test_negamax_passes_radius_and_max_candidates_to_generator() -> None:
    import gomoku_ai.negamax_athenan.search.search_core as minimax_module

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(8, 8)],
        current_player=BLACK,
    )

    original = minimax_module.generate_candidate_actions
    seen_calls: list[tuple[int, int | None]] = []

    def fake_generate_candidate_actions(
        env_arg: GomokuEnv,
        *,
        radius: int = 2,
        max_candidates: int | None = None,
        evaluator: GreedyHeuristicEvaluator | None = None,
        player: int | None = None,
    ) -> list[int]:
        seen_calls.append((radius, max_candidates))
        legal = env_arg.get_legal_actions()
        return legal[:2]

    minimax_module.generate_candidate_actions = fake_generate_candidate_actions
    try:
        _ = AthenanNegamaxSearcher(max_depth=1, candidate_radius=1, max_candidates=2).search(env)
    finally:
        minimax_module.generate_candidate_actions = original

    assert seen_calls == [(1, 2)]


def test_negamax_is_deterministic_on_same_position() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(0, 0)],
        current_player=BLACK,
    )
    searcher = AthenanNegamaxSearcher(max_depth=2, max_candidates=10)

    result_a = searcher.search(env)
    result_b = searcher.search(env)

    assert result_a.best_action == result_b.best_action
