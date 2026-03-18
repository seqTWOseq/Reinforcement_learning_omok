"""Tests for alpha-beta pruning on top of the stage-5 negamax core."""

from __future__ import annotations

import math
import numpy as np

from gomoku_ai.negamax_athenan.eval import GreedyHeuristicEvaluator
from gomoku_ai.negamax_athenan.search import (
    AthenanNegamaxSearcher,
    negamax,
    run_negamax_search,
)
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


def test_alphabeta_immediate_win_is_preserved() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 3), (7, 4), (7, 5), (7, 6)],
        white_coords=[(7, 2), (0, 0)],
        current_player=BLACK,
    )

    result = AthenanNegamaxSearcher(max_depth=2, use_alpha_beta=True).search(env)

    assert result.best_action == env.coord_to_action(7, 7)


def test_alphabeta_immediate_block_is_preserved() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(8, 2), (0, 0)],
        white_coords=[(8, 3), (8, 4), (8, 5), (8, 6)],
        current_player=BLACK,
    )

    result = AthenanNegamaxSearcher(max_depth=2, use_alpha_beta=True).search(env)

    assert result.best_action == env.coord_to_action(8, 7)


def test_alphabeta_terminal_position_returns_no_action() -> None:
    env = GomokuEnv()
    env.reset()
    env.done = True
    env.winner = BLACK
    env.current_player = BLACK

    score, action = negamax(env, depth=3, use_alpha_beta=True)
    search_result = AthenanNegamaxSearcher(max_depth=3, use_alpha_beta=True).search(env)

    assert action is None
    assert score > 0.0
    assert search_result.best_action == -1
    assert search_result.root_value > 0.0


def test_alphabeta_does_not_mutate_original_env() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(8, 8)],
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

    _ = AthenanNegamaxSearcher(max_depth=3, max_candidates=10, use_alpha_beta=True).search(env)

    assert np.array_equal(env.board, board_before)
    assert state_before == (
        env.current_player,
        env.last_move,
        env.done,
        env.winner,
        env.move_count,
    )


def test_alphabeta_matches_plain_negamax_semantics() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(8, 8)],
        current_player=BLACK,
    )

    plain = run_negamax_search(env, depth=3, radius=2, max_candidates=10, use_alpha_beta=False)
    ab = run_negamax_search(env, depth=3, radius=2, max_candidates=10, use_alpha_beta=True)

    assert plain.best_action == ab.best_action
    assert math.isclose(plain.value, ab.value, rel_tol=0.0, abs_tol=1e-9)
    assert plain.principal_variation == ab.principal_variation


def test_alphabeta_records_cutoffs_and_reduces_nodes() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(8, 8)],
        current_player=BLACK,
    )

    plain = run_negamax_search(env, depth=3, radius=2, max_candidates=10, use_alpha_beta=False)
    ab = run_negamax_search(env, depth=3, radius=2, max_candidates=10, use_alpha_beta=True)

    assert ab.cutoffs > 0
    assert ab.pruned_branches > 0
    assert ab.nodes < plain.nodes


def test_alphabeta_passes_radius_and_max_candidates_to_generator() -> None:
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
        return env_arg.get_legal_actions()[:2]

    minimax_module.generate_candidate_actions = fake_generate_candidate_actions
    try:
        _ = AthenanNegamaxSearcher(
            max_depth=1,
            candidate_radius=1,
            max_candidates=2,
            use_alpha_beta=True,
        ).search(env)
    finally:
        minimax_module.generate_candidate_actions = original

    assert seen_calls == [(1, 2)]


def test_alphabeta_is_deterministic() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(0, 0)],
        current_player=BLACK,
    )
    searcher = AthenanNegamaxSearcher(max_depth=3, max_candidates=10, use_alpha_beta=True)

    result_a = searcher.search(env)
    result_b = searcher.search(env)

    assert result_a.best_action == result_b.best_action
