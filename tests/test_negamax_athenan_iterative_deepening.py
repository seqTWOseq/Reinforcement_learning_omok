"""Tests for iterative deepening on the alpha-beta negamax searcher."""

from __future__ import annotations

import math
import numpy as np

from gomoku_ai.negamax_athenan.search import (
    run_iterative_deepening_search,
    run_negamax_search,
)
from gomoku_ai.negamax_athenan.search import AthenanNegamaxSearcher
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


def test_iterative_deepening_runs_depths_up_to_max_depth() -> None:
    import gomoku_ai.negamax_athenan.search.search_core as minimax_module

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(8, 8)],
        current_player=BLACK,
    )

    original = minimax_module.run_negamax_search
    seen_depths: list[int] = []

    def recording_run_negamax_search(
        env_arg: GomokuEnv,
        *,
        depth: int,
        evaluator=None,
        radius: int = 2,
        max_candidates: int | None = None,
        use_alpha_beta: bool = True,
        use_transposition_table: bool = True,
        preferred_action: int | None = None,
        transposition_table=None,
    ):
        seen_depths.append(depth)
        return original(
            env_arg,
            depth=depth,
            evaluator=evaluator,
            radius=radius,
            max_candidates=max_candidates,
            use_alpha_beta=use_alpha_beta,
            use_transposition_table=use_transposition_table,
            preferred_action=preferred_action,
            transposition_table=transposition_table,
        )

    minimax_module.run_negamax_search = recording_run_negamax_search
    try:
        _ = run_iterative_deepening_search(env, max_depth=3, radius=2, max_candidates=10, use_alpha_beta=True)
    finally:
        minimax_module.run_negamax_search = original

    assert seen_depths == [1, 2, 3]


def test_iterative_deepening_matches_deepest_completed_result() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(8, 8)],
        current_player=BLACK,
    )

    iterative = run_iterative_deepening_search(env, max_depth=3, radius=2, max_candidates=10, use_alpha_beta=True)
    direct = run_negamax_search(env, depth=3, radius=2, max_candidates=10, use_alpha_beta=True)

    assert iterative.best_action == direct.best_action
    assert math.isclose(iterative.value, direct.value, rel_tol=0.0, abs_tol=1e-9)
    assert iterative.principal_variation == direct.principal_variation


def test_iterative_deepening_does_not_mutate_original_env() -> None:
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

    _ = run_iterative_deepening_search(env, max_depth=3, radius=2, max_candidates=10, use_alpha_beta=True)

    assert np.array_equal(env.board, board_before)
    assert state_before == (
        env.current_player,
        env.last_move,
        env.done,
        env.winner,
        env.move_count,
    )


def test_iterative_deepening_searcher_matches_single_depth_alpha_beta_semantics() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(8, 8)],
        current_player=BLACK,
    )

    direct = AthenanNegamaxSearcher(
        max_depth=3,
        max_candidates=10,
        use_alpha_beta=True,
        use_iterative_deepening=False,
    ).search(env)
    iterative = AthenanNegamaxSearcher(
        max_depth=3,
        max_candidates=10,
        use_alpha_beta=True,
        use_iterative_deepening=True,
    ).search(env)

    assert iterative.best_action == direct.best_action
    assert math.isclose(iterative.root_value, direct.root_value, rel_tol=0.0, abs_tol=1e-9)
    assert iterative.principal_variation == direct.principal_variation


def test_iterative_deepening_is_deterministic() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(0, 0)],
        current_player=BLACK,
    )
    searcher = AthenanNegamaxSearcher(
        max_depth=3,
        max_candidates=10,
        use_alpha_beta=True,
        use_iterative_deepening=True,
    )

    result_a = searcher.search(env)
    result_b = searcher.search(env)

    assert result_a.best_action == result_b.best_action
