"""Tests for transposition-table support in the negamax search core."""

from __future__ import annotations

import math
import numpy as np

from gomoku_ai.negamax_athenan.search.search_core import (
    _make_transposition_key,
    _resolve_tt_flag,
)
from gomoku_ai.negamax_athenan.search import (
    AthenanNegamaxSearcher,
    run_iterative_deepening_search,
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


def test_transposition_table_preserves_best_action_and_value() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(8, 8)],
        current_player=BLACK,
    )

    without_tt = run_negamax_search(
        env,
        depth=4,
        radius=2,
        max_candidates=10,
        use_alpha_beta=True,
        use_transposition_table=False,
    )
    with_tt = run_negamax_search(
        env,
        depth=4,
        radius=2,
        max_candidates=10,
        use_alpha_beta=True,
        use_transposition_table=True,
    )

    assert without_tt.best_action == with_tt.best_action
    assert math.isclose(without_tt.value, with_tt.value, rel_tol=0.0, abs_tol=1e-9)


def test_transposition_table_reduces_nodes_on_transposing_position() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(8, 8)],
        current_player=BLACK,
    )

    without_tt = run_negamax_search(
        env,
        depth=4,
        radius=2,
        max_candidates=10,
        use_alpha_beta=True,
        use_transposition_table=False,
    )
    with_tt = run_negamax_search(
        env,
        depth=4,
        radius=2,
        max_candidates=10,
        use_alpha_beta=True,
        use_transposition_table=True,
    )

    assert with_tt.tt_hits > 0
    assert with_tt.nodes < without_tt.nodes


def test_iterative_deepening_reuses_same_transposition_table_instance() -> None:
    import gomoku_ai.negamax_athenan.search.search_core as minimax_module

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(8, 8)],
        current_player=BLACK,
    )

    original = minimax_module.run_negamax_search
    seen_table_ids: list[int] = []

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
        if transposition_table is not None:
            seen_table_ids.append(id(transposition_table))
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
        result = run_iterative_deepening_search(
            env,
            max_depth=3,
            radius=2,
            max_candidates=10,
            use_alpha_beta=True,
            use_transposition_table=True,
        )
    finally:
        minimax_module.run_negamax_search = original

    assert result.tt_hits > 0
    assert len(seen_table_ids) == 3
    assert len(set(seen_table_ids)) == 1


def test_transposition_table_terminal_state_does_not_conflict() -> None:
    env = GomokuEnv()
    env.reset()
    env.done = True
    env.winner = BLACK
    env.current_player = BLACK

    result = AthenanNegamaxSearcher(
        max_depth=3,
        use_alpha_beta=True,
        use_transposition_table=True,
    ).search(env)

    assert result.best_action == -1
    assert result.root_value > 0.0


def test_transposition_table_search_does_not_mutate_original_env() -> None:
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

    _ = AthenanNegamaxSearcher(
        max_depth=4,
        max_candidates=10,
        use_alpha_beta=True,
        use_transposition_table=True,
    ).search(env)

    assert np.array_equal(env.board, board_before)
    assert state_before == (
        env.current_player,
        env.last_move,
        env.done,
        env.winner,
        env.move_count,
    )


def test_transposition_key_distinguishes_current_player() -> None:
    env_black = GomokuEnv()
    env_white = GomokuEnv()
    _place_stones(
        env_black,
        black_coords=[(7, 7)],
        white_coords=[(8, 8)],
        current_player=BLACK,
    )
    _place_stones(
        env_white,
        black_coords=[(7, 7)],
        white_coords=[(8, 8)],
        current_player=WHITE,
    )

    assert _make_transposition_key(env_black) != _make_transposition_key(env_white)


def test_transposition_table_search_is_deterministic() -> None:
    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 7)],
        white_coords=[(0, 0)],
        current_player=BLACK,
    )
    searcher = AthenanNegamaxSearcher(
        max_depth=4,
        max_candidates=10,
        use_alpha_beta=True,
        use_transposition_table=True,
    )

    result_a = searcher.search(env)
    result_b = searcher.search(env)

    assert result_a.best_action == result_b.best_action
    assert math.isclose(result_a.root_value, result_b.root_value, rel_tol=0.0, abs_tol=1e-9)


def test_transposition_flag_resolution_covers_exact_lower_upper() -> None:
    assert _resolve_tt_flag(3.0, alpha=1.0, beta=5.0) == "EXACT"
    assert _resolve_tt_flag(1.0, alpha=1.0, beta=5.0) == "UPPER"
    assert _resolve_tt_flag(5.0, alpha=1.0, beta=5.0) == "LOWER"
