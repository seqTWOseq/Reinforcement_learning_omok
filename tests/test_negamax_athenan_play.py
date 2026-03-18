"""Tests for the negamax Athenan human-play entrypoint."""

from __future__ import annotations

import numpy as np

import gomoku_ai.negamax_athenan.play.play_human_vs_negamax_athenan as negamax_play
from gomoku_ai.negamax_athenan.play import play_human_vs_negamax_athenan_game
from gomoku_ai.negamax_athenan.search import AthenanNegamaxSearcher
from gomoku_ai.env import BLACK, GomokuEnv, WHITE


def test_human_play_negamax_entrypoint_function_smoke_with_debug() -> None:
    """Negamax human-play function should complete a game and return debug payloads."""

    searcher = AthenanNegamaxSearcher(
        max_depth=2,
        candidate_radius=1,
        max_candidates=10,
        use_alpha_beta=True,
        use_iterative_deepening=True,
        use_transposition_table=True,
    )

    def first_legal_human_move(env: GomokuEnv) -> int:
        legal_actions = np.flatnonzero(np.asarray(env.get_valid_moves(), dtype=bool))
        return int(legal_actions[0])

    result = play_human_vs_negamax_athenan_game(
        searcher=searcher,
        human_color=BLACK,
        env_factory=lambda: GomokuEnv(board_size=5),
        human_move_selector=first_legal_human_move,
        debug=True,
        debug_top_k=3,
        print_fn=lambda _: None,
    )

    assert result.move_count == len(result.moves)
    assert result.move_count > 0
    assert result.ai_debug_turns
    for turn in result.ai_debug_turns:
        assert turn.principal_variation
        assert turn.top_actions


def test_negamax_play_cli_builds_negamax_searcher_with_expected_options() -> None:
    """Negamax CLI should build the negamax searcher with its dedicated options."""

    captured: dict[str, object] = {}
    original = negamax_play.play_human_vs_negamax_athenan_game

    def fake_play(**kwargs: object) -> object:
        captured.update(kwargs)
        return None

    negamax_play.play_human_vs_negamax_athenan_game = fake_play
    try:
        negamax_play.main(
            [
                "--human-color",
                "white",
                "--max-depth",
                "3",
                "--candidate-radius",
                "2",
                "--max-candidates",
                "16",
                "--no-iterative",
                "--no-alpha-beta",
                "--debug",
                "--debug-top-k",
                "4",
            ]
        )
    finally:
        negamax_play.play_human_vs_negamax_athenan_game = original

    assert captured["human_color"] == WHITE
    assert captured["debug"] is True
    assert captured["debug_top_k"] == 4
    searcher = captured["searcher"]
    assert isinstance(searcher, AthenanNegamaxSearcher)
    assert searcher.max_depth == 3
    assert searcher.candidate_radius == 2
    assert searcher.max_candidates == 16
    assert searcher.use_iterative_deepening is False
    assert searcher.use_alpha_beta is False
    assert searcher.use_transposition_table is True
