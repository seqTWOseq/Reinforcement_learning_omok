"""Tests for AlphaZero evaluation and best-model promotion."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from gomoku_ai.alphazero import (
    AlphaZeroEvaluator,
    EvaluationConfig,
    EvaluationResult,
    PolicyValueNet,
    PolicyValueNetConfig,
    build_evaluation_mcts_config,
    compute_candidate_score,
)
from gomoku_ai.env import BOARD_SIZE


class StrongScriptedPolicyValueNet(PolicyValueNet):
    """Deterministic strong model that builds a fast horizontal five."""

    def __init__(self) -> None:
        super().__init__(PolicyValueNetConfig(use_batch_norm=False))

    def predict_single(
        self,
        state_np: np.ndarray,
        device: object | None = None,
        *,
        move_model: bool = False,
    ) -> tuple[np.ndarray, float]:
        del device
        del move_model

        state = np.asarray(state_np, dtype=np.float32)
        occupied = (state[0] + state[1]) > 0.5
        black_to_move = bool(state[3, 0, 0] == 1.0)
        row = 7 if black_to_move else 8

        logits = np.full((BOARD_SIZE * BOARD_SIZE,), -100.0, dtype=np.float32)
        for col in range(5):
            if not occupied[row, col]:
                logits[row * BOARD_SIZE + col] = 100.0
                break
        else:
            empty_indices = np.flatnonzero(~occupied.reshape(-1))
            logits[int(empty_indices[0])] = 100.0
        return logits, 0.5


class WeakScriptedPolicyValueNet(PolicyValueNet):
    """Deterministic weak model that plays scattered non-threatening moves."""

    def __init__(self) -> None:
        super().__init__(PolicyValueNetConfig(use_batch_norm=False))

    def predict_single(
        self,
        state_np: np.ndarray,
        device: object | None = None,
        *,
        move_model: bool = False,
    ) -> tuple[np.ndarray, float]:
        del device
        del move_model

        state = np.asarray(state_np, dtype=np.float32)
        occupied = (state[0] + state[1]) > 0.5
        black_to_move = bool(state[3, 0, 0] == 1.0)
        targets = (
            [(0, 10), (1, 12), (2, 14), (4, 11), (6, 13)]
            if black_to_move
            else [(14, 10), (13, 12), (12, 14), (10, 11), (8, 13)]
        )

        logits = np.full((BOARD_SIZE * BOARD_SIZE,), -100.0, dtype=np.float32)
        for row, col in targets:
            if not occupied[row, col]:
                logits[row * BOARD_SIZE + col] = 100.0
                break
        else:
            empty_indices = np.flatnonzero(~occupied.reshape(-1))
            logits[int(empty_indices[0])] = 100.0
        return logits, -0.5


class SpyEvaluator(AlphaZeroEvaluator):
    """Evaluator subclass that captures the MCTS config used by `play_match`."""

    def __init__(self, config: EvaluationConfig) -> None:
        super().__init__(config=config)
        self.seen_configs: list[tuple[int, bool, float]] = []

    def _play_single_game(
        self,
        candidate_model: PolicyValueNet,
        reference_model: PolicyValueNet,
        *,
        candidate_is_black: bool,
        mcts_config: object,
    ) -> int:
        self.seen_configs.append(
            (mcts_config.num_simulations, mcts_config.add_root_noise, mcts_config.temperature)
        )
        return super()._play_single_game(
            candidate_model,
            reference_model,
            candidate_is_black=candidate_is_black,
            mcts_config=mcts_config,
        )


def _temp_best_model_path() -> Path:
    """Return an isolated temporary best-model checkpoint path."""

    return Path.cwd() / f"best_model_test_{uuid4().hex}.pt"


def test_play_match_returns_evaluation_result() -> None:
    """A candidate/reference match should return a populated evaluation result."""

    best_model_path = _temp_best_model_path()
    evaluator = AlphaZeroEvaluator(
        EvaluationConfig(
            num_eval_games=4,
            replace_win_rate_threshold=0.55,
            draw_score=0.5,
            eval_temperature=0.0,
            use_root_noise=False,
            eval_num_simulations=2,
            best_model_path=str(best_model_path),
        )
    )
    result = evaluator.play_match(StrongScriptedPolicyValueNet(), WeakScriptedPolicyValueNet())

    assert isinstance(result, EvaluationResult)
    assert result.num_games == 4
    assert result.candidate_wins == 4
    assert result.reference_wins == 0
    assert result.draws == 0
    assert result.candidate_score == 1.0
    assert result.candidate_win_rate == 1.0
    assert result.promoted is True


def test_candidate_and_reference_colors_are_alternated() -> None:
    """Candidate should alternate black and white assignments across the match."""

    evaluator = AlphaZeroEvaluator(
        EvaluationConfig(
            num_eval_games=4,
            eval_num_simulations=2,
            best_model_path=str(_temp_best_model_path()),
        )
    )
    result = evaluator.play_match(StrongScriptedPolicyValueNet(), WeakScriptedPolicyValueNet())

    assert result.candidate_black_games == 2
    assert result.candidate_white_games == 2
    assert result.candidate_as_black_wins == 2
    assert result.candidate_as_white_wins == 2


def test_candidate_score_computation() -> None:
    """Candidate score should award full wins and partial draw credit."""

    candidate_score, candidate_win_rate = compute_candidate_score(
        10,
        6,
        4,
        num_games=20,
        draw_score=0.5,
    )

    assert candidate_score == 0.6
    assert candidate_win_rate == 0.5


def test_should_promote_threshold_behavior() -> None:
    """Promotion should use candidate_score >= threshold semantics."""

    evaluator = AlphaZeroEvaluator(
        EvaluationConfig(
            replace_win_rate_threshold=0.55,
            best_model_path=str(_temp_best_model_path()),
        )
    )
    below = EvaluationResult(
        num_games=20,
        candidate_wins=10,
        reference_wins=9,
        draws=1,
        candidate_score=0.525,
        candidate_win_rate=0.5,
        promoted=False,
        candidate_as_black_wins=5,
        candidate_as_white_wins=5,
        candidate_black_games=10,
        candidate_white_games=10,
    )
    at_threshold = EvaluationResult(
        num_games=20,
        candidate_wins=10,
        reference_wins=8,
        draws=2,
        candidate_score=0.55,
        candidate_win_rate=0.5,
        promoted=True,
        candidate_as_black_wins=5,
        candidate_as_white_wins=5,
        candidate_black_games=10,
        candidate_white_games=10,
    )

    assert evaluator.should_promote(below) is False
    assert evaluator.should_promote(at_threshold) is True


def test_promote_if_better_saves_best_checkpoint() -> None:
    """Promoted candidates should be saved as standalone best-model checkpoints."""

    best_model_path = _temp_best_model_path()
    evaluator = AlphaZeroEvaluator(
        EvaluationConfig(
            num_eval_games=4,
            replace_win_rate_threshold=0.55,
            eval_num_simulations=2,
            best_model_path=str(best_model_path),
        )
    )
    candidate = StrongScriptedPolicyValueNet()
    result = EvaluationResult(
        num_games=4,
        candidate_wins=4,
        reference_wins=0,
        draws=0,
        candidate_score=1.0,
        candidate_win_rate=1.0,
        promoted=True,
        candidate_as_black_wins=2,
        candidate_as_white_wins=2,
        candidate_black_games=2,
        candidate_white_games=2,
    )

    try:
        saved = evaluator.promote_if_better(candidate, cycle_index=7, result=result)
        payload = torch.load(best_model_path, map_location="cpu")
        loaded_model = evaluator.load_best_model()
    finally:
        if best_model_path.exists():
            best_model_path.unlink()

    assert saved is True
    assert payload["format_version"] == 1
    assert payload["checkpoint_type"] == "alphazero_best_model"
    assert payload["source_cycle_index"] == 7
    assert payload["evaluation_result"]["candidate_score"] == 1.0
    assert isinstance(loaded_model, PolicyValueNet)


def test_bootstrap_best_model_when_missing() -> None:
    """The first candidate should bootstrap the best-model checkpoint when missing."""

    best_model_path = _temp_best_model_path()
    evaluator = AlphaZeroEvaluator(
        EvaluationConfig(
            num_eval_games=4,
            replace_win_rate_threshold=0.99,
            eval_num_simulations=2,
            best_model_path=str(best_model_path),
        )
    )
    candidate = StrongScriptedPolicyValueNet()
    below_threshold_result = EvaluationResult(
        num_games=4,
        candidate_wins=2,
        reference_wins=2,
        draws=0,
        candidate_score=0.5,
        candidate_win_rate=0.5,
        promoted=False,
        candidate_as_black_wins=1,
        candidate_as_white_wins=1,
        candidate_black_games=2,
        candidate_white_games=2,
    )

    try:
        saved = evaluator.promote_if_better(candidate, cycle_index=0, result=below_threshold_result)
        loaded_model = evaluator.load_best_model()
    finally:
        if best_model_path.exists():
            best_model_path.unlink()

    assert saved is True
    assert isinstance(loaded_model, PolicyValueNet)


def test_evaluation_mcts_config_is_deterministic() -> None:
    """Evaluation MCTS config should disable root noise and use zero temperature."""

    config = EvaluationConfig(
        num_eval_games=4,
        eval_temperature=0.0,
        use_root_noise=False,
        eval_num_simulations=7,
        best_model_path=str(_temp_best_model_path()),
    )
    mcts_config = build_evaluation_mcts_config(config)

    assert mcts_config.num_simulations == 7
    assert mcts_config.add_root_noise is False
    assert mcts_config.temperature == 0.0


def test_play_match_uses_deterministic_eval_settings() -> None:
    """`play_match` should pass deterministic evaluation settings into MCTS."""

    evaluator = SpyEvaluator(
        EvaluationConfig(
            num_eval_games=2,
            eval_temperature=0.0,
            use_root_noise=False,
            eval_num_simulations=3,
            best_model_path=str(_temp_best_model_path()),
        )
    )
    evaluator.play_match(StrongScriptedPolicyValueNet(), WeakScriptedPolicyValueNet())

    assert evaluator.seen_configs
    for num_simulations, add_root_noise, temperature in evaluator.seen_configs:
        assert num_simulations == 3
        assert add_root_noise is False
        assert temperature == 0.0
