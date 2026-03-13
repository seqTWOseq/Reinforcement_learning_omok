"""Evaluation and best-model promotion for AlphaZero Gomoku.

Evaluation is intentionally different from self-play:
- `root_noise=False` by default
- `temperature=0.0` by default
- no training samples or policy targets are stored
- candidate and reference must alternate black/white assignments

Best-model checkpoints are also intentionally separate from trainer
checkpoints. They store only the promoted model plus evaluation metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from gomoku_ai.alphazero.checkpoint_manager import load_best_model_checkpoint, save_best_model_checkpoint
from gomoku_ai.alphazero.evaluation_utils import (
    build_evaluation_mcts_config,
    compute_candidate_score,
    play_single_evaluation_game,
)
from gomoku_ai.alphazero.model import PolicyValueNet
from gomoku_ai.env import BLACK, DRAW, GomokuEnv, WHITE


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for deterministic candidate-vs-reference evaluation.

    `replace_win_rate_threshold` is applied to `candidate_score`, not pure win
    rate. Draws therefore contribute `draw_score` toward promotion.
    """

    num_eval_games: int = 20
    replace_win_rate_threshold: float = 0.55
    draw_score: float = 0.5
    eval_temperature: float = 0.0
    use_root_noise: bool = False
    eval_num_simulations: int = 50
    best_model_path: str = "checkpoints/alphazero/best_model.pt"

    def __post_init__(self) -> None:
        """Validate deterministic evaluation settings."""

        if self.num_eval_games <= 0:
            raise ValueError("num_eval_games must be positive.")
        if not 0.0 <= self.replace_win_rate_threshold <= 1.0:
            raise ValueError("replace_win_rate_threshold must be in [0.0, 1.0].")
        if not 0.0 <= self.draw_score <= 1.0:
            raise ValueError("draw_score must be in [0.0, 1.0].")
        if self.eval_temperature < 0.0:
            raise ValueError("eval_temperature must be non-negative.")
        if not isinstance(self.use_root_noise, bool):
            raise ValueError("use_root_noise must be a bool.")
        if self.eval_num_simulations <= 0:
            raise ValueError("eval_num_simulations must be positive.")
        if not self.best_model_path or not self.best_model_path.strip():
            raise ValueError("best_model_path must be a non-empty string.")


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated result of candidate-vs-reference evaluation games."""

    num_games: int
    candidate_wins: int
    reference_wins: int
    draws: int
    candidate_score: float
    candidate_win_rate: float
    promoted: bool
    candidate_as_black_wins: int = 0
    candidate_as_white_wins: int = 0
    candidate_black_games: int = 0
    candidate_white_games: int = 0

    def __post_init__(self) -> None:
        """Validate aggregate counts, scores, and color-split fields."""

        if self.num_games <= 0:
            raise ValueError("num_games must be positive.")
        if min(self.candidate_wins, self.reference_wins, self.draws) < 0:
            raise ValueError("Win/loss/draw counts must be non-negative.")
        if self.candidate_wins + self.reference_wins + self.draws != self.num_games:
            raise ValueError("candidate_wins + reference_wins + draws must equal num_games.")
        if not 0.0 <= self.candidate_score <= 1.0:
            raise ValueError("candidate_score must be in [0.0, 1.0].")
        if not 0.0 <= self.candidate_win_rate <= 1.0:
            raise ValueError("candidate_win_rate must be in [0.0, 1.0].")
        if self.candidate_black_games < 0 or self.candidate_white_games < 0:
            raise ValueError("Color assignment counts must be non-negative.")
        if self.candidate_black_games + self.candidate_white_games != self.num_games:
            raise ValueError("candidate_black_games + candidate_white_games must equal num_games.")
        if self.candidate_as_black_wins < 0 or self.candidate_as_white_wins < 0:
            raise ValueError("Color-split candidate wins must be non-negative.")
        if self.candidate_as_black_wins > self.candidate_black_games:
            raise ValueError("candidate_as_black_wins cannot exceed candidate_black_games.")
        if self.candidate_as_white_wins > self.candidate_white_games:
            raise ValueError("candidate_as_white_wins cannot exceed candidate_white_games.")

    def as_dict(self) -> dict[str, Any]:
        """Convert the result to a plain dictionary for checkpoint metadata."""

        return {
            "num_games": self.num_games,
            "candidate_wins": self.candidate_wins,
            "reference_wins": self.reference_wins,
            "draws": self.draws,
            "candidate_score": self.candidate_score,
            "candidate_win_rate": self.candidate_win_rate,
            "promoted": self.promoted,
            "candidate_as_black_wins": self.candidate_as_black_wins,
            "candidate_as_white_wins": self.candidate_as_white_wins,
            "candidate_black_games": self.candidate_black_games,
            "candidate_white_games": self.candidate_white_games,
        }


class AlphaZeroEvaluator:
    """Evaluate a candidate model against the current reference(best) model.

    Differences from self-play:
    - evaluation is deterministic by default (`temperature=0.0`)
    - root noise is disabled by default
    - no training data is collected
    - candidate/reference colors alternate across games
    """

    def __init__(
        self,
        config: EvaluationConfig | None = None,
        env_factory: Callable[[], GomokuEnv] | None = None,
    ) -> None:
        """Initialize the evaluator and reusable environment factory."""

        self.config = config or EvaluationConfig()
        self.env_factory = env_factory or GomokuEnv

    def play_match(
        self,
        candidate_model: PolicyValueNet,
        reference_model: PolicyValueNet,
    ) -> EvaluationResult:
        """Play a deterministic evaluation match and aggregate the result."""

        if not isinstance(candidate_model, PolicyValueNet):
            raise TypeError("candidate_model must be a PolicyValueNet instance.")
        if not isinstance(reference_model, PolicyValueNet):
            raise TypeError("reference_model must be a PolicyValueNet instance.")

        mcts_config = build_evaluation_mcts_config(self.config)
        candidate_was_training = candidate_model.training
        reference_was_training = reference_model.training
        same_model = candidate_model is reference_model

        candidate_model.eval()
        if not same_model:
            reference_model.eval()

        candidate_wins = 0
        reference_wins = 0
        draws = 0
        candidate_as_black_wins = 0
        candidate_as_white_wins = 0
        candidate_black_games = 0
        candidate_white_games = 0

        try:
            for game_index in range(self.config.num_eval_games):
                candidate_is_black = game_index % 2 == 0
                if candidate_is_black:
                    candidate_black_games += 1
                else:
                    candidate_white_games += 1

                winner = self._play_single_game(
                    candidate_model,
                    reference_model,
                    candidate_is_black=candidate_is_black,
                    mcts_config=mcts_config,
                )
                if winner == BLACK and candidate_is_black:
                    candidate_wins += 1
                    candidate_as_black_wins += 1
                elif winner == WHITE and not candidate_is_black:
                    candidate_wins += 1
                    candidate_as_white_wins += 1
                elif winner == BLACK or winner == WHITE:
                    reference_wins += 1
                elif winner == DRAW:
                    draws += 1
                else:
                    raise ValueError(f"Unexpected winner value: {winner!r}.")
        finally:
            if same_model:
                if candidate_was_training:
                    candidate_model.train()
                else:
                    candidate_model.eval()
            else:
                if candidate_was_training:
                    candidate_model.train()
                else:
                    candidate_model.eval()
                if reference_was_training:
                    reference_model.train()
                else:
                    reference_model.eval()

        candidate_score, candidate_win_rate = compute_candidate_score(
            candidate_wins,
            reference_wins,
            draws,
            num_games=self.config.num_eval_games,
            draw_score=self.config.draw_score,
        )
        provisional_result = EvaluationResult(
            num_games=self.config.num_eval_games,
            candidate_wins=candidate_wins,
            reference_wins=reference_wins,
            draws=draws,
            candidate_score=candidate_score,
            candidate_win_rate=candidate_win_rate,
            promoted=False,
            candidate_as_black_wins=candidate_as_black_wins,
            candidate_as_white_wins=candidate_as_white_wins,
            candidate_black_games=candidate_black_games,
            candidate_white_games=candidate_white_games,
        )
        return EvaluationResult(
            num_games=provisional_result.num_games,
            candidate_wins=provisional_result.candidate_wins,
            reference_wins=provisional_result.reference_wins,
            draws=provisional_result.draws,
            candidate_score=provisional_result.candidate_score,
            candidate_win_rate=provisional_result.candidate_win_rate,
            promoted=self.should_promote(provisional_result),
            candidate_as_black_wins=provisional_result.candidate_as_black_wins,
            candidate_as_white_wins=provisional_result.candidate_as_white_wins,
            candidate_black_games=provisional_result.candidate_black_games,
            candidate_white_games=provisional_result.candidate_white_games,
        )

    def should_promote(self, result: EvaluationResult) -> bool:
        """Return `True` when the candidate score reaches the promotion threshold."""

        if not isinstance(result, EvaluationResult):
            raise TypeError("result must be an EvaluationResult instance.")
        return result.candidate_score >= self.config.replace_win_rate_threshold

    def promote_if_better(
        self,
        candidate_model: PolicyValueNet,
        cycle_index: int,
        result: EvaluationResult | None = None,
    ) -> bool:
        """Promote the candidate to best model if bootstrap or threshold allows it."""

        if not isinstance(candidate_model, PolicyValueNet):
            raise TypeError("candidate_model must be a PolicyValueNet instance.")
        if cycle_index < 0:
            raise ValueError("cycle_index must be non-negative.")

        best_exists = Path(self.config.best_model_path).exists()
        if not best_exists:
            bootstrap_result = result or EvaluationResult(
                num_games=1,
                candidate_wins=1,
                reference_wins=0,
                draws=0,
                candidate_score=1.0,
                candidate_win_rate=1.0,
                promoted=True,
                candidate_as_black_wins=1,
                candidate_as_white_wins=0,
                candidate_black_games=1,
                candidate_white_games=0,
            )
            save_best_model_checkpoint(
                candidate_model,
                self.config.best_model_path,
                source_cycle_index=cycle_index,
                evaluation_result=bootstrap_result,
            )
            return True

        if result is None:
            raise ValueError("result must be provided when a best model already exists.")
        if not self.should_promote(result):
            return False

        save_best_model_checkpoint(
            candidate_model,
            self.config.best_model_path,
            source_cycle_index=cycle_index,
            evaluation_result=result,
        )
        return True

    def load_best_model(self) -> PolicyValueNet | None:
        """Load the current best model checkpoint if it exists."""

        return load_best_model_checkpoint(self.config.best_model_path)

    def _play_single_game(
        self,
        candidate_model: PolicyValueNet,
        reference_model: PolicyValueNet,
        *,
        candidate_is_black: bool,
        mcts_config: Any,
    ) -> int:
        """Play one evaluation game. Split out for targeted tests and overrides."""

        return play_single_evaluation_game(
            candidate_model,
            reference_model,
            candidate_is_black=candidate_is_black,
            mcts_config=mcts_config,
            env_factory=self.env_factory,
        )
