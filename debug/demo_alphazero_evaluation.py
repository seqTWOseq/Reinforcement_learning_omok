"""Manual demo for AlphaZero candidate-vs-reference evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_ai.alphazero import AlphaZeroEvaluator, EvaluationConfig, PolicyValueNet, PolicyValueNetConfig
from gomoku_ai.env import BOARD_SIZE


class StrongScriptedPolicyValueNet(PolicyValueNet):
    """Fast scripted candidate for the evaluation demo."""

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
    """Fast scripted reference for the evaluation demo."""

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


def main() -> None:
    """Run a compact deterministic evaluation demo and print the result."""

    evaluator = AlphaZeroEvaluator(
        EvaluationConfig(
            num_eval_games=4,
            eval_num_simulations=2,
            best_model_path="checkpoints/alphazero/demo_best_model.pt",
        )
    )
    candidate_model = StrongScriptedPolicyValueNet()
    reference_model = WeakScriptedPolicyValueNet()

    result = evaluator.play_match(candidate_model, reference_model)
    promoted = evaluator.promote_if_better(candidate_model, cycle_index=0, result=result)

    print("=== AlphaZero Evaluation Demo ===")
    print(f"num_games: {result.num_games}")
    print(f"candidate_wins: {result.candidate_wins}")
    print(f"reference_wins: {result.reference_wins}")
    print(f"draws: {result.draws}")
    print(f"candidate_score: {result.candidate_score:.3f}")
    print(f"candidate_win_rate: {result.candidate_win_rate:.3f}")
    print(f"promoted: {promoted}")
    print(f"best_model_path: {evaluator.config.best_model_path}")


if __name__ == "__main__":
    main()
