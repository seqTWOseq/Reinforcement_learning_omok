"""Manual demo for a scripted human-vs-AlphaZero game."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_ai.alphazero import HumanPlayConfig, HumanVsAlphaZeroGameRunner, PolicyValueNet, PolicyValueNetConfig
from gomoku_ai.env import BLACK, BOARD_SIZE


class StrongScriptedPolicyValueNet(PolicyValueNet):
    """Fast scripted AI for the human-play demo."""

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


def main() -> None:
    """Run one scripted human-vs-AI game and print the stored record summary."""

    runner = HumanVsAlphaZeroGameRunner(
        HumanPlayConfig(
            human_color="select_each_game",
            ai_temperature=0.0,
            use_root_noise=False,
            ai_num_simulations=2,
            record_ai_turn_only=True,
            game_id_prefix="humanplay",
        )
    )
    record = runner.play_game(
        StrongScriptedPolicyValueNet(),
        human_color=BLACK,
        human_moves=[0, 17, 34, 51, 68],
    )

    print("=== Human vs AlphaZero Demo ===")
    print(f"game_id: {record.game_id}")
    print(f"winner: {record.winner}")
    print(f"num_moves: {len(record.moves)}")
    print(f"source: {record.source}")
    print(f"metadata: {record.metadata}")
    print(f"samples: {len(record.samples)}")
    print("first 3 samples:")
    for sample in record.samples[:3]:
        print(
            "  "
            f"move_index={sample.move_index} "
            f"player_to_move={sample.player_to_move} "
            f"value_target={float(sample.value_target):.1f} "
            f"argmax_action={int(sample.policy_target.argmax())}"
        )
    print(f"moves[:10]: {list(record.moves[:10])}")


if __name__ == "__main__":
    main()
