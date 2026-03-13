"""Manual demo for one simple single-process AlphaZero training cycle."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_ai.alphazero import (
    AlphaZeroTrainer,
    MCTSConfig,
    PolicyValueNet,
    SelfPlayConfig,
    TrainerConfig,
)


def main() -> None:
    """Run one small demo cycle and print the resulting training metrics."""

    trainer = AlphaZeroTrainer(
        model=PolicyValueNet(),
        trainer_config=TrainerConfig(
            num_self_play_games_per_cycle=2,
            batch_size=16,
            epochs_per_cycle=1,
            checkpoint_dir="checkpoints/alphazero/demo",
            device="cpu",
        ),
        self_play_config=SelfPlayConfig(),
        mcts_config=MCTSConfig(num_simulations=4),
    )

    metrics = trainer.run_training_cycle(cycle_index=0)

    print("=== AlphaZero Train Cycle Demo ===")
    print(f"generated self-play games: {int(metrics['num_self_play_games'])}")
    print(f"buffer samples: {int(metrics['buffer_samples'])}")
    print(f"avg policy_loss: {metrics['policy_loss']:.6f}")
    print(f"avg value_loss: {metrics['value_loss']:.6f}")
    print(f"avg total_loss: {metrics['total_loss']:.6f}")
    print(f"checkpoint: {metrics['checkpoint_path']}")


if __name__ == "__main__":
    main()
