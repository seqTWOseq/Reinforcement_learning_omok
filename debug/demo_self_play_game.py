"""Manual demo for generating one AlphaZero self-play game."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_ai.alphazero import MCTSConfig, PolicyValueNet, SelfPlayConfig, SelfPlayGameGenerator


def main() -> None:
    """Generate one self-play game and print a compact summary."""

    generator = SelfPlayGameGenerator(
        config=SelfPlayConfig(),
        mcts_config=MCTSConfig(num_simulations=4),
    )
    record = generator.play_one_game(PolicyValueNet())

    print("=== Self-Play Demo ===")
    print(f"game_id: {record.game_id}")
    print(f"winner: {record.winner}")
    print(f"num_moves: {len(record.moves)}")
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
    print(f"metadata: {record.metadata}")


if __name__ == "__main__":
    main()
