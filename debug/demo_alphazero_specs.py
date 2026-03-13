"""Manual demo for AlphaZero-compatible specs and helper utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_ai.alphazero import (
    ACTION_SIZE,
    build_game_step_sample,
    policy_logits_to_probs,
    winner_to_value_target,
)
from gomoku_ai.env import BLACK, GomokuEnv


def main() -> None:
    """Print a short walkthrough of the AlphaZero specs layer."""

    env = GomokuEnv()
    env.reset()

    encoded_state = env.encode_state()
    valid_moves = env.get_valid_moves()

    policy_logits = np.linspace(-1.0, 1.0, ACTION_SIZE, dtype=np.float32)
    masked_probs = policy_logits_to_probs(policy_logits, valid_moves)

    policy_target = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy_target[env.coord_to_action(7, 7)] = 1.0
    sample = build_game_step_sample(
        env=env,
        policy_target=policy_target,
        value_target=0.0,
        move_index=0,
        action_taken=env.coord_to_action(7, 7),
        game_id="demo-game",
    )

    print("=== AlphaZero Specs Demo ===")
    print(f"encode_state shape: {encoded_state.shape}, dtype: {encoded_state.dtype}")
    print(f"valid_moves shape: {valid_moves.shape}, dtype: {valid_moves.dtype}, legal_count: {int(valid_moves.sum())}")
    print(f"masked_probs shape: {masked_probs.shape}, dtype: {masked_probs.dtype}")
    print(f"top-5 move probs: {np.sort(masked_probs)[-5:][::-1]}")
    print(f"value target example (black winner, black to move): {winner_to_value_target(BLACK, BLACK)}")
    print("GameStepSample example:")
    print(f"  state shape: {sample.state.shape}")
    print(f"  policy target sum: {float(sample.policy_target.sum()):.1f}")
    print(f"  value target: {float(sample.value_target):.1f}")
    print(f"  player_to_move: {sample.player_to_move}")
    print(f"  move_index: {sample.move_index}")
    print(f"  action_taken: {sample.action_taken}")
    print(f"  game_id: {sample.game_id}")


if __name__ == "__main__":
    main()
