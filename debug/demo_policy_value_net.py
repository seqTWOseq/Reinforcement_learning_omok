"""Manual demo for the AlphaZero baseline policy/value network."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_ai.alphazero import PolicyValueNet, count_parameters, predict_single
from gomoku_ai.env import GomokuEnv


def main() -> None:
    """Print a short walkthrough of the model input/output contract."""

    env = GomokuEnv()
    env.reset()
    state = env.encode_state()
    model = PolicyValueNet()

    batch_tensor = torch.from_numpy(state).unsqueeze(0).to(dtype=torch.float32)
    policy_logits, value = model(batch_tensor)
    single_policy_logits, single_value = predict_single(model, state)

    print("=== PolicyValueNet Demo ===")
    print(f"encode_state shape: {state.shape}, dtype: {state.dtype}")
    print(f"forward policy shape: {tuple(policy_logits.shape)}, dtype: {policy_logits.dtype}")
    print(f"forward value shape: {tuple(value.shape)}, dtype: {value.dtype}")
    print(f"predict_single logits shape: {single_policy_logits.shape}, dtype: {single_policy_logits.dtype}")
    print(f"parameter count: {count_parameters(model)}")
    print(f"value output example: {single_value:.6f}")


if __name__ == "__main__":
    main()
