"""Manual demo for AlphaZero-compatible MCTS search."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_ai.alphazero import MCTS, MCTSConfig, PolicyValueNet
from gomoku_ai.env import GomokuEnv


def main() -> None:
    """Run a short MCTS search and print the root statistics."""

    env = GomokuEnv()
    env.reset()
    model = PolicyValueNet()
    mcts = MCTS(MCTSConfig(num_simulations=25, add_root_noise=False, temperature=1.0))

    root = mcts.run(env, model)
    action_probs = mcts.get_action_probs(root, temperature=1.0)
    selected_action = mcts.select_action(root, temperature=0.0)
    selected_coord = env.action_to_coord(selected_action)

    top_actions = np.argsort(action_probs)[-5:][::-1]

    print("=== MCTS Demo ===")
    print(env.render())
    print()
    print(f"root child count: {len(root.children)}")
    print(f"root visit count: {root.visit_count}")
    print("top-5 action probs:")
    for action in top_actions:
        child = root.children.get(int(action))
        visit_count = 0 if child is None else child.visit_count
        print(f"  action={int(action):3d} coord={env.action_to_coord(int(action))} prob={action_probs[action]:.4f} visits={visit_count}")
    print(f"selected action: {selected_action}")
    print(f"selected coord: {selected_coord}")


if __name__ == "__main__":
    main()
