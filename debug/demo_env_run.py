"""Small manual runner for inspecting the Gomoku environment behaviour."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_ai.env import BLACK, DRAW, WHITE, GomokuEnv


def main() -> None:
    """Run a short scripted game and print the resulting board states."""

    env = GomokuEnv()
    env.reset()

    scripted_moves = [
        env.coord_to_action(7, 7),
        env.coord_to_action(0, 0),
        env.coord_to_action(7, 8),
        env.coord_to_action(0, 1),
        env.coord_to_action(7, 9),
        env.coord_to_action(0, 2),
        env.coord_to_action(7, 10),
        env.coord_to_action(0, 3),
        env.coord_to_action(7, 11),
    ]

    print("=== GomokuEnv Demo ===")
    print(env.render())
    print()

    for index, action in enumerate(scripted_moves, start=1):
        result = env.apply_move(action)
        player = result["player"]
        print(f"[move {index}] player={player} action={action} coord=({result['row']}, {result['col']})")
        print(env.render())
        print()
        if result["done"]:
            break

    if env.winner == BLACK:
        print("Winner: BLACK")
    elif env.winner == WHITE:
        print("Winner: WHITE")
    elif env.winner == DRAW:
        print("Result: DRAW")
    else:
        print("Game is still ongoing.")

    print(f"Terminal: {env.is_terminal()}")
    print(f"Last move: {env.last_move}")
    print(f"Move count: {env.move_count}")


if __name__ == "__main__":
    main()
