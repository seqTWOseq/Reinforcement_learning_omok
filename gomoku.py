"""Tkinter Gomoku GUI for playing against a trained AlphaZero model."""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from gomoku_ai.alphazero import MCTS, MCTSConfig, load_model_for_inference, resolve_model_checkpoint_path
from gomoku_ai.env import BLACK, BOARD_SIZE, DRAW, EMPTY, GomokuEnv, WHITE


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer.")
    return parsed


def _non_negative_float(value: str) -> float:
    """Parse a non-negative float for argparse."""

    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative.")
    return parsed


def _validate_device(device: str) -> str:
    """Validate the requested torch device string before loading a checkpoint."""

    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid torch device string: {device!r}.") from exc

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is False.")
    return str(resolved)


def _parse_human_color(raw_color: str) -> int:
    """Resolve the human color argument."""

    normalized = raw_color.strip().lower()
    if normalized in {"black", "b"}:
        return BLACK
    if normalized in {"white", "w"}:
        return WHITE
    raise ValueError("human_color must be one of {'black', 'white'}.")


def _player_name(player: int) -> str:
    """Return a readable label for a player color."""

    if player == BLACK:
        return "black"
    if player == WHITE:
        return "white"
    if player == DRAW:
        return "draw"
    return f"unknown({player})"


class OmokEnvGUI:
    """Small Tkinter Gomoku board used for human-vs-AI play."""

    def __init__(self, render_mode: str = "human") -> None:
        self._env = GomokuEnv(board_size=BOARD_SIZE)
        self.board_size = self._env.board_size
        self.render_mode = render_mode

        self.window: tk.Tk | None = None
        self.canvas: tk.Canvas | None = None
        self.cell_size = 40
        self.margin = 30

    @property
    def board(self) -> np.ndarray:
        return self._env.board

    @board.setter
    def board(self, value: np.ndarray) -> None:
        self._env.board = value

    @property
    def current_player(self) -> int:
        return self._env.current_player

    @current_player.setter
    def current_player(self, value: int) -> None:
        self._env.current_player = value

    @property
    def last_move(self) -> tuple[int, int] | None:
        return self._env.last_move

    @last_move.setter
    def last_move(self, value: tuple[int, int] | None) -> None:
        self._env.last_move = value

    @property
    def winner(self) -> int | None:
        return self._env.winner

    @winner.setter
    def winner(self, value: int | None) -> None:
        self._env.winner = value

    @property
    def done(self) -> bool:
        return self._env.done

    @done.setter
    def done(self, value: bool) -> None:
        self._env.done = value

    @property
    def move_count(self) -> int:
        return self._env.move_count

    @move_count.setter
    def move_count(self, value: int) -> None:
        self._env.move_count = value

    def action_to_coord(self, action: int) -> tuple[int, int]:
        """Delegate flat-action to `(row, col)` conversion to the rule engine."""

        return self._env.action_to_coord(action)

    def coord_to_action(self, row: int, col: int) -> int:
        """Delegate `(row, col)` to flat-action conversion to the rule engine."""

        return self._env.coord_to_action(row, col)

    def reset(self) -> tuple[np.ndarray, dict[str, int]]:
        """Reset the board to the initial state."""

        self._env.reset()
        return self.board.copy(), {"current_player": self.current_player}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, int | str | bool | tuple[int, int] | None]]:
        """Apply one move through `GomokuEnv` and adapt the result for the GUI loop."""

        result = self._env.apply_move(action)
        reward = 1.0 if result["reason"] == "win" else 0.0
        info: dict[str, int | str | bool | tuple[int, int] | None] = dict(result)
        if result["next_player"] is not None:
            info["current_player"] = result["next_player"]
        return self.board.copy(), reward, bool(result["done"]), False, info

    def get_valid_moves(self) -> np.ndarray:
        """Return the engine-provided flat boolean mask of legal moves."""

        return self._env.get_valid_moves()

    def is_legal_action(self, action: int) -> bool:
        """Return `True` when `action` is legal in the current engine state."""

        return self._env.is_legal_action(action)

    def get_legal_actions(self) -> list[int]:
        """Return a list of legal flat actions from the current engine state."""

        return self._env.get_legal_actions()

    def to_alphazero_env(self) -> GomokuEnv:
        """Build a `GomokuEnv` snapshot that matches the GUI state."""

        return self._env.clone()

    def render(self) -> None:
        """Draw or refresh the Tkinter board."""

        if self.render_mode != "human":
            return

        if self.window is None:
            self.window = tk.Tk()
            self.window.title("Gomoku vs AlphaZero")
            size = (self.board_size - 1) * self.cell_size + self.margin * 2
            self.canvas = tk.Canvas(self.window, width=size, height=size, bg="#D7A85A", highlightthickness=0)
            self.canvas.pack()

        if self.canvas is None or self.window is None:
            raise RuntimeError("GUI canvas was not initialized.")

        self.window.title(f"Gomoku vs AlphaZero - {_player_name(self.current_player).title()} to move")
        self.canvas.delete("all")

        for index in range(self.board_size):
            start = self.margin + index * self.cell_size
            end = self.margin + (self.board_size - 1) * self.cell_size
            self.canvas.create_line(self.margin, start, end, start, width=1)
            self.canvas.create_line(start, self.margin, start, end, width=1)

        for row in range(self.board_size):
            for col in range(self.board_size):
                value = int(self.board[row, col])
                if value == EMPTY:
                    continue
                x = self.margin + col * self.cell_size
                y = self.margin + row * self.cell_size
                radius = self.cell_size // 2 - 2
                fill = "black" if value == BLACK else "white"
                outline = "#C0392B" if self.last_move == (row, col) else "black"
                width = 3 if self.last_move == (row, col) else 1
                self.canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=fill,
                    outline=outline,
                    width=width,
                )

        if self.done:
            result_text = "Draw" if self.winner == DRAW else f"{_player_name(self.winner).title()} wins"
        else:
            result_text = f"{_player_name(self.current_player).title()} to move"
        self.canvas.create_text(
            self.margin,
            12,
            text=result_text,
            anchor="w",
            fill="#1F2937",
            font=("Segoe UI", 11, "bold"),
        )

        self.window.update_idletasks()
        self.window.update()

    def close(self) -> None:
        """Destroy the Tkinter window."""

        if self.window is not None:
            self.window.destroy()
            self.window = None
            self.canvas = None


class HumanAgent:
    """Mouse-driven human player for the Tkinter board."""

    def __init__(self, env: OmokEnvGUI, *, name: str) -> None:
        self.name = name
        self.env = env
        self.clicked_action: int | None = None

    def select_action(self, _: OmokEnvGUI) -> int:
        """Wait until the user clicks a legal move."""

        if self.env.canvas is None or self.env.window is None:
            raise RuntimeError("The GUI must be rendered before reading a human move.")

        self.clicked_action = None
        self.env.canvas.bind("<Button-1>", self._click_handler)

        while self.clicked_action is None:
            self.env.window.update()
            time.sleep(0.05)

        self.env.canvas.unbind("<Button-1>")
        return self.clicked_action

    def _click_handler(self, event: tk.Event[tk.Misc]) -> None:
        """Convert a click position into a legal board action."""

        col = round((event.x - self.env.margin) / self.env.cell_size)
        row = round((event.y - self.env.margin) / self.env.cell_size)

        if not 0 <= row < self.env.board_size or not 0 <= col < self.env.board_size:
            return

        action = self.env.coord_to_action(row, col)
        if self.env.is_legal_action(action):
            self.clicked_action = action


class AlphaZeroAgent:
    """AlphaZero-powered AI player backed by the trained policy/value net."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str,
        num_simulations: int,
        temperature: float,
        name: str = "AlphaZero",
    ) -> None:
        self.checkpoint_path = resolve_model_checkpoint_path(checkpoint_path)
        self.name = f"{name} ({self.checkpoint_path.name})"
        self.model = load_model_for_inference(self.checkpoint_path, device=device)
        self.mcts = MCTS(
            MCTSConfig(
                num_simulations=num_simulations,
                add_root_noise=False,
                temperature=temperature,
            )
        )

    def select_action(self, env: OmokEnvGUI) -> int:
        """Search the current GUI position and return one move."""

        alphazero_env = env.to_alphazero_env()
        root = self.mcts.run(alphazero_env, self.model)
        return self.mcts.select_action(root, temperature=self.mcts.config.temperature)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the Tkinter human-vs-AlphaZero GUI."""

    parser = argparse.ArgumentParser(
        description="Play Gomoku against a trained AlphaZero checkpoint in the Tkinter GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/alphazero",
        help="Checkpoint file or checkpoint directory. Directories prefer best_model.pt, then the latest trainer checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string, for example 'cpu' or 'cuda'.")
    parser.add_argument(
        "--human-color",
        type=str,
        default="black",
        help="Human color: 'black' or 'white'.",
    )
    parser.add_argument(
        "--simulations",
        type=_positive_int,
        default=50,
        help="Number of AlphaZero MCTS simulations per AI move.",
    )
    parser.add_argument(
        "--temperature",
        type=_non_negative_float,
        default=0.0,
        help="AI move-selection temperature. Use 0.0 for deterministic play.",
    )
    parser.add_argument(
        "--ai-move-delay",
        type=_non_negative_float,
        default=0.15,
        help="Small delay after AI moves so the GUI is easier to follow.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run one Tkinter human-vs-AlphaZero game."""

    args = parse_args(argv)
    args.device = _validate_device(args.device)
    human_player = _parse_human_color(args.human_color)
    ai_player = WHITE if human_player == BLACK else BLACK

    env = OmokEnvGUI(render_mode="human")
    human_agent = HumanAgent(env, name=f"Human ({_player_name(human_player)})")
    ai_agent = AlphaZeroAgent(
        args.checkpoint_path,
        device=args.device,
        num_simulations=args.simulations,
        temperature=args.temperature,
    )

    board, info = env.reset()
    env.render()
    print("=== Gomoku vs AlphaZero ===")
    print(f"checkpoint_path={ai_agent.checkpoint_path}")
    print(f"human_color={_player_name(human_player)}")
    print(f"ai_color={_player_name(ai_player)}")
    print(f"ai_num_simulations={args.simulations}")
    print(f"ai_temperature={args.temperature}")

    try:
        while not env.done:
            current_player = int(info["current_player"])
            if current_player == human_player:
                action = human_agent.select_action(env)
            else:
                action = ai_agent.select_action(env)

            board, _, _, _, info = env.step(action)
            env.render()
            if current_player == ai_player and args.ai_move_delay > 0.0:
                time.sleep(args.ai_move_delay)
    finally:
        if env.window is not None:
            env.window.update_idletasks()

    print("\n=== Game Result ===")
    if env.winner == DRAW:
        print("winner=draw")
    elif env.winner == human_player:
        print("winner=human")
    else:
        print("winner=alphazero")
    print(f"num_moves={env.move_count}")

    if env.window is not None:
        try:
            env.window.mainloop()
        finally:
            env.close()


if __name__ == "__main__":
    main()
