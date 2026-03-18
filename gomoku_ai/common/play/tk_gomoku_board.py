"""Tkinter Gomoku board used by human-vs-Athenan play entrypoints."""

from __future__ import annotations

import time
import tkinter as tk

from gomoku_ai.env import BLACK, DRAW, EMPTY, GomokuEnv, WHITE


def _player_name(player: int | None) -> str:
    if player == BLACK:
        return "black"
    if player == WHITE:
        return "white"
    if player == DRAW:
        return "draw"
    return "unknown"


class TkGomokuBoard:
    """Small Tkinter board that mirrors one `GomokuEnv` instance."""

    def __init__(self, env: GomokuEnv, *, title_prefix: str) -> None:
        self.env = env
        self.title_prefix = title_prefix
        self.cell_size = 40
        self.margin = 30
        self.window: tk.Tk | None = None
        self.canvas: tk.Canvas | None = None
        self._button_bar: tk.Frame | None = None
        self._quit_button: tk.Button | None = None
        self._clicked_action: int | None = None

    def render(self) -> None:
        """Create or refresh the board window from the current env state."""

        if self.window is None:
            self.window = tk.Tk()
            self.window.protocol("WM_DELETE_WINDOW", self.close)
            size = (self.env.board_size - 1) * self.cell_size + self.margin * 2
            self.canvas = tk.Canvas(self.window, width=size, height=size, bg="#D7A85A", highlightthickness=0)
            self.canvas.pack()
            self._button_bar = tk.Frame(self.window, bg="#D7A85A")
            self._button_bar.pack(fill="x", padx=8, pady=(0, 8))
            self._quit_button = tk.Button(self._button_bar, text="Exit", command=self.close, width=10)
            self._quit_button.pack(side="right")

        if self.window is None or self.canvas is None:
            raise RuntimeError("Tkinter board failed to initialize.")

        if self.env.done:
            state_text = "Draw" if self.env.winner == DRAW else f"{_player_name(self.env.winner).title()} wins"
        else:
            state_text = f"{_player_name(self.env.current_player).title()} to move"

        self.window.title(f"{self.title_prefix} - {state_text}")
        self.canvas.delete("all")

        for index in range(self.env.board_size):
            start = self.margin + index * self.cell_size
            end = self.margin + (self.env.board_size - 1) * self.cell_size
            self.canvas.create_line(self.margin, start, end, start, width=1)
            self.canvas.create_line(start, self.margin, start, end, width=1)

        for row in range(self.env.board_size):
            for col in range(self.env.board_size):
                value = int(self.env.board[row, col])
                if value == EMPTY:
                    continue
                x = self.margin + col * self.cell_size
                y = self.margin + row * self.cell_size
                radius = self.cell_size // 2 - 2
                fill = "black" if value == BLACK else "white"
                outline = "#C0392B" if self.env.last_move == (row, col) else "black"
                width = 3 if self.env.last_move == (row, col) else 1
                self.canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=fill,
                    outline=outline,
                    width=width,
                )

        self.canvas.create_text(
            self.margin,
            12,
            text=state_text,
            anchor="w",
            fill="#1F2937",
            font=("Segoe UI", 11, "bold"),
        )

        self.window.update_idletasks()
        self.window.update()

    def select_human_action(self) -> int:
        """Block until the user clicks one legal move on the board."""

        if self.window is None or self.canvas is None:
            raise RuntimeError("The board must be rendered before selecting a move.")

        self._clicked_action = None
        self.canvas.bind("<Button-1>", self._click_handler)

        try:
            while self._clicked_action is None:
                self.window.update_idletasks()
                self.window.update()
                time.sleep(0.05)
        except tk.TclError as exc:
            raise RuntimeError("The Gomoku window was closed during human input.") from exc
        finally:
            self.canvas.unbind("<Button-1>")

        return int(self._clicked_action)

    def keep_open(self) -> None:
        """Keep the final board visible until the window is closed."""

        if self.window is None:
            return
        try:
            self.window.mainloop()
        finally:
            self.close()

    def close(self) -> None:
        """Destroy the window if it exists."""

        if self.window is not None:
            self.window.destroy()
            self.window = None
            self.canvas = None
            self._button_bar = None
            self._quit_button = None

    def _click_handler(self, event: tk.Event[tk.Misc]) -> None:
        col = round((event.x - self.margin) / self.cell_size)
        row = round((event.y - self.margin) / self.cell_size)

        if not 0 <= row < self.env.board_size or not 0 <= col < self.env.board_size:
            return

        action = self.env.coord_to_action(row, col)
        if self.env.is_legal_action(action):
            self._clicked_action = int(action)
