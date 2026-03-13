"""Core Gomoku environment for reusable game-rule handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gomoku_ai.env.constants import BLACK, BOARD_SIZE, DIRECTIONS, DRAW, EMPTY, STONE_SYMBOLS, WHITE


@dataclass(frozen=True)
class MoveResult:
    """Structured result returned after applying a move."""

    action: int
    row: int
    col: int
    player: int
    next_player: int | None
    winner: int | None
    done: bool
    reason: str
    move_count: int
    last_move: tuple[int, int] | None

    def as_dict(self) -> dict[str, Any]:
        """Convert the immutable result object into a plain dictionary."""

        return {
            "action": self.action,
            "row": self.row,
            "col": self.col,
            "player": self.player,
            "next_player": self.next_player,
            "winner": self.winner,
            "done": self.done,
            "reason": self.reason,
            "move_count": self.move_count,
            "last_move": self.last_move,
        }


class GomokuEnv:
    """A reusable 15x15 free-style Gomoku rule engine.

    The environment only manages game state and deterministic rules. It does not
    implement rewards, MCTS, neural networks, or self-play logic so that higher
    level algorithms can reuse the same rule engine.

    `board` is intentionally exposed for inspection and deterministic test
    fixtures, but production gameplay should use `reset()` and `apply_move()`
    rather than mutating the board directly.
    """

    def __init__(self, board_size: int = BOARD_SIZE) -> None:
        """Initialize a new Gomoku environment."""

        self.board_size = board_size
        self.board: np.ndarray = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player: int = BLACK
        self.last_move: tuple[int, int] | None = None
        self.winner: int | None = None
        self.done: bool = False
        self.move_count: int = 0

    def reset(self) -> np.ndarray:
        """Reset the environment to the initial empty-board state."""

        self.board.fill(EMPTY)
        self.current_player = BLACK
        self.last_move = None
        self.winner = None
        self.done = False
        self.move_count = 0
        return self.board.copy()

    def clone(self) -> "GomokuEnv":
        """Return a deep copy of the current environment state.

        The clone preserves raw state fields exactly, including any manually
        prepared board used in tests or debugging.
        """

        cloned = GomokuEnv(board_size=self.board_size)
        cloned.board = self.board.copy()
        cloned.current_player = self.current_player
        cloned.last_move = self.last_move
        cloned.winner = self.winner
        cloned.done = self.done
        cloned.move_count = self.move_count
        return cloned

    def action_to_coord(self, action: int) -> tuple[int, int]:
        """Convert a flat action index into a `(row, col)` board coordinate."""

        if not 0 <= action < self.board_size * self.board_size:
            raise ValueError(
                f"Action {action} is out of range for a {self.board_size}x{self.board_size} board."
            )
        return divmod(action, self.board_size)

    def coord_to_action(self, row: int, col: int) -> int:
        """Convert a `(row, col)` board coordinate into a flat action index."""

        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            raise ValueError(
                f"Coordinate {(row, col)} is out of range for a {self.board_size}x{self.board_size} board."
            )
        return row * self.board_size + col

    def get_valid_moves(self) -> np.ndarray:
        """Return a flat boolean mask indicating which moves are currently legal."""

        return (self.board.reshape(-1) == EMPTY).astype(bool, copy=False)

    def apply_move(self, action: int) -> dict[str, Any]:
        """Apply a move for the current player and return the resulting metadata.

        Normal callers should use this method instead of mutating `board`
        directly so that `last_move`, `winner`, `done`, and `move_count` stay
        consistent.

        Raises:
            RuntimeError: If a move is attempted after the game has ended.
            ValueError: If the action is out of bounds or targets an occupied cell.
        """

        if self.done:
            raise RuntimeError("Cannot apply a move after the game has already ended.")

        row, col = self.action_to_coord(action)
        if self.board[row, col] != EMPTY:
            raise ValueError(f"Cell {(row, col)} is already occupied.")

        player = self.current_player
        self.board[row, col] = player
        self.last_move = (row, col)
        self.move_count += 1

        if self.check_win_from_move(row, col, player):
            self.done = True
            self.winner = player
            reason = "win"
        elif self.move_count == self.board_size * self.board_size:
            self.done = True
            self.winner = DRAW
            reason = "draw"
        else:
            self.current_player = self._opponent_of(player)
            reason = "ongoing"

        result = MoveResult(
            action=action,
            row=row,
            col=col,
            player=player,
            next_player=None if self.done else self.current_player,
            winner=self.winner,
            done=self.done,
            reason=reason,
            move_count=self.move_count,
            last_move=self.last_move,
        )
        return result.as_dict()

    def check_win_from_move(self, row: int, col: int, player: int) -> bool:
        """Check whether the given move created a five-or-more line for `player`."""

        if self.board[row, col] != player:
            return False

        for delta_row, delta_col in DIRECTIONS:
            count = 1
            count += self._count_direction(row, col, player, delta_row, delta_col)
            count += self._count_direction(row, col, player, -delta_row, -delta_col)
            if count >= 5:
                return True
        return False

    def is_terminal(self) -> bool:
        """Return `True` when the game has reached a terminal state."""

        return self.done

    def encode_state(self) -> np.ndarray:
        """Encode the state from the current player's perspective.

        Channels:
            0. Stones belonging to the player who is about to move
            1. Stones belonging to that player's opponent
            2. Last move plane
            3. Absolute black-to-move plane (`1.0` if the side to move is black,
               `0.0` if the side to move is white)

        Channels 0 and 1 are relative to the side to move. Channel 3 is an
        absolute color indicator so downstream models can distinguish whether the
        relative "current player" corresponds to black or white.
        """

        current_player_stones = (self.board == self.current_player).astype(np.float32)
        opponent_stones = (
            (self.board != EMPTY) & (self.board != self.current_player)
        ).astype(np.float32)
        last_move_plane = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        if self.last_move is not None:
            last_row, last_col = self.last_move
            last_move_plane[last_row, last_col] = 1.0
        black_to_move_plane = np.full(
            (self.board_size, self.board_size),
            1.0 if self.current_player == BLACK else 0.0,
            dtype=np.float32,
        )
        return np.stack(
            (
                current_player_stones,
                opponent_stones,
                last_move_plane,
                black_to_move_plane,
            ),
            axis=0,
        ).astype(np.float32, copy=False)

    def render(self) -> str:
        """Return an indexed ASCII rendering of the current board."""

        header = "   " + " ".join(f"{column:>2}" for column in range(self.board_size))
        rows = [header]
        for row_index in range(self.board_size):
            stones = " ".join(STONE_SYMBOLS[int(value)] for value in self.board[row_index])
            rows.append(f"{row_index:>2} {stones}")
        return "\n".join(rows)

    def _count_direction(
        self,
        row: int,
        col: int,
        player: int,
        delta_row: int,
        delta_col: int,
    ) -> int:
        """Count contiguous stones for `player` in one direction from `(row, col)`."""

        count = 0
        next_row = row + delta_row
        next_col = col + delta_col
        while (
            0 <= next_row < self.board_size
            and 0 <= next_col < self.board_size
            and self.board[next_row, next_col] == player
        ):
            count += 1
            next_row += delta_row
            next_col += delta_col
        return count

    @staticmethod
    def _opponent_of(player: int) -> int:
        """Return the opposite player color."""

        return WHITE if player == BLACK else BLACK
