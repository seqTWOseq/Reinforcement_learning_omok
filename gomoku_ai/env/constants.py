"""Shared constants for the reusable Gomoku environment."""

from __future__ import annotations

from typing import Final

BOARD_SIZE: Final[int] = 15
EMPTY: Final[int] = 0
BLACK: Final[int] = 1
WHITE: Final[int] = 2
DRAW: Final[int] = -1
DIRECTIONS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
)
STONE_SYMBOLS: Final[dict[int, str]] = {
    EMPTY: ".",
    BLACK: "X",
    WHITE: "O",
}
