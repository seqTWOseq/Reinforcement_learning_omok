"""Unit tests for the reusable Gomoku environment."""

from __future__ import annotations

import numpy as np

from gomoku_ai.env import BLACK, BOARD_SIZE, DRAW, EMPTY, WHITE, GomokuEnv


def _place_stones(
    env: GomokuEnv,
    black_coords: list[tuple[int, int]],
    white_coords: list[tuple[int, int]],
    *,
    current_player: int,
) -> None:
    """Populate an environment with a controlled board position.

    Direct board mutation is acceptable inside tests to build deterministic
    fixtures. Runtime code should prefer `reset()` and `apply_move()`.
    """

    env.reset()
    for row, col in black_coords:
        env.board[row, col] = BLACK
    for row, col in white_coords:
        env.board[row, col] = WHITE
    env.current_player = current_player
    env.last_move = None
    env.winner = None
    env.done = False
    env.move_count = len(black_coords) + len(white_coords)


def _make_draw_pattern_board() -> np.ndarray:
    """Return a full 15x15 board pattern with no five-in-a-row for either side.

    The 4x4 tile below was chosen because repeating it across the full board
    avoids creating any horizontal, vertical, or diagonal five-in-a-row.
    That makes it suitable for a deterministic draw fixture.
    """

    tile = (
        (BLACK, BLACK, BLACK, WHITE),
        (BLACK, BLACK, BLACK, WHITE),
        (BLACK, WHITE, BLACK, BLACK),
        (WHITE, BLACK, WHITE, WHITE),
    )
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            board[row, col] = tile[row % 4][col % 4]
    return board


def _has_any_five(env: GomokuEnv) -> bool:
    """Return `True` if the current board already contains a winning line."""

    for row in range(env.board_size):
        for col in range(env.board_size):
            player = int(env.board[row, col])
            if player != EMPTY and env.check_win_from_move(row, col, player):
                return True
    return False


def test_reset_initial_state() -> None:
    """A reset environment should start with an empty board and black to move."""

    env = GomokuEnv()

    board = env.reset()

    assert board.shape == (BOARD_SIZE, BOARD_SIZE)
    assert board.dtype == np.int8
    assert np.count_nonzero(board) == 0
    assert env.current_player == BLACK
    assert env.last_move is None
    assert env.winner is None
    assert env.done is False
    assert env.move_count == 0


def test_apply_first_move() -> None:
    """The first move should place a black stone and pass the turn to white."""

    env = GomokuEnv()
    env.reset()

    result = env.apply_move(0)

    assert env.board[0, 0] == BLACK
    assert env.current_player == WHITE
    assert env.last_move == (0, 0)
    assert env.move_count == 1
    assert env.done is False
    assert result["player"] == BLACK
    assert result["next_player"] == WHITE
    assert result["done"] is False
    assert result["reason"] == "ongoing"


def test_invalid_move_raises() -> None:
    """Occupied cells and out-of-range actions should raise `ValueError`."""

    env = GomokuEnv()
    env.reset()
    env.apply_move(10)

    try:
        env.apply_move(10)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for an occupied cell.")

    try:
        env.apply_move(BOARD_SIZE * BOARD_SIZE)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for an out-of-range action.")


def test_horizontal_win() -> None:
    """Five consecutive stones in a row should end the game with a black win."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 3), (7, 4), (7, 5), (7, 6)],
        white_coords=[(6, 0), (6, 1), (6, 2), (6, 3)],
        current_player=BLACK,
    )

    result = env.apply_move(env.coord_to_action(7, 7))

    assert env.done is True
    assert env.winner == BLACK
    assert result["winner"] == BLACK
    assert result["reason"] == "win"


def test_vertical_win() -> None:
    """Five consecutive stones in a column should end the game with a black win."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(3, 7), (4, 7), (5, 7), (6, 7)],
        white_coords=[(0, 0), (0, 1), (0, 2), (0, 3)],
        current_player=BLACK,
    )

    env.apply_move(env.coord_to_action(7, 7))

    assert env.done is True
    assert env.winner == BLACK


def test_diagonal_win_down_right() -> None:
    """A down-right diagonal of five should be detected as a win."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(2, 2), (3, 3), (4, 4), (5, 5)],
        white_coords=[(0, 5), (0, 6), (0, 7), (0, 8)],
        current_player=BLACK,
    )

    env.apply_move(env.coord_to_action(6, 6))

    assert env.done is True
    assert env.winner == BLACK


def test_diagonal_win_down_left() -> None:
    """A down-left diagonal of five should be detected as a win."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(2, 10), (3, 9), (4, 8), (5, 7)],
        white_coords=[(0, 0), (0, 1), (0, 2), (0, 3)],
        current_player=BLACK,
    )

    env.apply_move(env.coord_to_action(6, 6))

    assert env.done is True
    assert env.winner == BLACK


def test_draw_detection() -> None:
    """A full board without a winner should end as a draw."""

    env = GomokuEnv()
    board = _make_draw_pattern_board()

    last_row, last_col = BOARD_SIZE - 1, BOARD_SIZE - 1
    final_player = int(board[last_row, last_col])

    filled_preview = GomokuEnv()
    filled_preview.board = board.copy()
    filled_preview.move_count = BOARD_SIZE * BOARD_SIZE
    assert _has_any_five(filled_preview) is False

    env.board = board.copy()
    env.board[last_row, last_col] = EMPTY
    env.current_player = final_player
    env.last_move = None
    env.winner = None
    env.done = False
    env.move_count = BOARD_SIZE * BOARD_SIZE - 1

    result = env.apply_move(env.coord_to_action(last_row, last_col))

    assert env.done is True
    assert env.winner == DRAW
    assert result["winner"] == DRAW
    assert result["reason"] == "draw"


def test_encode_state_shape() -> None:
    """Encoded states should follow the `(4, 15, 15)` float32 specification."""

    env = GomokuEnv()
    env.reset()
    env.apply_move(env.coord_to_action(0, 0))
    env.apply_move(env.coord_to_action(1, 1))

    encoded = env.encode_state()

    assert encoded.shape == (4, BOARD_SIZE, BOARD_SIZE)
    assert encoded.dtype == np.float32
    assert encoded[0, 0, 0] == 1.0
    assert encoded[1, 1, 1] == 1.0
    assert encoded[2, 1, 1] == 1.0
    assert np.all(encoded[3] == 1.0)


def test_encode_state_white_turn_perspective() -> None:
    """White-to-move states should stay relative while channel 3 marks white turn."""

    env = GomokuEnv()
    env.reset()
    env.apply_move(env.coord_to_action(4, 4))

    encoded = env.encode_state()

    assert env.current_player == WHITE
    assert encoded[0, 4, 4] == 0.0
    assert encoded[1, 4, 4] == 1.0
    assert encoded[2, 4, 4] == 1.0
    assert np.all(encoded[3] == 0.0)


def test_action_coord_roundtrip() -> None:
    """Action and coordinate conversions should be exact inverses."""

    env = GomokuEnv()
    action = env.coord_to_action(9, 4)
    row, col = env.action_to_coord(action)

    assert action == 9 * BOARD_SIZE + 4
    assert (row, col) == (9, 4)
