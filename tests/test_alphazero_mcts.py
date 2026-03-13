"""Tests for AlphaZero-compatible MCTS search."""

from __future__ import annotations

import numpy as np

from gomoku_ai.alphazero import PolicyValueNet, PolicyValueNetConfig
from gomoku_ai.alphazero.mcts import MCTS, MCTSConfig, MCTSNode
from gomoku_ai.alphazero.mcts_utils import resolve_env_player_to_move, terminal_value_for_player
from gomoku_ai.alphazero.specs import ACTION_SIZE
from gomoku_ai.env import BLACK, BOARD_SIZE, DRAW, EMPTY, WHITE, GomokuEnv


class DummyPolicyValueNet(PolicyValueNet):
    """Deterministic test double for MCTS search."""

    def __init__(self, policy_logits: np.ndarray | None = None, value: float = 0.0) -> None:
        super().__init__(PolicyValueNetConfig(use_batch_norm=False))
        self.fixed_policy_logits = (
            np.zeros(ACTION_SIZE, dtype=np.float32)
            if policy_logits is None
            else np.asarray(policy_logits, dtype=np.float32)
        )
        self.fixed_value = float(value)
        self.predict_calls = 0

    def predict_single(
        self,
        state_np: np.ndarray,
        device: object | None = None,
        *,
        move_model: bool = False,
    ) -> tuple[np.ndarray, float]:
        del state_np
        del device
        del move_model
        self.predict_calls += 1
        return self.fixed_policy_logits.copy(), self.fixed_value


def _place_stones(
    env: GomokuEnv,
    black_coords: list[tuple[int, int]],
    white_coords: list[tuple[int, int]],
    *,
    current_player: int,
) -> None:
    """Populate a deterministic board fixture directly for tests."""

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
    """Return a full 15x15 board pattern with no five-in-a-row for either side."""

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


def test_mcts_run_returns_root_with_children() -> None:
    """Running MCTS should expand the root without mutating the caller's env."""

    env = GomokuEnv()
    env.reset()
    board_before = env.board.copy()
    current_player_before = env.current_player
    move_count_before = env.move_count
    model = DummyPolicyValueNet(value=0.2)
    mcts = MCTS(MCTSConfig(num_simulations=8, add_root_noise=False))

    root = mcts.run(env, model)

    assert root.player_to_move == BLACK
    assert root.expanded() is True
    assert len(root.children) == ACTION_SIZE
    assert root.visit_count == 8
    assert np.array_equal(env.board, board_before)
    assert env.current_player == current_player_before
    assert env.move_count == move_count_before
    assert model.predict_calls >= 1


def test_get_action_probs_shape_and_sum() -> None:
    """Visit-count probabilities should match the fixed policy target contract."""

    env = GomokuEnv()
    env.reset()
    env.apply_move(0)
    env.apply_move(1)
    logits = np.zeros(ACTION_SIZE, dtype=np.float32)
    logits[env.coord_to_action(7, 7)] = 3.0
    model = DummyPolicyValueNet(policy_logits=logits, value=0.0)
    mcts = MCTS(MCTSConfig(num_simulations=12, add_root_noise=False, temperature=1.0))

    root = mcts.run(env, model)
    probs = mcts.get_action_probs(root)

    assert probs.shape == (ACTION_SIZE,)
    assert probs.dtype == np.float32
    assert np.isclose(float(probs.sum()), 1.0)
    assert probs[0] == 0.0
    assert probs[1] == 0.0


def test_select_action_returns_legal_move() -> None:
    """Selected actions must be legal under the root environment's valid-move mask."""

    env = GomokuEnv()
    env.reset()
    env.apply_move(0)
    env.apply_move(1)
    model = DummyPolicyValueNet(value=0.0)
    mcts = MCTS(MCTSConfig(num_simulations=10, add_root_noise=False, temperature=1.0))

    root = mcts.run(env, model)
    action = mcts.select_action(root, temperature=1.0)

    assert env.get_valid_moves()[action]


def test_terminal_value_helper_win_and_draw() -> None:
    """Terminal helpers must respect GomokuEnv terminal/current_player semantics."""

    win_env = GomokuEnv()
    _place_stones(
        win_env,
        black_coords=[(7, 3), (7, 4), (7, 5), (7, 6)],
        white_coords=[(6, 0), (6, 1), (6, 2), (6, 3)],
        current_player=BLACK,
    )
    win_env.apply_move(win_env.coord_to_action(7, 7))
    win_player_to_move = resolve_env_player_to_move(win_env)

    assert win_env.done is True
    assert win_env.winner == BLACK
    assert win_env.current_player == BLACK
    assert win_player_to_move == WHITE
    assert terminal_value_for_player(win_env, win_player_to_move) == -1.0

    draw_env = GomokuEnv()
    board = _make_draw_pattern_board()
    last_row, last_col = BOARD_SIZE - 1, BOARD_SIZE - 1
    final_player = int(board[last_row, last_col])
    draw_env.board = board.copy()
    draw_env.board[last_row, last_col] = EMPTY
    draw_env.current_player = final_player
    draw_env.last_move = None
    draw_env.winner = None
    draw_env.done = False
    draw_env.move_count = ACTION_SIZE - 1
    draw_env.apply_move(draw_env.coord_to_action(last_row, last_col))
    draw_player_to_move = resolve_env_player_to_move(draw_env)

    assert draw_env.done is True
    assert draw_env.winner == DRAW
    assert terminal_value_for_player(draw_env, draw_player_to_move) == 0.0


def test_backpropagation_sign_flip_behavior() -> None:
    """Backpropagation must flip the value sign at each level in the search path."""

    mcts = MCTS(MCTSConfig(num_simulations=1, add_root_noise=False))
    root = MCTSNode(prior=1.0, player_to_move=BLACK)
    child = MCTSNode(prior=0.5, player_to_move=WHITE, action_taken=0)
    leaf = MCTSNode(prior=0.5, player_to_move=BLACK, action_taken=1)

    mcts._backpropagate([root, child, leaf], 0.75)

    assert root.visit_count == 1
    assert child.visit_count == 1
    assert leaf.visit_count == 1
    assert np.isclose(root.value_sum, 0.75)
    assert np.isclose(child.value_sum, -0.75)
    assert np.isclose(leaf.value_sum, 0.75)


def test_root_dirichlet_noise_changes_priors() -> None:
    """Root Dirichlet noise should perturb the original policy priors."""

    env = GomokuEnv()
    env.reset()
    model = DummyPolicyValueNet(value=0.0)

    np.random.seed(7)
    noisy_root = MCTS(MCTSConfig(num_simulations=4, add_root_noise=True)).run(env, model)
    noiseless_root = MCTS(MCTSConfig(num_simulations=4, add_root_noise=False)).run(env, model)

    noisy_priors = np.array([noisy_root.children[action].prior for action in sorted(noisy_root.children)], dtype=np.float32)
    noiseless_priors = np.array(
        [noiseless_root.children[action].prior for action in sorted(noiseless_root.children)],
        dtype=np.float32,
    )

    assert np.isclose(float(noisy_priors.sum()), 1.0)
    assert np.isclose(float(noiseless_priors.sum()), 1.0)
    assert not np.allclose(noisy_priors, noiseless_priors)


def test_temperature_zero_is_deterministic() -> None:
    """Temperature zero should return an argmax one-hot distribution and action."""

    mcts = MCTS(MCTSConfig(num_simulations=1, add_root_noise=False, temperature=0.0))
    root = MCTSNode(prior=1.0, player_to_move=BLACK)
    root.children[3] = MCTSNode(prior=0.5, player_to_move=WHITE, action_taken=3, visit_count=7)
    root.children[9] = MCTSNode(prior=0.5, player_to_move=WHITE, action_taken=9, visit_count=2)

    probs = mcts.get_action_probs(root, temperature=0.0)
    action = mcts.select_action(root, temperature=0.0)

    assert probs[3] == 1.0
    assert np.count_nonzero(probs) == 1
    assert action == 3


def test_terminal_root_skips_network_call() -> None:
    """Terminal roots should use the terminal helper without querying the model."""

    env = GomokuEnv()
    _place_stones(
        env,
        black_coords=[(7, 3), (7, 4), (7, 5), (7, 6)],
        white_coords=[(6, 0), (6, 1), (6, 2), (6, 3)],
        current_player=BLACK,
    )
    env.apply_move(env.coord_to_action(7, 7))

    model = DummyPolicyValueNet(value=0.0)
    root = MCTS(MCTSConfig(num_simulations=5, add_root_noise=False)).run(env, model)

    assert root.is_terminal is True
    assert root.terminal_value == -1.0
    assert model.predict_calls == 0


def test_evaluate_leaf_rejects_expanded_non_terminal_node() -> None:
    """Expanded non-terminal leaves should be rejected instead of silently reused."""

    env = GomokuEnv()
    env.reset()
    model = DummyPolicyValueNet(value=0.0)
    mcts = MCTS(MCTSConfig(num_simulations=1, add_root_noise=False))
    node = MCTSNode(prior=1.0, player_to_move=BLACK)
    node.children[0] = MCTSNode(prior=1.0, player_to_move=WHITE, action_taken=0)

    try:
        mcts._evaluate_leaf(node, env, model)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError for expanded non-terminal leaf evaluation.")
