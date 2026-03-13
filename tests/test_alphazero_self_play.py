"""Tests for AlphaZero self-play game generation."""

from __future__ import annotations

import numpy as np

from gomoku_ai.alphazero import (
    GameRecord,
    MCTSConfig,
    PolicyValueNet,
    PolicyValueNetConfig,
    SelfPlayConfig,
    SelfPlayGameGenerator,
    build_self_play_game_id,
    get_temperature_for_move,
)
from gomoku_ai.env import BLACK, BOARD_SIZE, GomokuEnv, WHITE


class ScriptedPolicyValueNet(PolicyValueNet):
    """Deterministic model that drives a short scripted self-play game."""

    def __init__(self) -> None:
        super().__init__(PolicyValueNetConfig(use_batch_norm=False))

    def predict_single(
        self,
        state_np: np.ndarray,
        device: object | None = None,
        *,
        move_model: bool = False,
    ) -> tuple[np.ndarray, float]:
        del device
        del move_model

        state = np.asarray(state_np, dtype=np.float32)
        occupied = (state[0] + state[1]) > 0.5
        black_to_move = bool(state[3, 0, 0] == 1.0)
        target_coords = (
            [(7, 0), (7, 1), (7, 2), (7, 3), (7, 4)]
            if black_to_move
            else [(0, 10), (0, 11), (0, 12), (0, 13), (0, 14)]
        )

        logits = np.full((BOARD_SIZE * BOARD_SIZE,), -100.0, dtype=np.float32)
        chosen_action: int | None = None
        for row, col in target_coords:
            if not occupied[row, col]:
                chosen_action = row * BOARD_SIZE + col
                break
        if chosen_action is None:
            empty_indices = np.flatnonzero(~occupied.reshape(-1))
            chosen_action = int(empty_indices[0])

        logits[chosen_action] = 100.0
        return logits, 0.25


def _play_scripted_game() -> GameRecord:
    """Return a deterministic self-play game record for test reuse."""

    generator = SelfPlayGameGenerator(
        config=SelfPlayConfig(
            opening_temperature_moves=0,
            opening_temperature=1.0,
            late_temperature=0.0,
            use_root_noise=False,
            game_id_prefix="selfplay",
        ),
        mcts_config=MCTSConfig(num_simulations=2, add_root_noise=False, temperature=0.0),
    )
    return generator.play_one_game(ScriptedPolicyValueNet())


def test_play_one_game_returns_game_record() -> None:
    """Self-play should return a populated `GameRecord`."""

    record = _play_scripted_game()

    assert isinstance(record, GameRecord)
    assert record.game_id.startswith("selfplay-")
    assert record.source == "self_play"


def test_self_play_samples_match_moves_length_or_expected_count() -> None:
    """This generator should emit one sample per move for self-play."""

    record = _play_scripted_game()

    assert len(record.samples) == len(record.moves)
    assert record.metadata["num_moves"] == len(record.moves)


def test_policy_targets_sum_to_one() -> None:
    """Each self-play sample policy target should be a normalized distribution."""

    record = _play_scripted_game()

    for sample in record.samples:
        assert np.isclose(float(sample.policy_target.sum()), 1.0)


def test_value_targets_follow_winner_perspective() -> None:
    """Value targets should match the final winner from each sample's perspective."""

    record = _play_scripted_game()

    assert record.winner == BLACK
    for sample in record.samples:
        expected_value = 1.0 if sample.player_to_move == BLACK else -1.0
        assert float(sample.value_target) == expected_value


def test_temperature_schedule_behavior() -> None:
    """The self-play schedule should use stochastic openings and deterministic late play by default."""

    config = SelfPlayConfig()

    assert config.opening_temperature_moves == 10
    assert config.opening_temperature == 1.0
    assert config.late_temperature == 0.0
    assert config.use_root_noise is True
    assert config.game_id_prefix == "selfplay"
    assert get_temperature_for_move(0, config) == 1.0
    assert get_temperature_for_move(9, config) == 1.0
    assert get_temperature_for_move(10, config) == 0.0


def test_game_record_source_is_self_play() -> None:
    """Generated records must be marked as self-play data."""

    record = _play_scripted_game()

    assert record.source == "self_play"


def test_metadata_contains_required_keys_and_game_id_prefix() -> None:
    """Metadata must include the user-required keys and the game id must use the selfplay prefix."""

    record = _play_scripted_game()

    assert record.game_id.startswith("selfplay-")
    assert "num_moves" in record.metadata
    assert "use_root_noise" in record.metadata
    assert "opening_temperature_moves" in record.metadata


def test_move_indices_and_moves_are_consistent() -> None:
    """Move indices and stored move list should align one-to-one with samples."""

    record = _play_scripted_game()

    for index, sample in enumerate(record.samples):
        assert sample.move_index == index
        assert record.moves[index] == sample.action_taken


def test_state_is_captured_before_action_is_applied() -> None:
    """Each stored state should match the environment state before its move is played."""

    record = _play_scripted_game()
    env = GomokuEnv()
    env.reset()

    for sample in record.samples:
        expected_state = env.encode_state()
        assert np.array_equal(sample.state, expected_state)
        assert env.get_valid_moves()[sample.action_taken]
        env.apply_move(sample.action_taken)


def test_build_self_play_game_id_uses_prefix() -> None:
    """Game id helper should prepend the configured prefix."""

    game_id = build_self_play_game_id("selfplay")

    assert game_id.startswith("selfplay-")
