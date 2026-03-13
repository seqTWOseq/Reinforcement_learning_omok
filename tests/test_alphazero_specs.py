"""Tests for AlphaZero state, policy, and value specs."""

from __future__ import annotations

import numpy as np

from gomoku_ai.alphazero import (
    ACTION_SIZE,
    POLICY_SHAPE,
    STATE_DTYPE,
    STATE_SHAPE,
    GameRecord,
    GameStepSample,
    NetworkInputSpec,
    PolicyValueOutputSpec,
    build_game_step_sample,
    mask_policy_logits,
    normalize_visit_counts,
    policy_logits_to_probs,
    winner_to_value_target,
)
from gomoku_ai.env import BLACK, DRAW, WHITE, GomokuEnv


def test_state_spec_shape_dtype_matches_env() -> None:
    """The default network spec should match `GomokuEnv.encode_state()` output."""

    env = GomokuEnv()
    state = env.reset()
    del state

    encoded_state = env.encode_state()
    spec = NetworkInputSpec()

    assert encoded_state.shape == spec.state_shape == STATE_SHAPE
    assert encoded_state.dtype == spec.state_dtype == STATE_DTYPE


def test_winner_to_value_target() -> None:
    """Winner mapping should follow side-to-move perspective."""

    assert winner_to_value_target(BLACK, BLACK) == 1.0
    assert winner_to_value_target(BLACK, WHITE) == -1.0
    assert winner_to_value_target(DRAW, BLACK) == 0.0


def test_normalize_visit_counts() -> None:
    """Visit counts should normalize into a probability target."""

    visit_counts = np.zeros(ACTION_SIZE, dtype=np.float32)
    visit_counts[:3] = np.array([2.0, 3.0, 5.0], dtype=np.float32)

    normalized = normalize_visit_counts(visit_counts)

    assert normalized.shape == POLICY_SHAPE
    assert normalized.dtype == np.float32
    assert np.isclose(float(normalized.sum()), 1.0)
    assert np.all(normalized[3:] == 0.0)


def test_mask_policy_logits() -> None:
    """Invalid moves should be masked out with `-inf`."""

    logits = np.linspace(-1.0, 1.0, ACTION_SIZE, dtype=np.float32)
    valid_moves = np.zeros(ACTION_SIZE, dtype=bool)
    valid_moves[0] = True
    valid_moves[10] = True

    masked = mask_policy_logits(logits, valid_moves)

    assert masked.shape == POLICY_SHAPE
    assert np.isfinite(masked[0])
    assert np.isfinite(masked[10])
    assert np.isneginf(masked[1])


def test_policy_logits_to_probs() -> None:
    """Masked softmax should assign zero mass to invalid moves and sum to 1.0."""

    logits = np.zeros(ACTION_SIZE, dtype=np.float32)
    logits[3] = 2.0
    logits[7] = 1.0
    valid_moves = np.zeros(ACTION_SIZE, dtype=bool)
    valid_moves[3] = True
    valid_moves[7] = True

    probabilities = policy_logits_to_probs(logits, valid_moves)

    assert probabilities.shape == POLICY_SHAPE
    assert np.isclose(float(probabilities.sum()), 1.0)
    assert probabilities[3] > probabilities[7] > 0.0
    assert np.all(probabilities[~valid_moves] == 0.0)


def test_policy_logits_to_probs_single_valid_move() -> None:
    """A single legal move should receive probability 1.0 after masking."""

    logits = np.linspace(-1.0, 1.0, ACTION_SIZE, dtype=np.float32)
    valid_moves = np.zeros(ACTION_SIZE, dtype=bool)
    valid_moves[42] = True

    probabilities = policy_logits_to_probs(logits, valid_moves)

    assert np.isclose(float(probabilities.sum()), 1.0)
    assert probabilities[42] == np.float32(1.0)
    assert np.count_nonzero(probabilities) == 1


def test_build_game_step_sample() -> None:
    """Game step samples should capture the encoded state and metadata."""

    env = GomokuEnv()
    env.reset()
    policy_target = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy_target[5] = 1.0

    sample = build_game_step_sample(
        env=env,
        policy_target=policy_target,
        value_target=0.25,
        move_index=0,
        action_taken=5,
        game_id="game-001",
    )

    assert isinstance(sample, GameStepSample)
    assert sample.state.shape == STATE_SHAPE
    assert sample.state.dtype == np.float32
    assert sample.policy_target.shape == POLICY_SHAPE
    assert float(sample.value_target) == np.float32(0.25)
    assert sample.player_to_move == BLACK
    assert sample.action_taken == 5
    assert sample.game_id == "game-001"


def test_policy_value_output_spec_accepts_scalar_and_vector_value() -> None:
    """Policy/value output spec should normalize `(1,)` values into float32 scalars."""

    logits = np.zeros(ACTION_SIZE, dtype=np.float32)
    output = PolicyValueOutputSpec(policy_logits=logits, value=np.array([0.5], dtype=np.float32))

    assert output.policy_logits.shape == POLICY_SHAPE
    assert isinstance(output.value, np.float32)
    assert float(output.value) == np.float32(0.5)


def test_game_record_accepts_ai_turn_only_samples() -> None:
    """GameRecord should allow fewer samples than moves for human-play data."""

    env = GomokuEnv()
    env.reset()
    policy_target = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy_target[0] = 1.0
    sample = build_game_step_sample(
        env=env,
        policy_target=policy_target,
        value_target=1.0,
        move_index=0,
        action_taken=0,
        game_id="human-001",
    )

    record = GameRecord(
        game_id="human-001",
        moves=[0, 1, 2],
        winner=BLACK,
        source="human_play",
        samples=[sample],
    )

    assert record.source == "human_play"
    assert len(record.moves) == 3
    assert len(record.samples) == 1


def test_game_record_rejects_mismatched_sample_game_id() -> None:
    """GameRecord should reject samples that point to a different game id."""

    env = GomokuEnv()
    env.reset()
    policy_target = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy_target[0] = 1.0
    sample = build_game_step_sample(
        env=env,
        policy_target=policy_target,
        value_target=0.0,
        move_index=0,
        action_taken=0,
        game_id="different-game",
    )

    try:
        GameRecord(
            game_id="game-001",
            moves=[0],
            winner=BLACK,
            source="self_play",
            samples=[sample],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched sample.game_id.")


def test_invalid_shape_or_value_inputs_raise() -> None:
    """Invalid shapes, ranges, and value semantics should raise exceptions."""

    invalid_policy_shape = np.zeros(ACTION_SIZE - 1, dtype=np.float32)
    valid_policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    valid_policy[0] = 1.0
    valid_mask = np.ones(ACTION_SIZE, dtype=bool)
    env = GomokuEnv()
    env.reset()

    try:
        normalize_visit_counts(np.zeros(ACTION_SIZE, dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for zero-sum visit counts.")

    try:
        mask_policy_logits(np.zeros(ACTION_SIZE, dtype=np.float32), np.ones(ACTION_SIZE - 1, dtype=bool))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid valid_moves shape.")

    try:
        build_game_step_sample(
            env=env,
            policy_target=invalid_policy_shape,
            value_target=0.0,
            move_index=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid policy_target shape.")

    try:
        build_game_step_sample(
            env=env,
            policy_target=valid_policy,
            value_target=1.5,
            move_index=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for out-of-range value_target.")

    try:
        winner_to_value_target(DRAW, DRAW)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid player_to_move.")
