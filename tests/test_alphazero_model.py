"""Tests for the AlphaZero policy/value network implementation."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from gomoku_ai.alphazero import (
    ACTION_SIZE,
    STATE_SHAPE,
    PolicyValueNet,
    PolicyValueNetConfig,
    count_parameters,
    load_policy_value_net,
    predict_single,
    save_policy_value_net,
)
from gomoku_ai.env import GomokuEnv


def test_policy_value_net_forward_shapes() -> None:
    """Forward pass should return the fixed policy and value shapes for multiple batch sizes."""

    model = PolicyValueNet()

    batch_one = torch.zeros((1, *STATE_SHAPE), dtype=torch.float32)
    batch_many = torch.zeros((3, *STATE_SHAPE), dtype=torch.float32)

    policy_one, value_one = model(batch_one)
    policy_many, value_many = model(batch_many)

    assert policy_one.shape == (1, ACTION_SIZE)
    assert value_one.shape == (1, 1)
    assert policy_many.shape == (3, ACTION_SIZE)
    assert value_many.shape == (3, 1)
    assert policy_one.dtype == torch.float32
    assert value_one.dtype == torch.float32


def test_policy_value_net_rejects_invalid_input_shape() -> None:
    """Shape or dtype mismatches should raise `ValueError` before the forward pass."""

    model = PolicyValueNet()

    try:
        model(torch.zeros(STATE_SHAPE, dtype=torch.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for missing batch dimension.")

    try:
        model(torch.zeros((1, 5, STATE_SHAPE[1], STATE_SHAPE[2]), dtype=torch.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid channel count.")

    try:
        model(torch.zeros((1, *STATE_SHAPE), dtype=torch.float64))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid input dtype.")


def test_predict_single_output_shapes() -> None:
    """Single-state prediction should return a flat logits vector and a Python float."""

    env = GomokuEnv()
    env.reset()
    model = PolicyValueNet()

    policy_logits, value = predict_single(model, env.encode_state())

    assert isinstance(policy_logits, np.ndarray)
    assert policy_logits.shape == (ACTION_SIZE,)
    assert policy_logits.dtype == np.float32
    assert isinstance(value, float)


def test_predict_single_restores_training_mode() -> None:
    """Single-state prediction should restore the model's prior training mode."""

    env = GomokuEnv()
    env.reset()
    model = PolicyValueNet()
    model.train()

    predict_single(model, env.encode_state())

    assert model.training is True


def test_predict_single_rejects_nan_state() -> None:
    """Inference should reject states containing NaN values."""

    env = GomokuEnv()
    env.reset()
    state = env.encode_state()
    state[0, 0, 0] = np.nan
    model = PolicyValueNet()

    try:
        predict_single(model, state)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for NaN-containing state input.")


def test_value_output_in_range() -> None:
    """Tanh value head should constrain outputs to the `[-1, 1]` range."""

    model = PolicyValueNet()
    inputs = torch.randn((4, *STATE_SHAPE), dtype=torch.float32)

    _, values = model(inputs)

    assert torch.all(values <= 1.0)
    assert torch.all(values >= -1.0)


def test_count_parameters_positive() -> None:
    """Parameter count should be a positive integer for the baseline model."""

    model = PolicyValueNet()

    assert count_parameters(model) > 0


def test_save_and_load_roundtrip() -> None:
    """Saving and loading should preserve the config and numeric outputs."""

    model = PolicyValueNet(PolicyValueNetConfig())
    model.eval()
    sample_input = torch.randn((2, *STATE_SHAPE), dtype=torch.float32)

    with torch.no_grad():
        original_policy, original_value = model(sample_input)

    checkpoint_path = Path.cwd() / f"policy_value_net_test_{uuid4().hex}.pt"
    try:
        save_policy_value_net(model, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert checkpoint["format_version"] == 1
        assert checkpoint["model_type"] == "PolicyValueNet"
        loaded_model = load_policy_value_net(checkpoint_path, device="cpu")
        loaded_model.eval()

        with torch.no_grad():
            loaded_policy, loaded_value = loaded_model(sample_input)
    finally:
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    assert loaded_model.config == model.config
    assert torch.allclose(original_policy, loaded_policy)
    assert torch.allclose(original_value, loaded_value)
