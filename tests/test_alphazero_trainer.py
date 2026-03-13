"""Tests for the simple single-process AlphaZero trainer."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from gomoku_ai.alphazero import (
    AlphaZeroSampleDataset,
    AlphaZeroTrainer,
    GameRecord,
    GameStepSample,
    MCTSConfig,
    PolicyValueNet,
    PolicyValueNetConfig,
    RecentGameBuffer,
    SelfPlayConfig,
    TrainerConfig,
    compute_policy_loss,
    compute_value_loss,
)
from gomoku_ai.env import BLACK, BOARD_SIZE, GomokuEnv, WHITE


class ScriptedPolicyValueNet(PolicyValueNet):
    """Deterministic model for fast trainer and self-play tests."""

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


def _make_sample(
    move_index: int,
    player_to_move: int,
    action_taken: int,
    value_target: float,
    *,
    game_id: str = "sample-game",
) -> GameStepSample:
    """Build one valid training sample for dataset and buffer tests."""

    env = GomokuEnv()
    env.reset()
    if move_index > 0:
        env.apply_move(0)
    if player_to_move != env.current_player:
        env.apply_move(1)
    state = env.encode_state()
    policy_target = np.zeros((BOARD_SIZE * BOARD_SIZE,), dtype=np.float32)
    policy_target[action_taken] = 1.0
    return GameStepSample(
        state=state,
        policy_target=policy_target,
        value_target=value_target,
        player_to_move=player_to_move,
        move_index=move_index,
        action_taken=action_taken,
        game_id=game_id,
    )


def _make_record_with_n_samples(sample_count: int) -> GameRecord:
    """Build a synthetic record with `sample_count` samples."""

    game_id = f"record-{sample_count}"
    samples: list[GameStepSample] = []
    moves: list[int] = []
    for index in range(sample_count):
        action_taken = (index + 2) % (BOARD_SIZE * BOARD_SIZE)
        player_to_move = BLACK if index % 2 == 0 else WHITE
        samples.append(
            _make_sample(
                index,
                player_to_move,
                action_taken,
                1.0 if player_to_move == BLACK else -1.0,
                game_id=game_id,
            )
        )
        moves.append(action_taken)
    return GameRecord(
        game_id=game_id,
        moves=moves,
        winner=BLACK,
        source="self_play",
        samples=samples,
        metadata={"num_moves": sample_count},
    )


def _make_test_trainer(checkpoint_dir: str) -> AlphaZeroTrainer:
    """Construct a fast trainer configuration for tests."""

    return AlphaZeroTrainer(
        model=ScriptedPolicyValueNet(),
        trainer_config=TrainerConfig(
            num_self_play_games_per_cycle=2,
            max_buffer_samples=2000,
            batch_size=4,
            epochs_per_cycle=1,
            learning_rate=1e-3,
            weight_decay=1e-4,
            checkpoint_dir=checkpoint_dir,
            device="cpu",
        ),
        self_play_config=SelfPlayConfig(
            opening_temperature_moves=0,
            opening_temperature=1.0,
            late_temperature=0.0,
            use_root_noise=False,
            game_id_prefix="selfplay",
        ),
        mcts_config=MCTSConfig(
            num_simulations=2,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            add_root_noise=False,
            temperature=0.0,
        ),
    )


def test_dataset_returns_expected_tensor_shapes() -> None:
    """Dataset items should match the fixed tensor contract."""

    dataset = AlphaZeroSampleDataset([_make_sample(0, BLACK, 5, 1.0)])
    state, policy_target, value_target = dataset[0]

    assert state.shape == (4, BOARD_SIZE, BOARD_SIZE)
    assert policy_target.shape == (BOARD_SIZE * BOARD_SIZE,)
    assert value_target.shape == (1,)
    assert state.dtype == torch.float32
    assert policy_target.dtype == torch.float32
    assert value_target.dtype == torch.float32


def test_recent_game_buffer_trims_samples() -> None:
    """Recent buffer must trim by sample count, not by game count."""

    buffer = RecentGameBuffer(max_buffer_samples=5)
    buffer.add_game(_make_record_with_n_samples(3))
    buffer.add_game(_make_record_with_n_samples(4))

    samples = buffer.get_all_samples()
    assert len(samples) == 5
    assert samples[0].move_index == 2


def test_policy_and_value_loss_are_finite() -> None:
    """Soft cross entropy and MSE loss helpers should return finite scalars."""

    policy_logits = torch.zeros((2, BOARD_SIZE * BOARD_SIZE), dtype=torch.float32)
    policy_target = torch.zeros((2, BOARD_SIZE * BOARD_SIZE), dtype=torch.float32)
    policy_target[0, 3] = 1.0
    policy_target[1, 7] = 1.0
    value_pred = torch.zeros((2, 1), dtype=torch.float32)
    value_target = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)

    policy_loss = compute_policy_loss(policy_logits, policy_target)
    value_loss = compute_value_loss(value_pred, value_target)

    assert torch.isfinite(policy_loss)
    assert torch.isfinite(value_loss)


def test_collect_self_play_games_returns_records() -> None:
    """Trainer should collect self-play records in eval mode."""

    checkpoint_dir = str(Path.cwd() / f"trainer_test_ckpt_{uuid4().hex}")
    trainer = _make_test_trainer(checkpoint_dir)
    trainer.model.train()

    try:
        records = trainer.collect_self_play_games(num_games=2)
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    assert len(records) == 2
    assert all(isinstance(record, GameRecord) for record in records)
    assert trainer.model.training is True


def test_collect_self_play_games_preserves_existing_model_mode() -> None:
    """Self-play collection should preserve both eval and train entry modes."""

    checkpoint_dir = str(Path.cwd() / f"trainer_test_ckpt_{uuid4().hex}")
    trainer = _make_test_trainer(checkpoint_dir)

    try:
        trainer.model.eval()
        trainer.collect_self_play_games(num_games=1)
        assert trainer.model.training is False

        trainer.model.train()
        trainer.collect_self_play_games(num_games=1)
        assert trainer.model.training is True
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


def test_train_on_buffer_raises_on_empty_buffer() -> None:
    """Training should fail fast when the recent-sample buffer is empty."""

    checkpoint_dir = str(Path.cwd() / f"trainer_test_ckpt_{uuid4().hex}")
    trainer = _make_test_trainer(checkpoint_dir)

    try:
        try:
            trainer.train_on_buffer()
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for training on an empty buffer.")
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


def test_train_on_buffer_runs_and_returns_metrics() -> None:
    """Training on buffered samples should return finite average metrics and update parameters."""

    checkpoint_dir = str(Path.cwd() / f"trainer_test_ckpt_{uuid4().hex}")
    trainer = _make_test_trainer(checkpoint_dir)
    records = trainer.collect_self_play_games(num_games=1)
    trainer.buffer.add_games(records)

    before = torch.cat([parameter.detach().reshape(-1).clone() for parameter in trainer.model.parameters()])
    try:
        metrics = trainer.train_on_buffer()
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    after = torch.cat([parameter.detach().reshape(-1).clone() for parameter in trainer.model.parameters()])

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "total_loss" in metrics
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])
    assert np.isfinite(metrics["total_loss"])
    assert not torch.allclose(before, after)


def test_run_training_cycle_saves_checkpoint() -> None:
    """A full training cycle should save a trainer checkpoint."""

    checkpoint_dir = Path.cwd() / f"trainer_test_ckpt_{uuid4().hex}"
    trainer = _make_test_trainer(str(checkpoint_dir))

    try:
        metrics = trainer.run_training_cycle(cycle_index=2)
        checkpoint_path = Path(metrics["checkpoint_path"])
        payload = torch.load(checkpoint_path, map_location="cpu")
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    assert checkpoint_path.exists() or "model_state_dict" in payload
    assert payload["format_version"] == 1
    assert payload["checkpoint_type"] == "alphazero_trainer"
    assert "model_state_dict" in payload
    assert "optimizer_state_dict" in payload
    assert "trainer_config" in payload
    assert payload["cycle_index"] == 2
    assert trainer.buffer.get_all_samples()


def test_optimizer_is_adamw_and_value_targets_have_shape_one() -> None:
    """Trainer must use AdamW and dataset must expose value targets with shape `(1,)`."""

    checkpoint_dir = str(Path.cwd() / f"trainer_test_ckpt_{uuid4().hex}")
    trainer = _make_test_trainer(checkpoint_dir)
    dataset = AlphaZeroSampleDataset([_make_sample(0, BLACK, 5, 1.0)])
    _, _, value_target = dataset[0]

    try:
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert value_target.shape == (1,)
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
