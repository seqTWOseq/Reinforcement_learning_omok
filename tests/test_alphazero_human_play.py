"""Tests for human-vs-AlphaZero play and human-play data recording."""

from __future__ import annotations

import numpy as np

from gomoku_ai.alphazero import (
    GameRecord,
    HumanPlayConfig,
    HumanVsAlphaZeroGameRunner,
    PolicyValueNet,
    PolicyValueNetConfig,
    extract_human_play_samples,
    maybe_extract_ai_turn_samples,
)
from gomoku_ai.env import BLACK, BOARD_SIZE, GomokuEnv, WHITE


class StrongScriptedPolicyValueNet(PolicyValueNet):
    """Deterministic AI that builds a fast five without blocking human test moves."""

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
        row = 7 if black_to_move else 8

        logits = np.full((BOARD_SIZE * BOARD_SIZE,), -100.0, dtype=np.float32)
        for col in range(5):
            if not occupied[row, col]:
                logits[row * BOARD_SIZE + col] = 100.0
                break
        else:
            empty_indices = np.flatnonzero(~occupied.reshape(-1))
            logits[int(empty_indices[0])] = 100.0
        return logits, 0.5


class SpyHumanVsAlphaZeroGameRunner(HumanVsAlphaZeroGameRunner):
    """Runner subclass that records AI MCTS settings used during play."""

    def __init__(self, config: HumanPlayConfig) -> None:
        super().__init__(config=config)
        self.seen_mcts_configs: list[tuple[int, bool, float]] = []

    def _select_ai_action(
        self,
        env: GomokuEnv,
        model: PolicyValueNet,
        mcts_config: object,
    ) -> tuple[np.ndarray, int]:
        self.seen_mcts_configs.append(
            (mcts_config.num_simulations, mcts_config.add_root_noise, mcts_config.temperature)
        )
        return super()._select_ai_action(env, model, mcts_config)


def _make_runner() -> HumanVsAlphaZeroGameRunner:
    """Construct a deterministic human-play runner for tests."""

    return HumanVsAlphaZeroGameRunner(
        config=HumanPlayConfig(
            human_color="select_each_game",
            ai_temperature=0.0,
            use_root_noise=False,
            ai_num_simulations=2,
            record_ai_turn_only=True,
            game_id_prefix="humanplay",
        )
    )


def _play_scripted_game(human_color: int) -> GameRecord:
    """Play one deterministic scripted human-vs-AI game."""

    if human_color == BLACK:
        human_moves = [0, 17, 34, 51, 68]
    else:
        human_moves = [10, 27, 44, 61]

    runner = _make_runner()
    return runner.play_game(
        StrongScriptedPolicyValueNet(),
        human_color=human_color,
        human_moves=human_moves,
    )


def test_play_game_returns_human_play_record() -> None:
    """Human-vs-AI play should return a populated `human_play` record."""

    record = _play_scripted_game(BLACK)

    assert isinstance(record, GameRecord)
    assert record.game_id.startswith("humanplay-")


def test_source_is_human_play() -> None:
    """Recorded human-vs-AI games must use `source='human_play'`."""

    record = _play_scripted_game(BLACK)

    assert record.source == "human_play"


def test_moves_store_all_actions() -> None:
    """The move list should contain every human and AI action in order."""

    record = _play_scripted_game(BLACK)

    assert len(record.moves) == 10
    assert record.moves[0] == 0
    assert record.moves[1] == 8 * BOARD_SIZE + 0


def test_samples_store_ai_turn_only_when_enabled() -> None:
    """Stage-8 baseline should store only AI turns as training samples."""

    record = _play_scripted_game(BLACK)

    assert len(record.samples) == 5
    assert all(sample.player_to_move == WHITE for sample in record.samples)
    assert len(maybe_extract_ai_turn_samples(record)) == len(record.samples)
    assert len(extract_human_play_samples([record])) == len(record.samples)


def test_state_is_captured_before_ai_move() -> None:
    """Each stored sample state should match the board immediately before the AI move."""

    record = _play_scripted_game(BLACK)
    env = GomokuEnv()
    env.reset()
    samples_by_move_index = {sample.move_index: sample for sample in record.samples}

    for move_index, action in enumerate(record.moves):
        if move_index in samples_by_move_index:
            sample = samples_by_move_index[move_index]
            assert np.array_equal(sample.state, env.encode_state())
            assert sample.action_taken == action
        env.apply_move(action)


def test_human_and_ai_colors_are_recorded_in_metadata() -> None:
    """Metadata should store resolved human and AI colors."""

    black_record = _play_scripted_game(BLACK)
    white_record = _play_scripted_game(WHITE)

    assert black_record.metadata["human_color"] == BLACK
    assert black_record.metadata["ai_color"] == WHITE
    assert white_record.metadata["human_color"] == WHITE
    assert white_record.metadata["ai_color"] == BLACK


def test_ai_uses_deterministic_eval_style_settings() -> None:
    """Human-play AI should use evaluation-style settings, not self-play settings."""

    runner = SpyHumanVsAlphaZeroGameRunner(
        HumanPlayConfig(
            human_color="select_each_game",
            ai_temperature=0.0,
            use_root_noise=False,
            ai_num_simulations=3,
            record_ai_turn_only=True,
            game_id_prefix="humanplay",
        )
    )
    runner.play_game(
        StrongScriptedPolicyValueNet(),
        human_color=BLACK,
        human_moves=[0, 17, 34, 51, 68],
    )

    assert runner.seen_mcts_configs
    for num_simulations, add_root_noise, temperature in runner.seen_mcts_configs:
        assert num_simulations == 3
        assert add_root_noise is False
        assert temperature == 0.0


def test_scripted_human_game_supports_both_black_and_white() -> None:
    """Scripted human play should work for both color assignments."""

    black_record = _play_scripted_game(BLACK)
    white_record = _play_scripted_game(WHITE)

    assert black_record.winner == WHITE
    assert white_record.winner == BLACK
    assert len(black_record.samples) == 5
    assert len(white_record.samples) == 5
