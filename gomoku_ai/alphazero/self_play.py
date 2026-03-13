"""Self-play game generation for AlphaZero-compatible Gomoku training data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from gomoku_ai.alphazero.mcts import MCTS, MCTSConfig
from gomoku_ai.alphazero.self_play_utils import (
    REQUIRED_SELF_PLAY_METADATA_KEYS,
    build_self_play_game_id,
    finalize_turn_records,
    get_temperature_for_move,
)
from gomoku_ai.alphazero.specs import ACTION_SIZE, GameRecord, POLICY_SHAPE, STATE_SHAPE
from gomoku_ai.alphazero.model import PolicyValueNet
from gomoku_ai.env import BLACK, DRAW, GomokuEnv, WHITE


@dataclass(frozen=True)
class SelfPlayConfig:
    """Configuration for generating a single AlphaZero self-play game.

    Fixed default behavior for this stage:
    - `opening_temperature_moves = 10`
    - `opening_temperature = 1.0`
    - `late_temperature = 0.0`
    - `use_root_noise = True`
    - `game_id_prefix = "selfplay"`
    """

    opening_temperature_moves: int = 10
    opening_temperature: float = 1.0
    late_temperature: float = 0.0
    use_root_noise: bool = True
    game_id_prefix: str = "selfplay"

    def __post_init__(self) -> None:
        """Validate self-play scheduling and naming settings."""

        if self.opening_temperature_moves < 0:
            raise ValueError("opening_temperature_moves must be non-negative.")
        if self.opening_temperature < 0.0:
            raise ValueError("opening_temperature must be non-negative.")
        if self.late_temperature < 0.0:
            raise ValueError("late_temperature must be non-negative.")
        if not isinstance(self.use_root_noise, bool):
            raise ValueError("use_root_noise must be a bool.")
        if not self.game_id_prefix or not self.game_id_prefix.strip():
            raise ValueError("game_id_prefix must be a non-empty string.")


@dataclass(frozen=True)
class SelfPlayTurnRecord:
    """Per-turn self-play snapshot captured before the selected action is applied.

    Contract:
    - `state` is always the encoded board state before the move.
    - `policy_target` is always the root visit-count distribution after applying
      the configured temperature schedule for that move.
    - `player_to_move` is the side to move for that pre-action state.
    """

    state: np.ndarray
    policy_target: np.ndarray
    player_to_move: int
    move_index: int
    action_taken: int

    def __post_init__(self) -> None:
        """Validate the turn record against the shared AlphaZero contracts."""

        state = np.asarray(self.state, dtype=np.float32)
        policy_target = np.asarray(self.policy_target, dtype=np.float32)

        if state.shape != STATE_SHAPE:
            raise ValueError(f"state must have shape {STATE_SHAPE}, got {state.shape}.")
        if state.dtype != np.float32:
            state = state.astype(np.float32, copy=False)
        if not np.isfinite(state).all():
            raise ValueError("state must contain only finite values.")

        if policy_target.shape != POLICY_SHAPE:
            raise ValueError(f"policy_target must have shape {POLICY_SHAPE}, got {policy_target.shape}.")
        if policy_target.dtype != np.float32:
            policy_target = policy_target.astype(np.float32, copy=False)
        if not np.isfinite(policy_target).all():
            raise ValueError("policy_target must contain only finite values.")
        if np.any(policy_target < 0.0):
            raise ValueError("policy_target must not contain negative probabilities.")
        if not np.isclose(float(policy_target.sum()), 1.0, atol=1e-5):
            raise ValueError(f"policy_target must sum to 1.0, got {float(policy_target.sum()):.6f}.")

        if self.player_to_move not in {BLACK, WHITE}:
            raise ValueError(f"player_to_move must be BLACK({BLACK}) or WHITE({WHITE}).")
        if self.move_index < 0:
            raise ValueError("move_index must be non-negative.")
        if not 0 <= self.action_taken < ACTION_SIZE:
            raise ValueError(f"action_taken must be in [0, {ACTION_SIZE - 1}].")

        object.__setattr__(self, "state", state)
        object.__setattr__(self, "policy_target", policy_target)


class SelfPlayGameGenerator:
    """Generate one full self-play game and return a `GameRecord`.

    The generated record follows the existing shared contracts:
    - `state` is captured before `env.apply_move(action)`
    - `policy_target` is built from root child visit counts after applying the
      configured per-move temperature
    - `value_target` is computed with `winner_to_value_target(winner, player_to_move)`
    """

    def __init__(
        self,
        config: SelfPlayConfig | None = None,
        mcts_config: MCTSConfig | None = None,
        env_factory: Callable[[], GomokuEnv] | None = None,
    ) -> None:
        """Initialize the self-play generator with reusable configs."""

        self.config = config or SelfPlayConfig()
        self.mcts_config = mcts_config or MCTSConfig()
        self.env_factory = env_factory or GomokuEnv

    def play_one_game(self, model: PolicyValueNet) -> GameRecord:
        """Play one self-play game and return a `GameRecord`.

        Each sample corresponds to exactly one move:
        - the sample state is the pre-action state
        - the sample policy target is the MCTS root visit-count distribution
          after applying the configured temperature schedule for that move
        - the sample value target is from that state's side-to-move perspective
        """

        if not isinstance(model, PolicyValueNet):
            raise TypeError("model must be a PolicyValueNet instance.")

        env = self.env_factory()
        if not isinstance(env, GomokuEnv):
            raise TypeError("env_factory must create GomokuEnv instances.")
        env.reset()

        game_id = build_self_play_game_id(self.config.game_id_prefix)
        turn_records: list[SelfPlayTurnRecord] = []
        moves: list[int] = []

        while not env.done:
            move_index = len(turn_records)
            state = np.asarray(env.encode_state(), dtype=np.float32).copy()
            player_to_move = env.current_player
            temperature = get_temperature_for_move(move_index=move_index, config=self.config)

            per_turn_mcts_config = MCTSConfig(
                num_simulations=self.mcts_config.num_simulations,
                c_puct=self.mcts_config.c_puct,
                dirichlet_alpha=self.mcts_config.dirichlet_alpha,
                dirichlet_epsilon=self.mcts_config.dirichlet_epsilon,
                add_root_noise=self.config.use_root_noise,
                temperature=temperature,
            )
            mcts = MCTS(per_turn_mcts_config)
            root = mcts.run(env, model)
            policy_target = mcts.get_action_probs(root, temperature=temperature)
            action = mcts.select_action(root, temperature=temperature)

            turn_records.append(
                SelfPlayTurnRecord(
                    state=state,
                    policy_target=policy_target,
                    player_to_move=player_to_move,
                    move_index=move_index,
                    action_taken=action,
                )
            )
            moves.append(action)
            env.apply_move(action)

        if env.winner is None:
            raise RuntimeError("Self-play game finished without a resolved winner value.")

        samples = finalize_turn_records(turn_records, env.winner, game_id)
        metadata = {
            "num_moves": len(moves),
            "use_root_noise": self.config.use_root_noise,
            "opening_temperature_moves": self.config.opening_temperature_moves,
            "opening_temperature": self.config.opening_temperature,
            "late_temperature": self.config.late_temperature,
            "num_simulations": self.mcts_config.num_simulations,
            "c_puct": self.mcts_config.c_puct,
        }
        missing_required_keys = [key for key in REQUIRED_SELF_PLAY_METADATA_KEYS if key not in metadata]
        if missing_required_keys:
            raise RuntimeError(f"Missing required self-play metadata keys: {missing_required_keys}")

        return GameRecord(
            game_id=game_id,
            moves=moves,
            winner=env.winner,
            source="self_play",
            samples=samples,
            metadata=metadata,
        )
