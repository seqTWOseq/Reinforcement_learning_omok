"""Helper utilities for deterministic AlphaZero evaluation games.

Evaluation is intentionally different from self-play:
- no root Dirichlet noise
- temperature-driven exploration is disabled by default
- no policy targets or training samples are generated
- only win/draw/loss outcomes are aggregated for model comparison
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from gomoku_ai.alphazero.mcts import MCTS, MCTSConfig
from gomoku_ai.alphazero.model import PolicyValueNet
from gomoku_ai.env import BLACK, DRAW, GomokuEnv, WHITE

if TYPE_CHECKING:
    from gomoku_ai.alphazero.evaluation import EvaluationConfig


def build_evaluation_mcts_config(config: "EvaluationConfig") -> MCTSConfig:
    """Build the deterministic MCTS config used during evaluation.

    Evaluation must not reuse self-play exploration settings. In particular:
    - `add_root_noise` follows `config.use_root_noise` and defaults to `False`
    - `temperature` follows `config.eval_temperature` and defaults to `0.0`
    - `num_simulations` follows `config.eval_num_simulations`
    """

    default_mcts_config = MCTSConfig()
    return MCTSConfig(
        num_simulations=config.eval_num_simulations,
        c_puct=default_mcts_config.c_puct,
        dirichlet_alpha=default_mcts_config.dirichlet_alpha,
        dirichlet_epsilon=default_mcts_config.dirichlet_epsilon,
        add_root_noise=config.use_root_noise,
        temperature=config.eval_temperature,
    )


def compute_candidate_score(
    candidate_wins: int,
    reference_wins: int,
    draws: int,
    *,
    num_games: int,
    draw_score: float,
) -> tuple[float, float]:
    """Compute candidate score and pure win rate from aggregated results."""

    if num_games <= 0:
        raise ValueError("num_games must be positive.")
    if candidate_wins < 0 or reference_wins < 0 or draws < 0:
        raise ValueError("Win/loss/draw counts must be non-negative.")
    if candidate_wins + reference_wins + draws != num_games:
        raise ValueError("candidate_wins + reference_wins + draws must equal num_games.")
    if not 0.0 <= draw_score <= 1.0:
        raise ValueError("draw_score must be in [0.0, 1.0].")

    candidate_score = (candidate_wins + draw_score * draws) / num_games
    candidate_win_rate = candidate_wins / num_games
    return candidate_score, candidate_win_rate


def play_single_evaluation_game(
    candidate_model: PolicyValueNet,
    reference_model: PolicyValueNet,
    *,
    candidate_is_black: bool,
    mcts_config: MCTSConfig,
    env_factory: Callable[[], GomokuEnv] | None = None,
) -> int:
    """Play one deterministic evaluation game and return the final winner.

    Evaluation uses deterministic move selection:
    - no self-play data is recorded
    - no root noise should be applied
    - temperature should be zero in the passed `mcts_config`
    - candidate and reference colors are assigned externally
    """

    if not isinstance(candidate_model, PolicyValueNet):
        raise TypeError("candidate_model must be a PolicyValueNet instance.")
    if not isinstance(reference_model, PolicyValueNet):
        raise TypeError("reference_model must be a PolicyValueNet instance.")
    if not isinstance(mcts_config, MCTSConfig):
        raise TypeError("mcts_config must be an MCTSConfig instance.")

    factory = GomokuEnv if env_factory is None else env_factory
    env = factory()
    if not isinstance(env, GomokuEnv):
        raise TypeError("env_factory must create GomokuEnv instances.")
    env.reset()

    black_model = candidate_model if candidate_is_black else reference_model
    white_model = reference_model if candidate_is_black else candidate_model
    mcts = MCTS(mcts_config)

    while not env.done:
        active_model = black_model if env.current_player == BLACK else white_model
        root = mcts.run(env, active_model)
        action = mcts.select_action(root, temperature=mcts_config.temperature)
        env.apply_move(action)

    if env.winner is None:
        raise RuntimeError("Evaluation game ended without a resolved winner.")
    if env.winner not in {BLACK, WHITE, DRAW}:
        raise ValueError(f"Unexpected winner value: {env.winner!r}.")
    return env.winner
