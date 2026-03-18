"""Move-ordering helpers for shallow Athenan search baselines."""

from __future__ import annotations

from gomoku_ai.negamax_athenan.search.move_generator import (
    order_candidate_actions,
    score_candidate_action,
)
from gomoku_ai.env import GomokuEnv


def score_action(
    env: GomokuEnv,
    action: int,
    *,
    player: int | None = None,
) -> float:
    """Return a heuristic ordering score for one legal action."""

    return score_candidate_action(env, action, player=player)


def order_actions(
    env: GomokuEnv,
    actions: list[int] | tuple[int, ...],
    *,
    player: int | None = None,
    candidate_limit: int | None = None,
) -> list[int]:
    """Sort candidate actions by descending ordering score."""

    return order_candidate_actions(
        env,
        actions,
        max_candidates=candidate_limit,
        player=player,
    )
