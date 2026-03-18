"""Candidate move generation and ordering for Athenan search/agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gomoku_ai.env import BLACK, EMPTY, GomokuEnv, WHITE

if TYPE_CHECKING:
    from gomoku_ai.negamax_athenan.eval.evaluator import GreedyHeuristicEvaluator


def generate_candidate_actions(
    env: GomokuEnv,
    *,
    radius: int = 2,
    max_candidates: int | None = None,
    evaluator: GreedyHeuristicEvaluator | None = None,
    player: int | None = None,
) -> list[int]:
    """Return ordered candidate actions near existing stones.

    Policy:
    - terminal positions have no candidates
    - empty boards return the unique center move
    - otherwise only legal empty cells within Chebyshev `radius` of any stone
    - candidates are ordered tactically/heuristically for reuse by greedy play
      and future search
    """

    if radius < 0:
        raise ValueError("radius must be non-negative.")
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive when provided.")

    legal_actions = env.get_legal_actions()
    if not legal_actions:
        return []

    if _is_board_empty(env):
        center_action = env.coord_to_action(env.board_size // 2, env.board_size // 2)
        ordered_center = [center_action] if env.is_legal_action(center_action) else [legal_actions[0]]
        return ordered_center[:max_candidates] if max_candidates is not None else ordered_center

    candidate_set = _collect_proximity_candidates(env, radius=radius)
    if not candidate_set:
        return order_candidate_actions(
            env,
            legal_actions,
            max_candidates=max_candidates,
            evaluator=evaluator,
            player=player,
        )

    return order_candidate_actions(
        env,
        sorted(candidate_set),
        max_candidates=max_candidates,
        evaluator=evaluator,
        player=player,
    )


def order_candidate_actions(
    env: GomokuEnv,
    actions: list[int] | tuple[int, ...],
    *,
    max_candidates: int | None = None,
    evaluator: GreedyHeuristicEvaluator | None = None,
    player: int | None = None,
) -> list[int]:
    """Order candidate actions by tactical urgency, heuristic score, and tie-breaks."""

    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive when provided.")
    if env.done:
        return []

    player_to_move = int(env.current_player if player is None else player)
    if player_to_move not in {BLACK, WHITE}:
        raise ValueError(f"player must be BLACK({BLACK}) or WHITE({WHITE}).")
    if player is not None and player_to_move != int(env.current_player):
        raise ValueError("order_candidate_actions requires player to equal env.current_player.")

    resolved_actions = _resolve_unique_legal_actions(env, actions)
    if not resolved_actions:
        return []

    resolved_evaluator = _resolve_evaluator(evaluator)
    opponent = WHITE if player_to_move == BLACK else BLACK
    scored_actions = []
    for action in resolved_actions:
        own_win = resolved_evaluator.would_action_win_for_player(env, action, player_to_move)
        block = resolved_evaluator.would_action_win_for_player(env, action, opponent)
        heuristic_score = float(resolved_evaluator.score_action_for_player(env, action, player_to_move))
        scored_actions.append(
            (
                int(action),
                bool(own_win),
                bool(block),
                heuristic_score,
                _center_distance(env, int(action)),
            )
        )

    scored_actions.sort(
        key=lambda item: (
            0 if item[1] else 1,
            0 if item[2] else 1,
            -item[3],
            item[4],
            item[0],
        )
    )
    ordered = [action for action, _, _, _, _ in scored_actions]
    if max_candidates is not None:
        return ordered[:max_candidates]
    return ordered


def score_candidate_action(
    env: GomokuEnv,
    action: int,
    *,
    evaluator: GreedyHeuristicEvaluator | None = None,
    player: int | None = None,
) -> float:
    """Return a numeric ordering score for one legal action."""

    if env.done:
        raise ValueError("Cannot score actions on a finished game.")

    player_to_move = int(env.current_player if player is None else player)
    if player_to_move not in {BLACK, WHITE}:
        raise ValueError(f"player must be BLACK({BLACK}) or WHITE({WHITE}).")
    if player is not None and player_to_move != int(env.current_player):
        raise ValueError("score_candidate_action requires player to equal env.current_player.")
    if not env.is_legal_action(action):
        raise ValueError(f"Action {action} is not legal in the current position.")

    resolved_evaluator = _resolve_evaluator(evaluator)
    opponent = WHITE if player_to_move == BLACK else BLACK
    own_win_bonus = (
        float(resolved_evaluator.config.terminal_win_score)
        if resolved_evaluator.would_action_win_for_player(env, action, player_to_move)
        else 0.0
    )
    block_bonus = (
        float(resolved_evaluator.config.terminal_win_score * 0.5)
        if resolved_evaluator.would_action_win_for_player(env, action, opponent)
        else 0.0
    )
    heuristic_score = float(resolved_evaluator.score_action_for_player(env, action, player_to_move))
    return float(own_win_bonus + block_bonus + heuristic_score)


def _resolve_unique_legal_actions(
    env: GomokuEnv,
    actions: list[int] | tuple[int, ...],
) -> list[int]:
    legal_mask = env.get_valid_moves()
    resolved: list[int] = []
    seen: set[int] = set()
    for action in actions:
        normalized = int(action)
        if normalized in seen:
            continue
        if 0 <= normalized < legal_mask.size and bool(legal_mask[normalized]):
            resolved.append(normalized)
            seen.add(normalized)
    return resolved


def _collect_proximity_candidates(env: GomokuEnv, *, radius: int) -> set[int]:
    legal_mask = env.get_valid_moves()
    occupied_coords = np.argwhere(env.board != EMPTY)
    candidate_set: set[int] = set()
    for row, col in occupied_coords:
        for delta_row in range(-radius, radius + 1):
            for delta_col in range(-radius, radius + 1):
                next_row = int(row + delta_row)
                next_col = int(col + delta_col)
                if not (0 <= next_row < env.board_size and 0 <= next_col < env.board_size):
                    continue
                action = env.coord_to_action(next_row, next_col)
                if bool(legal_mask[action]):
                    candidate_set.add(action)
    return candidate_set


def _is_board_empty(env: GomokuEnv) -> bool:
    return not bool(np.any(env.board != EMPTY))


def _center_distance(env: GomokuEnv, action: int) -> float:
    row, col = env.action_to_coord(action)
    center = (env.board_size - 1) / 2.0
    return float(abs(row - center) + abs(col - center))


def _resolve_evaluator(evaluator: GreedyHeuristicEvaluator | None) -> GreedyHeuristicEvaluator:
    if evaluator is not None:
        return evaluator

    from gomoku_ai.negamax_athenan.eval.evaluator import GreedyHeuristicEvaluator

    return GreedyHeuristicEvaluator()
