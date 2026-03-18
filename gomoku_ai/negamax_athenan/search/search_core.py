"""Negamax search core for the negamax Athenan stack."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from gomoku_ai.negamax_athenan.search.move_generator import generate_candidate_actions
from gomoku_ai.env import BLACK, DRAW, GomokuEnv, WHITE

if TYPE_CHECKING:
    from gomoku_ai.negamax_athenan.eval.evaluator import GreedyHeuristicEvaluator


TTFlag = Literal["EXACT", "LOWER", "UPPER"]


@dataclass(frozen=True)
class NegamaxSearchSummary:
    """Pure negamax search output for one root position."""

    value: float
    best_action: int | None
    action_values: dict[int, float]
    principal_variation: list[int]
    nodes: int
    depth_reached: int
    cutoffs: int
    pruned_branches: int
    tt_hits: int
    tt_stores: int


@dataclass(frozen=True)
class _NegamaxNodeResult:
    """Internal pure-negamax node payload."""

    value: float
    pv: list[int]
    nodes: int
    depth_reached: int
    cutoffs: int
    pruned_branches: int
    tt_hits: int
    tt_stores: int


@dataclass(frozen=True)
class TTEntry:
    """Transposition-table entry for one searched position."""

    key: tuple[bytes, int]
    value: float
    depth: int
    flag: TTFlag
    best_action: int | None = None
    pv: tuple[int, ...] = ()


def negamax(
    env: GomokuEnv,
    depth: int,
    evaluator: GreedyHeuristicEvaluator | None = None,
    *,
    radius: int = 2,
    max_candidates: int | None = None,
    use_alpha_beta: bool = True,
    use_transposition_table: bool = True,
    preferred_action: int | None = None,
    transposition_table: dict[tuple[bytes, int], TTEntry] | None = None,
) -> tuple[float, int | None]:
    """Return `(best_value, best_action)` from a depth-limited negamax search."""

    result = run_negamax_search(
        env,
        depth=depth,
        evaluator=evaluator,
        radius=radius,
        max_candidates=max_candidates,
        use_alpha_beta=use_alpha_beta,
        use_transposition_table=use_transposition_table,
        preferred_action=preferred_action,
        transposition_table=transposition_table,
    )
    return float(result.value), result.best_action


def run_negamax_search(
    env: GomokuEnv,
    *,
    depth: int,
    evaluator: GreedyHeuristicEvaluator | None = None,
    radius: int = 2,
    max_candidates: int | None = None,
    use_alpha_beta: bool = True,
    use_transposition_table: bool = True,
    preferred_action: int | None = None,
    transposition_table: dict[tuple[bytes, int], TTEntry] | None = None,
) -> NegamaxSearchSummary:
    """Run a negamax search using the heuristic evaluator at leaf nodes."""

    if not isinstance(env, GomokuEnv):
        raise TypeError("env must be a GomokuEnv instance.")
    if depth < 0:
        raise ValueError("depth must be non-negative.")
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive when provided.")
    if not isinstance(use_alpha_beta, bool):
        raise ValueError("use_alpha_beta must be a bool.")
    if not isinstance(use_transposition_table, bool):
        raise ValueError("use_transposition_table must be a bool.")

    resolved_evaluator = _resolve_greedy_evaluator(evaluator)
    root_player = int(env.current_player)
    resolved_tt = None if not use_transposition_table else ({} if transposition_table is None else transposition_table)

    terminal_value = _evaluate_terminal_for_player(
        env,
        player_to_move=root_player,
        depth_remaining=depth,
        evaluator=resolved_evaluator,
    )
    if terminal_value is not None:
        return NegamaxSearchSummary(
            value=float(terminal_value),
            best_action=None,
            action_values={},
            principal_variation=[],
            nodes=1,
            depth_reached=0,
            cutoffs=0,
            pruned_branches=0,
            tt_hits=0,
            tt_stores=0,
        )

    if depth == 0:
        return NegamaxSearchSummary(
            value=float(_evaluate_leaf_for_player(env, player=root_player, evaluator=resolved_evaluator)),
            best_action=None,
            action_values={},
            principal_variation=[],
            nodes=1,
            depth_reached=0,
            cutoffs=0,
            pruned_branches=0,
            tt_hits=0,
            tt_stores=0,
        )

    actions = generate_candidate_actions(
        env,
        radius=radius,
        max_candidates=max_candidates,
        evaluator=resolved_evaluator,
    )
    actions = _prioritize_preferred_action(actions, _tt_best_action_for_key(resolved_tt, _make_transposition_key(env)))
    actions = _prioritize_preferred_action(actions, preferred_action)
    if not actions:
        return NegamaxSearchSummary(
            value=float(_evaluate_leaf_for_player(env, player=root_player, evaluator=resolved_evaluator)),
            best_action=None,
            action_values={},
            principal_variation=[],
            nodes=1,
            depth_reached=0,
            cutoffs=0,
            pruned_branches=0,
            tt_hits=0,
            tt_stores=0,
        )

    best_action = actions[0]
    best_value = -math.inf
    best_child_pv: list[int] = []
    action_values: dict[int, float] = {}
    nodes = 1
    depth_reached = 0
    cutoffs = 0
    pruned_branches = 0
    tt_hits = 0
    tt_stores = 0
    alpha = -math.inf
    beta = math.inf
    local_alpha = alpha

    for index, action in enumerate(actions):
        child_env = env.clone()
        child_env.apply_move(action)
        child_result = _negamax_node(
            child_env,
            depth_remaining=depth - 1,
            player_to_move=_opponent_of(root_player),
            evaluator=resolved_evaluator,
            radius=radius,
            max_candidates=max_candidates,
            ply=1,
            alpha=-beta,
            beta=-local_alpha,
            use_alpha_beta=use_alpha_beta,
            use_transposition_table=use_transposition_table,
            transposition_table=resolved_tt,
        )
        value = -child_result.value
        action_values[int(action)] = float(value)
        nodes += child_result.nodes
        depth_reached = max(depth_reached, child_result.depth_reached)
        cutoffs += child_result.cutoffs
        pruned_branches += child_result.pruned_branches
        tt_hits += child_result.tt_hits
        tt_stores += child_result.tt_stores

        if value > best_value:
            best_value = float(value)
            best_action = int(action)
            best_child_pv = child_result.pv

        if use_alpha_beta:
            local_alpha = max(local_alpha, value)
            if local_alpha >= beta:
                cutoffs += 1
                pruned_branches += max(0, len(actions) - index - 1)
                break

    principal_variation = [best_action] + best_child_pv
    tt_stores += _store_tt_entry(
        resolved_tt,
        _make_transposition_key(env),
        TTEntry(
            key=_make_transposition_key(env),
            value=float(best_value),
            depth=depth,
            flag="EXACT",
            best_action=int(best_action),
            pv=tuple(principal_variation),
        ),
    )
    return NegamaxSearchSummary(
        value=float(best_value),
        best_action=int(best_action),
        action_values=action_values,
        principal_variation=principal_variation,
        nodes=nodes,
        depth_reached=max(1, depth_reached),
        cutoffs=cutoffs,
        pruned_branches=pruned_branches,
        tt_hits=tt_hits,
        tt_stores=tt_stores,
    )


def run_iterative_deepening_search(
    env: GomokuEnv,
    *,
    max_depth: int,
    evaluator: GreedyHeuristicEvaluator | None = None,
    radius: int = 2,
    max_candidates: int | None = None,
    use_alpha_beta: bool = True,
    use_transposition_table: bool = True,
    transposition_table: dict[tuple[bytes, int], TTEntry] | None = None,
) -> NegamaxSearchSummary:
    """Run repeated depth-1..N root searches and return the deepest completed result."""

    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")

    if max_depth == 0:
        return run_negamax_search(
            env,
            depth=0,
            evaluator=evaluator,
            radius=radius,
            max_candidates=max_candidates,
            use_alpha_beta=use_alpha_beta,
            use_transposition_table=use_transposition_table,
            transposition_table=transposition_table,
        )

    best_completed_result: NegamaxSearchSummary | None = None
    preferred_action: int | None = None
    shared_tt = None if not use_transposition_table else ({} if transposition_table is None else transposition_table)
    for depth in range(1, max_depth + 1):
        result = run_negamax_search(
            env,
            depth=depth,
            evaluator=evaluator,
            radius=radius,
            max_candidates=max_candidates,
            use_alpha_beta=use_alpha_beta,
            use_transposition_table=use_transposition_table,
            preferred_action=preferred_action,
            transposition_table=shared_tt,
        )
        best_completed_result = result
        preferred_action = result.best_action

    if best_completed_result is None:
        raise RuntimeError("Iterative deepening did not complete any search depth.")
    return best_completed_result


def _negamax_node(
    env: GomokuEnv,
    *,
    depth_remaining: int,
    player_to_move: int,
    evaluator: GreedyHeuristicEvaluator,
    radius: int,
    max_candidates: int | None,
    ply: int,
    alpha: float,
    beta: float,
    use_alpha_beta: bool,
    use_transposition_table: bool,
    transposition_table: dict[tuple[bytes, int], TTEntry] | None,
) -> _NegamaxNodeResult:
    """Recursive negamax node evaluation with optional alpha-beta pruning."""

    original_alpha = alpha
    original_beta = beta
    key = _make_transposition_key(env)
    tt_entry = None if transposition_table is None else transposition_table.get(key)
    tt_hits = 1 if tt_entry is not None else 0
    tt_stores = 0

    terminal_value = _evaluate_terminal_for_player(
        env,
        player_to_move=player_to_move,
        depth_remaining=depth_remaining,
        evaluator=evaluator,
    )
    if terminal_value is not None:
        tt_stores += _store_tt_entry(
            transposition_table if use_transposition_table else None,
            key,
            TTEntry(key=key, value=float(terminal_value), depth=depth_remaining, flag="EXACT"),
        )
        return _NegamaxNodeResult(
            value=float(terminal_value),
            pv=[],
            nodes=1,
            depth_reached=ply,
            cutoffs=0,
            pruned_branches=0,
            tt_hits=tt_hits,
            tt_stores=tt_stores,
        )

    if int(env.current_player) != int(player_to_move):
        raise ValueError("player_to_move must match env.current_player on non-terminal nodes.")

    if use_transposition_table and tt_entry is not None and tt_entry.depth >= depth_remaining:
        if tt_entry.flag == "EXACT":
            return _NegamaxNodeResult(
                value=float(tt_entry.value),
                pv=_pv_from_tt_entry(tt_entry),
                nodes=1,
                depth_reached=ply,
                cutoffs=0,
                pruned_branches=0,
                tt_hits=tt_hits,
                tt_stores=tt_stores,
            )
        if tt_entry.flag == "LOWER":
            alpha = max(alpha, float(tt_entry.value))
        else:
            beta = min(beta, float(tt_entry.value))
        if alpha >= beta:
            return _NegamaxNodeResult(
                value=float(tt_entry.value),
                pv=_pv_from_tt_entry(tt_entry),
                nodes=1,
                depth_reached=ply,
                cutoffs=1,
                pruned_branches=0,
                tt_hits=tt_hits,
                tt_stores=tt_stores,
            )

    if depth_remaining <= 0:
        leaf_value = float(_evaluate_leaf_for_player(env, player=player_to_move, evaluator=evaluator))
        tt_stores += _store_tt_entry(
            transposition_table if use_transposition_table else None,
            key,
            TTEntry(key=key, value=leaf_value, depth=depth_remaining, flag="EXACT"),
        )
        return _NegamaxNodeResult(
            value=leaf_value,
            pv=[],
            nodes=1,
            depth_reached=ply,
            cutoffs=0,
            pruned_branches=0,
            tt_hits=tt_hits,
            tt_stores=tt_stores,
        )

    actions = generate_candidate_actions(
        env,
        radius=radius,
        max_candidates=max_candidates,
        evaluator=evaluator,
    )
    actions = _prioritize_preferred_action(actions, None if tt_entry is None else tt_entry.best_action)
    if not actions:
        leaf_value = float(_evaluate_leaf_for_player(env, player=player_to_move, evaluator=evaluator))
        tt_stores += _store_tt_entry(
            transposition_table if use_transposition_table else None,
            key,
            TTEntry(key=key, value=leaf_value, depth=depth_remaining, flag="EXACT"),
        )
        return _NegamaxNodeResult(
            value=leaf_value,
            pv=[],
            nodes=1,
            depth_reached=ply,
            cutoffs=0,
            pruned_branches=0,
            tt_hits=tt_hits,
            tt_stores=tt_stores,
        )

    best_value = -math.inf
    best_pv: list[int] = []
    nodes = 1
    depth_reached = ply
    cutoffs = 0
    pruned_branches = 0
    local_alpha = alpha

    for index, action in enumerate(actions):
        child_env = env.clone()
        child_env.apply_move(action)
        child_result = _negamax_node(
            child_env,
            depth_remaining=depth_remaining - 1,
            player_to_move=_opponent_of(player_to_move),
            evaluator=evaluator,
            radius=radius,
            max_candidates=max_candidates,
            ply=ply + 1,
            alpha=-beta,
            beta=-local_alpha,
            use_alpha_beta=use_alpha_beta,
            use_transposition_table=use_transposition_table,
            transposition_table=transposition_table,
        )
        value = -child_result.value
        nodes += child_result.nodes
        depth_reached = max(depth_reached, child_result.depth_reached)
        cutoffs += child_result.cutoffs
        pruned_branches += child_result.pruned_branches
        tt_hits += child_result.tt_hits
        tt_stores += child_result.tt_stores

        if value > best_value:
            best_value = float(value)
            best_pv = [int(action)] + child_result.pv

        if use_alpha_beta:
            local_alpha = max(local_alpha, value)
            if local_alpha >= beta:
                cutoffs += 1
                pruned_branches += max(0, len(actions) - index - 1)
                break

    best_action = None if not best_pv else int(best_pv[0])
    tt_flag = _resolve_tt_flag(best_value, original_alpha, original_beta)
    tt_stores += _store_tt_entry(
        transposition_table if use_transposition_table else None,
        key,
        TTEntry(
            key=key,
            value=float(best_value),
            depth=depth_remaining,
            flag=tt_flag,
            best_action=best_action,
            pv=tuple(best_pv),
        ),
    )

    return _NegamaxNodeResult(
        value=float(best_value),
        pv=best_pv,
        nodes=nodes,
        depth_reached=depth_reached,
        cutoffs=cutoffs,
        pruned_branches=pruned_branches,
        tt_hits=tt_hits,
        tt_stores=tt_stores,
    )


def _evaluate_leaf_for_player(
    env: GomokuEnv,
    *,
    player: int,
    evaluator: GreedyHeuristicEvaluator,
) -> float:
    return float(evaluator.evaluate_for_player(env, int(player)))


def _evaluate_terminal_for_player(
    env: GomokuEnv,
    *,
    player_to_move: int,
    depth_remaining: int,
    evaluator: GreedyHeuristicEvaluator,
) -> float | None:
    """Return terminal score from `player_to_move` perspective, otherwise `None`."""

    if not env.done:
        return None
    if env.winner == DRAW:
        return 0.0
    if env.winner not in {BLACK, WHITE}:
        raise ValueError(f"Unexpected terminal winner value: {env.winner!r}.")

    base = float(evaluator.config.terminal_win_score + max(0, depth_remaining))
    return base if int(env.winner) == int(player_to_move) else -base


def _resolve_greedy_evaluator(
    evaluator: GreedyHeuristicEvaluator | None,
) -> GreedyHeuristicEvaluator:
    if evaluator is not None:
        return evaluator

    from gomoku_ai.negamax_athenan.eval.evaluator import GreedyHeuristicEvaluator

    return GreedyHeuristicEvaluator()


def _opponent_of(player: int) -> int:
    return WHITE if int(player) == BLACK else BLACK


def _prioritize_preferred_action(
    actions: list[int],
    preferred_action: int | None,
) -> list[int]:
    if preferred_action is None or preferred_action not in actions:
        return actions
    if actions and actions[0] == preferred_action:
        return actions
    return [int(preferred_action)] + [action for action in actions if int(action) != int(preferred_action)]


def _make_transposition_key(env: GomokuEnv) -> tuple[bytes, int]:
    """Build a safe TT key from board bytes and side to move."""

    return env.board.tobytes(), int(env.current_player)


def _resolve_tt_flag(value: float, alpha: float, beta: float) -> TTFlag:
    if value <= alpha:
        return "UPPER"
    if value >= beta:
        return "LOWER"
    return "EXACT"


def _pv_from_tt_entry(entry: TTEntry) -> list[int]:
    if entry.pv:
        return [int(action) for action in entry.pv]
    if entry.best_action is None:
        return []
    return [int(entry.best_action)]


def _store_tt_entry(
    transposition_table: dict[tuple[bytes, int], TTEntry] | None,
    key: tuple[bytes, int],
    entry: TTEntry,
) -> int:
    if transposition_table is None:
        return 0
    existing = transposition_table.get(key)
    if existing is not None:
        if existing.depth > entry.depth:
            return 0
        if (
            existing.depth == entry.depth
            and existing.flag == "EXACT"
            and entry.flag != "EXACT"
        ):
            return 0
    transposition_table[key] = entry
    return 1


def _tt_best_action_for_key(
    transposition_table: dict[tuple[bytes, int], TTEntry] | None,
    key: tuple[bytes, int],
) -> int | None:
    if transposition_table is None:
        return None
    entry = transposition_table.get(key)
    if entry is None:
        return None
    return None if entry.best_action is None else int(entry.best_action)
