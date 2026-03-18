"""Evaluator-based searcher for the negamax Athenan stack."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gomoku_ai.negamax_athenan.search.search_core import (
    run_iterative_deepening_search,
    run_negamax_search,
)
from gomoku_ai.common.agents import BaseSearcher, SearchResult
from gomoku_ai.env import GomokuEnv

if TYPE_CHECKING:
    from gomoku_ai.negamax_athenan.eval.evaluator import GreedyHeuristicEvaluator


class AthenanNegamaxSearcher(BaseSearcher):
    """Evaluator-based negamax searcher with optional alpha-beta, ID, and TT."""

    def __init__(
        self,
        *,
        evaluator: GreedyHeuristicEvaluator | None = None,
        max_depth: int = 2,
        candidate_radius: int = 2,
        max_candidates: int | None = None,
        use_alpha_beta: bool = True,
        use_iterative_deepening: bool = False,
        use_transposition_table: bool = True,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        if candidate_radius < 0:
            raise ValueError("candidate_radius must be non-negative.")
        if max_candidates is not None and max_candidates <= 0:
            raise ValueError("max_candidates must be positive when provided.")
        if not isinstance(use_alpha_beta, bool):
            raise ValueError("use_alpha_beta must be a bool.")
        if not isinstance(use_iterative_deepening, bool):
            raise ValueError("use_iterative_deepening must be a bool.")
        if not isinstance(use_transposition_table, bool):
            raise ValueError("use_transposition_table must be a bool.")

        self.evaluator = evaluator
        self.max_depth = int(max_depth)
        self.candidate_radius = int(candidate_radius)
        self.max_candidates = max_candidates
        self.use_alpha_beta = use_alpha_beta
        self.use_iterative_deepening = use_iterative_deepening
        self.use_transposition_table = use_transposition_table

    def search(self, env: GomokuEnv) -> SearchResult:
        """Search one root move with the pure negamax core."""

        if self.use_iterative_deepening:
            result = run_iterative_deepening_search(
                env,
                max_depth=self.max_depth,
                evaluator=self.evaluator,
                radius=self.candidate_radius,
                max_candidates=self.max_candidates,
                use_alpha_beta=self.use_alpha_beta,
                use_transposition_table=self.use_transposition_table,
            )
        else:
            result = run_negamax_search(
                env,
                depth=self.max_depth,
                evaluator=self.evaluator,
                radius=self.candidate_radius,
                max_candidates=self.max_candidates,
                use_alpha_beta=self.use_alpha_beta,
                use_transposition_table=self.use_transposition_table,
            )
        best_action = -1 if result.best_action is None else int(result.best_action)
        return SearchResult(
            best_action=best_action,
            root_value=float(result.value),
            action_values=dict(result.action_values),
            principal_variation=list(result.principal_variation),
            nodes=int(result.nodes),
            depth_reached=int(result.depth_reached),
            forced_tactical=False,
            cutoffs=int(result.cutoffs),
            pruned_branches=int(result.pruned_branches),
            tt_hits=int(result.tt_hits),
            tt_stores=int(result.tt_stores),
        )
