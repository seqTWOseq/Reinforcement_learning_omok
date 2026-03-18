"""Greedy heuristic Athenan agent for stage-3 non-search play."""

from __future__ import annotations

from gomoku_ai.negamax_athenan.eval.evaluator import GreedyHeuristicEvaluator
from gomoku_ai.negamax_athenan.search.move_generator import generate_candidate_actions
from gomoku_ai.common.agents import BaseAgent
from gomoku_ai.env import GomokuEnv


class AthenanGreedyHeuristicAgent(BaseAgent):
    """Choose one move from the shared candidate ordering.

    Candidate selection is delegated to the shared move generator so the same
    pruning/order policy can later feed deeper search.
    """

    def __init__(
        self,
        evaluator: GreedyHeuristicEvaluator | None = None,
        *,
        candidate_radius: int = 2,
        max_candidates: int | None = None,
    ) -> None:
        self.evaluator = evaluator or GreedyHeuristicEvaluator()
        self.candidate_radius = int(candidate_radius)
        self.max_candidates = max_candidates

    def select_action(self, env: GomokuEnv) -> int:
        """Return one deterministic legal action for the current position."""

        if not isinstance(env, GomokuEnv):
            raise TypeError("env must be a GomokuEnv instance.")

        candidate_actions = generate_candidate_actions(
            env,
            radius=self.candidate_radius,
            max_candidates=self.max_candidates,
            evaluator=self.evaluator,
        )
        if not candidate_actions:
            raise RuntimeError("No legal moves are available in the current position.")
        return candidate_actions[0]
