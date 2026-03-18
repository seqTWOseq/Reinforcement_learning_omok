"""Pattern-evaluator + negamax Athenan package."""

from gomoku_ai.negamax_athenan.agent import AthenanGreedyHeuristicAgent
from gomoku_ai.negamax_athenan.eval import (
    GreedyHeuristicConfig,
    GreedyHeuristicEvaluator,
    PatternSummary,
)
from gomoku_ai.negamax_athenan.search import (
    AthenanNegamaxSearcher,
    TTEntry,
    apply_forced_tactical_rule,
    find_immediate_blocking_actions,
    find_immediate_winning_actions,
    generate_candidate_actions,
    generate_proximity_candidates,
    negamax,
    order_actions,
    order_candidate_actions,
    run_iterative_deepening_search,
    run_negamax_search,
    score_action,
)

__all__ = [
    "AthenanGreedyHeuristicAgent",
    "AthenanNegamaxSearcher",
    "GreedyHeuristicConfig",
    "GreedyHeuristicEvaluator",
    "PatternSummary",
    "TTEntry",
    "apply_forced_tactical_rule",
    "find_immediate_blocking_actions",
    "find_immediate_winning_actions",
    "generate_candidate_actions",
    "generate_proximity_candidates",
    "negamax",
    "order_actions",
    "order_candidate_actions",
    "run_iterative_deepening_search",
    "run_negamax_search",
    "score_action",
]
