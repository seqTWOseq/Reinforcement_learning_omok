"""Play entrypoints for the negamax Athenan package."""

from gomoku_ai.common.play.human_vs_searcher import AthenanAIDebugTurn, HumanVsAthenanResult
from gomoku_ai.negamax_athenan.play.play_human_vs_negamax_athenan import (
    play_human_vs_negamax_athenan_game,
)

__all__ = [
    "AthenanAIDebugTurn",
    "HumanVsAthenanResult",
    "play_human_vs_negamax_athenan_game",
]
