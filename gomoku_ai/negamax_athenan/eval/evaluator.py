"""Pattern-based evaluator for the negamax Athenan stack."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from gomoku_ai.env import BLACK, DIRECTIONS, DRAW, EMPTY, GomokuEnv, WHITE


@dataclass(frozen=True)
class GreedyHeuristicConfig:
    """Weight/config bundle for the 1-ply heuristic evaluator."""

    terminal_win_score: float = 1_000_000.0
    opponent_weight: float = 1.1
    five_score: float = 250_000.0
    open_four_score: float = 40_000.0
    closed_four_score: float = 12_000.0
    open_three_score: float = 2_500.0
    closed_three_score: float = 600.0
    open_two_score: float = 120.0
    center_weight: float = 0.15
    connectivity_weight: float = 1.0


@dataclass(frozen=True)
class PatternSummary:
    """Contiguous-pattern counts for one player in one position."""

    five: int = 0
    open_four: int = 0
    closed_four: int = 0
    open_three: int = 0
    closed_three: int = 0
    open_two: int = 0

    def weighted_score(self, config: GreedyHeuristicConfig) -> float:
        """Convert counted patterns into one scalar heuristic score."""

        return float(
            (self.five * config.five_score)
            + (self.open_four * config.open_four_score)
            + (self.closed_four * config.closed_four_score)
            + (self.open_three * config.open_three_score)
            + (self.closed_three * config.closed_three_score)
            + (self.open_two * config.open_two_score)
        )


class GreedyHeuristicEvaluator:
    """Pattern-based static evaluator used by greedy play and search leaves.

    The evaluator stays lightweight but now distinguishes core Gomoku threats:
    - terminal positions dominate the score
    - FIVE / OPEN_FOUR / CLOSED_FOUR / OPEN_THREE / CLOSED_THREE / OPEN_TWO
      are scored explicitly
    - center control and local connectivity add small tie-breaking guidance
    """

    def __init__(self, config: GreedyHeuristicConfig | None = None) -> None:
        self.config = config or GreedyHeuristicConfig()

    def evaluate_for_player(self, env: GomokuEnv, player: int) -> float:
        """Return a static score from `player`'s perspective."""

        if player not in {BLACK, WHITE}:
            raise ValueError(f"player must be BLACK({BLACK}) or WHITE({WHITE}).")

        if env.done:
            if env.winner == DRAW:
                return 0.0
            if env.winner == player:
                return float(self.config.terminal_win_score)
            return float(-self.config.terminal_win_score)

        opponent = WHITE if player == BLACK else BLACK
        player_score = self._score_player(env, player)
        opponent_score = self._score_player(env, opponent)
        return float(player_score - (self.config.opponent_weight * opponent_score))

    def count_patterns_for_player(self, env: GomokuEnv, player: int) -> PatternSummary:
        """Count contiguous tactical patterns for one player.

        Duplicate counting is avoided by scanning only run start points:
        a stone contributes to a directional run only when the previous cell in
        that direction is not the same player's stone.
        """

        if player not in {BLACK, WHITE}:
            raise ValueError(f"player must be BLACK({BLACK}) or WHITE({WHITE}).")

        counts = {
            "five": 0,
            "open_four": 0,
            "closed_four": 0,
            "open_three": 0,
            "closed_three": 0,
            "open_two": 0,
        }
        for row in range(env.board_size):
            for col in range(env.board_size):
                if int(env.board[row, col]) != player:
                    continue
                for delta_row, delta_col in DIRECTIONS:
                    prev_row = row - delta_row
                    prev_col = col - delta_col
                    if (
                        0 <= prev_row < env.board_size
                        and 0 <= prev_col < env.board_size
                        and int(env.board[prev_row, prev_col]) == player
                    ):
                        continue

                    length, end_row, end_col = self._line_extent(env, row, col, player, delta_row, delta_col)
                    open_ends = self._count_open_ends(
                        env,
                        prev_row=prev_row,
                        prev_col=prev_col,
                        end_row=end_row,
                        end_col=end_col,
                    )
                    pattern_name = self._classify_contiguous_pattern(length, open_ends)
                    if pattern_name is not None:
                        counts[pattern_name] += 1

        return PatternSummary(**counts)

    def score_patterns_for_player(self, env: GomokuEnv, player: int) -> float:
        """Return the weighted contiguous-pattern score for one player."""

        return self.count_patterns_for_player(env, player).weighted_score(self.config)

    def score_action_for_player(self, env: GomokuEnv, action: int, player: int) -> float:
        """Score the child position after `player` plays one legal action now."""

        if env.done:
            raise RuntimeError("Cannot score child actions from a terminal position.")
        if player not in {BLACK, WHITE}:
            raise ValueError(f"player must be BLACK({BLACK}) or WHITE({WHITE}).")
        if int(env.current_player) != int(player):
            raise ValueError("score_action_for_player requires player to equal env.current_player.")
        if not env.is_legal_action(action):
            raise ValueError(f"Action {action} is not legal in the current position.")

        child = env.clone()
        child.apply_move(int(action))
        return self.evaluate_for_player(child, player)

    def would_action_win_for_player(self, env: GomokuEnv, action: int, player: int) -> bool:
        """Return `True` if placing `player` at `action` creates five immediately."""

        if player not in {BLACK, WHITE}:
            raise ValueError(f"player must be BLACK({BLACK}) or WHITE({WHITE}).")
        if env.done or not env.is_legal_action(action):
            return False

        row, col = env.action_to_coord(int(action))
        return self._projected_max_line_length(env, row, col, player) >= 5

    def _score_player(self, env: GomokuEnv, player: int) -> float:
        """Aggregate tactical patterns plus light positional tie-breakers."""

        return float(
            self.score_patterns_for_player(env, player)
            + (self.config.center_weight * self._score_center_control(env, player))
            + (self.config.connectivity_weight * self._score_connectivity(env, player))
        )

    def _score_center_control(self, env: GomokuEnv, player: int) -> float:
        center = (env.board_size - 1) / 2.0
        score = 0.0
        for row, col in np.argwhere(env.board == player):
            distance = abs(float(row) - center) + abs(float(col) - center)
            score += float((env.board_size * 2) - distance)
        return score

    def _score_connectivity(self, env: GomokuEnv, player: int) -> float:
        score = 0.0
        for row, col in np.argwhere(env.board == player):
            neighbors = 0
            for delta_row in (-1, 0, 1):
                for delta_col in (-1, 0, 1):
                    if delta_row == 0 and delta_col == 0:
                        continue
                    next_row = int(row + delta_row)
                    next_col = int(col + delta_col)
                    if 0 <= next_row < env.board_size and 0 <= next_col < env.board_size:
                        if int(env.board[next_row, next_col]) == player:
                            neighbors += 1
            score += neighbors * 0.5
        return score

    def _projected_max_line_length(self, env: GomokuEnv, row: int, col: int, player: int) -> int:
        max_length = 1
        for delta_row, delta_col in DIRECTIONS:
            length = 1
            length += self._count_direction(env, row, col, player, delta_row, delta_col)
            length += self._count_direction(env, row, col, player, -delta_row, -delta_col)
            if length > max_length:
                max_length = length
        return max_length

    @staticmethod
    def _count_open_ends(
        env: GomokuEnv,
        *,
        prev_row: int,
        prev_col: int,
        end_row: int,
        end_col: int,
    ) -> int:
        open_ends = 0
        if (
            0 <= prev_row < env.board_size
            and 0 <= prev_col < env.board_size
            and int(env.board[prev_row, prev_col]) == EMPTY
        ):
            open_ends += 1
        if (
            0 <= end_row < env.board_size
            and 0 <= end_col < env.board_size
            and int(env.board[end_row, end_col]) == EMPTY
        ):
            open_ends += 1
        return open_ends

    @staticmethod
    def _line_extent(
        env: GomokuEnv,
        row: int,
        col: int,
        player: int,
        delta_row: int,
        delta_col: int,
    ) -> tuple[int, int, int]:
        length = 1
        next_row = row + delta_row
        next_col = col + delta_col
        while (
            0 <= next_row < env.board_size
            and 0 <= next_col < env.board_size
            and int(env.board[next_row, next_col]) == player
        ):
            length += 1
            next_row += delta_row
            next_col += delta_col
        return length, next_row, next_col

    @staticmethod
    def _count_direction(
        env: GomokuEnv,
        row: int,
        col: int,
        player: int,
        delta_row: int,
        delta_col: int,
    ) -> int:
        count = 0
        next_row = row + delta_row
        next_col = col + delta_col
        while 0 <= next_row < env.board_size and 0 <= next_col < env.board_size:
            if int(env.board[next_row, next_col]) != player:
                break
            count += 1
            next_row += delta_row
            next_col += delta_col
        return count

    @staticmethod
    def _classify_contiguous_pattern(length: int, open_ends: int) -> str | None:
        if length >= 5:
            return "five"
        if length == 4:
            if open_ends == 2:
                return "open_four"
            if open_ends == 1:
                return "closed_four"
            return None
        if length == 3:
            if open_ends == 2:
                return "open_three"
            if open_ends == 1:
                return "closed_three"
            return None
        if length == 2:
            return "open_two" if open_ends == 2 else None
        return None

