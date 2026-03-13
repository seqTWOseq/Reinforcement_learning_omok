"""Monte Carlo Tree Search implementation for AlphaZero-compatible Gomoku."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gomoku_ai.alphazero.mcts_utils import opponent_of, resolve_env_player_to_move, terminal_value_for_player
from gomoku_ai.alphazero.model import PolicyValueNet
from gomoku_ai.alphazero.specs import ACTION_SIZE
from gomoku_ai.alphazero.utils import policy_logits_to_probs
from gomoku_ai.env import BLACK, GomokuEnv, WHITE


@dataclass(frozen=True)
class MCTSConfig:
    """Configuration for AlphaZero-style MCTS search."""

    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    add_root_noise: bool = True
    temperature: float = 1.0

    def __post_init__(self) -> None:
        """Validate the search hyper-parameters."""

        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be positive.")
        if self.c_puct <= 0.0:
            raise ValueError("c_puct must be positive.")
        if self.dirichlet_alpha <= 0.0:
            raise ValueError("dirichlet_alpha must be positive.")
        if not 0.0 <= self.dirichlet_epsilon <= 1.0:
            raise ValueError("dirichlet_epsilon must be in [0.0, 1.0].")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative.")


@dataclass
class MCTSNode:
    """Single MCTS node.

    `q_value()` is always stored from this node's `player_to_move` perspective.
    When a parent selects among children, it must negate the child's Q value to
    convert it into the parent's perspective.
    """

    prior: float
    player_to_move: int
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    action_taken: int | None = None
    is_terminal: bool = False
    terminal_value: float | None = None

    def __post_init__(self) -> None:
        """Validate node metadata and value ranges."""

        if not np.isfinite(self.prior) or self.prior < 0.0:
            raise ValueError("prior must be a finite non-negative float.")
        if self.visit_count < 0:
            raise ValueError("visit_count must be non-negative.")
        if self.player_to_move not in {BLACK, WHITE}:
            raise ValueError(f"player_to_move must be BLACK({BLACK}) or WHITE({WHITE}).")
        if self.action_taken is not None and not 0 <= self.action_taken < ACTION_SIZE:
            raise ValueError(f"action_taken must be in [0, {ACTION_SIZE - 1}] when provided.")
        if self.terminal_value is not None:
            if not np.isfinite(self.terminal_value) or not -1.0 <= self.terminal_value <= 1.0:
                raise ValueError("terminal_value must be finite and inside [-1.0, 1.0].")

    def q_value(self) -> float:
        """Return the average value from this node's `player_to_move` perspective."""

        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self) -> bool:
        """Return `True` when the node already has child priors."""

        return bool(self.children)


class MCTS:
    """AlphaZero-style Monte Carlo Tree Search.

    Value convention:
    - Network value predictions are interpreted from the leaf state's
      `player_to_move` perspective.
    - Each node stores accumulated values from its own `player_to_move`
      perspective.
    - During backpropagation the sign flips at every level.
    - During PUCT selection, the parent negates `child.q_value()` to score the
      child action from the parent's perspective.
    """

    def __init__(self, config: MCTSConfig | None = None) -> None:
        """Initialize the MCTS engine with a validated config."""

        self.config = config or MCTSConfig()

    def run(self, root_env: GomokuEnv, model: PolicyValueNet) -> MCTSNode:
        """Run MCTS from `root_env` without mutating it and return the root node."""

        if not isinstance(root_env, GomokuEnv):
            raise TypeError("root_env must be a GomokuEnv instance.")
        if not isinstance(model, PolicyValueNet):
            raise TypeError("model must be a PolicyValueNet instance.")

        root_snapshot = root_env.clone()
        root = MCTSNode(
            prior=1.0,
            action_taken=None,
            player_to_move=resolve_env_player_to_move(root_snapshot),
            is_terminal=root_snapshot.done,
        )
        if root.is_terminal:
            root.terminal_value = terminal_value_for_player(root_snapshot, root.player_to_move)
            return root

        self._expand(root, root_snapshot, model)
        if self.config.add_root_noise:
            self._apply_dirichlet_noise(root)

        for _ in range(self.config.num_simulations):
            simulation_env = root_env.clone()
            node = root
            search_path = [root]

            while node.expanded() and not node.is_terminal:
                action, child = self._select_child(node)
                simulation_env.apply_move(action)
                node = child
                search_path.append(node)
                if node.is_terminal:
                    break

            leaf_value = self._evaluate_leaf(node, simulation_env, model)
            self._backpropagate(search_path, leaf_value)

        return root

    def get_action_probs(self, root: MCTSNode, temperature: float | None = None) -> np.ndarray:
        """Return a `(225,)` visit-count distribution from the root node."""

        if not isinstance(root, MCTSNode):
            raise TypeError("root must be an MCTSNode instance.")
        if not root.children:
            raise ValueError("Root node must have children to produce action probabilities.")

        resolved_temperature = self.config.temperature if temperature is None else temperature
        if resolved_temperature < 0.0:
            raise ValueError("temperature must be non-negative.")

        action_probs = np.zeros(ACTION_SIZE, dtype=np.float32)
        visit_counts = np.zeros(ACTION_SIZE, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = float(child.visit_count)

        if resolved_temperature == 0.0:
            best_action = max(sorted(root.children), key=lambda action: root.children[action].visit_count)
            action_probs[best_action] = 1.0
            return action_probs

        adjusted_counts = visit_counts ** (1.0 / resolved_temperature)
        total = float(adjusted_counts.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError("Visit counts must produce a positive finite probability mass.")
        action_probs = (adjusted_counts / total).astype(np.float32, copy=False)

        if not np.isclose(float(action_probs.sum()), 1.0, atol=1e-5):
            raise ValueError("Action probabilities must sum to 1.0.")
        return action_probs

    def select_action(self, root: MCTSNode, temperature: float | None = None) -> int:
        """Select an action from the root visit counts."""

        if not isinstance(root, MCTSNode):
            raise TypeError("root must be an MCTSNode instance.")
        if not root.children:
            raise ValueError("Root node must have children to select an action.")

        resolved_temperature = self.config.temperature if temperature is None else temperature
        if resolved_temperature == 0.0:
            return int(max(sorted(root.children), key=lambda action: root.children[action].visit_count))

        action_probs = self.get_action_probs(root, temperature=resolved_temperature)
        actions = np.flatnonzero(action_probs > 0.0)
        return int(np.random.choice(actions, p=action_probs[actions]))

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        """Select the child with the highest PUCT score."""

        if not node.children:
            raise ValueError("Cannot select a child from an unexpanded node.")

        parent_scale = float(np.sqrt(node.visit_count + 1.0))
        best_action: int | None = None
        best_child: MCTSNode | None = None
        best_score = -np.inf

        for action, child in node.children.items():
            q_score = -child.q_value()
            u_score = self.config.c_puct * child.prior * parent_scale / (1.0 + child.visit_count)
            score = q_score + u_score
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_action is None or best_child is None:
            raise RuntimeError("Failed to select a child despite the node having children.")
        return best_action, best_child

    def _expand(self, node: MCTSNode, env: GomokuEnv, model: PolicyValueNet) -> float:
        """Expand a non-terminal leaf and return its network value."""

        if env.done:
            raise ValueError("Use _evaluate_leaf() for terminal nodes.")

        policy_logits, value = self._predict_policy_value(env, model)
        valid_moves = env.get_valid_moves()
        priors = policy_logits_to_probs(policy_logits, valid_moves)
        child_player_to_move = opponent_of(node.player_to_move)

        for action in np.flatnonzero(valid_moves):
            action_int = int(action)
            if action_int not in node.children:
                node.children[action_int] = MCTSNode(
                    prior=float(priors[action_int]),
                    action_taken=action_int,
                    player_to_move=child_player_to_move,
                )

        return float(value)

    def _evaluate_leaf(self, node: MCTSNode, env: GomokuEnv, model: PolicyValueNet) -> float:
        """Evaluate a leaf node from the leaf node's perspective."""

        if env.done:
            node.is_terminal = True
            node.terminal_value = terminal_value_for_player(env, node.player_to_move)
            return node.terminal_value
        if node.expanded():
            raise RuntimeError("Encountered an expanded non-terminal leaf during evaluation.")
        return self._expand(node, env, model)

    def _backpropagate(self, search_path: list[MCTSNode], leaf_value: float) -> None:
        """Backpropagate a leaf value, flipping sign at each parent step."""

        if not np.isfinite(leaf_value) or not -1.0 <= leaf_value <= 1.0:
            raise ValueError("leaf_value must be finite and inside [-1.0, 1.0].")

        value = float(leaf_value)
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _apply_dirichlet_noise(self, root: MCTSNode) -> None:
        """Apply AlphaZero-style Dirichlet noise to root priors once."""

        if not root.children:
            return

        actions = sorted(root.children)
        noise = np.random.dirichlet(
            np.full(len(actions), self.config.dirichlet_alpha, dtype=np.float64)
        ).astype(np.float32, copy=False)

        for index, action in enumerate(actions):
            child = root.children[action]
            child.prior = (1.0 - self.config.dirichlet_epsilon) * child.prior + (
                self.config.dirichlet_epsilon * float(noise[index])
            )

        total_prior = sum(child.prior for child in root.children.values())
        if total_prior <= 0.0 or not np.isfinite(total_prior):
            raise ValueError("Dirichlet-adjusted priors must sum to a positive finite value.")
        for child in root.children.values():
            child.prior /= total_prior

    def _predict_policy_value(self, env: GomokuEnv, model: PolicyValueNet) -> tuple[np.ndarray, float]:
        """Run single-state network inference and validate the returned shapes."""

        policy_logits, value = model.predict_single(env.encode_state())
        policy_logits_array = np.asarray(policy_logits, dtype=np.float32)
        if policy_logits_array.shape != (ACTION_SIZE,):
            raise ValueError(f"Model policy output must have shape ({ACTION_SIZE},), got {policy_logits_array.shape}.")
        if not np.isfinite(policy_logits_array).all():
            raise ValueError("Model policy logits must contain only finite values.")

        value_scalar = float(value)
        if not np.isfinite(value_scalar) or not -1.0 <= value_scalar <= 1.0:
            raise ValueError("Model value output must be finite and inside [-1.0, 1.0].")
        return policy_logits_array, value_scalar
