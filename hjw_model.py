import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tkinter as tk
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. 기본 설정 및 장치
# ==========================================
BOARD_SIZE = 15
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 알파제로 두뇌 (Network) & 수읽기 (MCTS)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += res
        return F.relu(x)

class AlphaZeroNet(nn.Module):
    def __init__(self, num_blocks=3, channels=64):
        super().__init__()
        self.start_conv = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])
        
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, ACTION_SIZE)
        
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(-1, 1, BOARD_SIZE, BOARD_SIZE)
        x = self.start_conv(x)
        for block in self.res_blocks:
            x = block(x)
            
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 2 * BOARD_SIZE * BOARD_SIZE)
        policy = self.policy_fc(p)
        
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, BOARD_SIZE * BOARD_SIZE)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return F.log_softmax(policy, dim=1), value

class GomokuGame:
    """MCTS가 머릿속으로 시뮬레이션할 때 쓸 가상의 게임판"""
    def __init__(self, size=BOARD_SIZE):
        self.size = size

    def get_next_state(self, state, action, player):
        next_state = np.copy(state)
        next_state[action // self.size, action % self.size] = player
        return next_state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action is None: return False
        row, col = action // self.size, action % self.size
        player = state[row, col]
        if player == 0: return False
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for step in (1, -1):
                r, c = row + dr * step, col + dc * step
                while 0 <= r < self.size and 0 <= c < self.size and state[r, c] == player:
                    count += 1
                    r += dr * step; c += dc * step
            if count >= 5: return True
        return False

    def get_reward_and_ended(self, state, action):
        if self.check_win(state, action): return 1.0, True
        if np.sum(self.get_valid_moves(state)) == 0: return 0.0, True
        return 0.0, False

    def get_canonical_form(self, state, player):
        return state * player

class Node:
    def __init__(self, parent=None, prior_prob=1.0):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior = prior_prob

    def is_expanded(self):
        return len(self.children) > 0

    def get_ucb(self, c_puct=1.0):
        q_value = 0 if self.visits == 0 else self.value_sum / self.visits
        u_value = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q_value + u_value

class MCTS:
    def __init__(self, game, model, simulations=100):
        self.game = game
        self.model = model
        self.simulations = simulations

    @torch.no_grad()
    def search(self, state):
        root = Node()
        for _ in range(self.simulations):
            node = root
            current_state = np.copy(state)
            current_player = 1
            action_history = []

            while node.is_expanded():
                best_action = max(node.children.keys(), key=lambda a: node.children[a].get_ucb())
                node = node.children[best_action]
                current_state = self.game.get_next_state(current_state, best_action, current_player)
                action_history.append(best_action)
                current_player *= -1

            last_action = action_history[-1] if action_history else None
            reward, is_terminal = self.game.get_reward_and_ended(current_state, last_action)

            if not is_terminal:
                canonical_state = self.game.get_canonical_form(current_state, current_player)
                state_tensor = torch.FloatTensor(canonical_state).to(device)
                policy, value = self.model(state_tensor)
                policy = torch.exp(policy).cpu().numpy().flatten()
                value = value.item()

                valid_moves = self.game.get_valid_moves(current_state)
                policy = policy * valid_moves
                sum_policy = np.sum(policy)
                if sum_policy > 0: policy /= sum_policy
                else: policy = valid_moves / np.sum(valid_moves)

                for action, prob in enumerate(policy):
                    if valid_moves[action]:
                        node.children[action] = Node(parent=node, prior_prob=prob)
            else:
                value = reward

            while node is not None:
                node.visits += 1
                node.value_sum += value
                value = -value
                node = node.parent

        action_probs = np.zeros(ACTION_SIZE)
        for action, child in root.children.items():
            action_probs[action] = child.visits
        action_probs /= np.sum(action_probs)
        return action_probs