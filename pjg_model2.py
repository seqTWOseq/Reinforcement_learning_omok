from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

import pjg_model as core


@dataclass(frozen=True)
class TrainConfig:
    total_timesteps: int = 200_000
    save_interval: int = 10_000
    opponent_pool_size: int = 5
    lr: float = 3e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 4
    model_dir: str = "gomoku_models"
    seed: int = 42
    render: bool = False
    device_str: str = "auto"
    resume_from: Optional[str] = None


@dataclass(frozen=True)
class PlayConfig:
    model_path: Optional[str] = None
    model_dir: str = "gomoku_models"
    human_plays_black: bool = True
    device_str: str = "auto"


class ObservationEncoding:
    """Class-style facade for observation/mask helpers."""

    @staticmethod
    def make_obs_from_board(board, current_player):
        return core.make_obs_from_board(board, current_player)

    @staticmethod
    def get_action_mask(board):
        return core.get_action_mask(board)


class RuleMoveGenerator:
    """Class-style facade for tactical rule selection + shaping tiers."""

    SHAPING_REWARD_BY_TIER = core.SHAPING_REWARD_BY_TIER

    @staticmethod
    def get_rule_tier(board, current_player, action, mask):
        return core.get_rule_tier(board, current_player, action, mask)

    @staticmethod
    def select_action_with_rules(board, current_player, mask):
        return core.select_action_with_rules(board, current_player, mask)


class SymmetricAugment:
    """Class-style facade for 8-way symmetry augmentation."""

    @staticmethod
    def augment_obs_action(obs, action, board_size=15):
        return core.augment_obs_action(obs, action, board_size=board_size)

    @staticmethod
    def augment_mask(mask, action, board_size=15):
        return core.augment_mask(mask, action, board_size=board_size)


class TorchCheckpoint:
    """Unified checkpoint API."""

    @staticmethod
    def save(model, path: str):
        core.save_model(model, path)

    @staticmethod
    def load(model, path: str, device: str = "cpu"):
        return core.load_model(model, path, device=device)


class PJGModel2:
    """
    Class-oriented unified entrypoint built on top of `pjg_model.py`.
    This keeps the original training/play behavior, but exposes it with
    a structure similar to class-centric model modules.
    """

    def __init__(self, model_dir: str = "gomoku_models"):
        self.model_dir = model_dir

    @staticmethod
    def resolve_device(device_str: str = "auto") -> str:
        return core._resolve_device(device_str)

    @staticmethod
    def create_policy(board_size: int = 15):
        return core.GomokuCNN(board_size=board_size, n_actions=board_size**2)

    @staticmethod
    def load_policy(model_path: str, board_size: int = 15, device_str: str = "auto"):
        device = core._resolve_device(device_str)
        model = core.GomokuCNN(board_size=board_size, n_actions=board_size**2).to(device)
        core.load_model(model, model_path, device=device)
        model.eval()
        return model

    def train(self, config: TrainConfig = TrainConfig()):
        return core.run_training(
            total_timesteps=config.total_timesteps,
            save_interval=config.save_interval,
            opponent_pool_size=config.opponent_pool_size,
            lr=config.lr,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            model_dir=config.model_dir,
            seed=config.seed,
            render=config.render,
            device_str=config.device_str,
            resume_from=config.resume_from,
        )

    def play(self, config: PlayConfig = PlayConfig()):
        return core.run_play(
            model_path=config.model_path,
            model_dir=config.model_dir,
            human_plays_black=config.human_plays_black,
            device_str=config.device_str,
        )


def main():
    """
    Keep CLI behavior consistent with pjg_model.py.
    We simply delegate to the original main.
    """
    core.main()


__all__ = [
    "ObservationEncoding",
    "PlayConfig",
    "PJGModel2",
    "RuleMoveGenerator",
    "SymmetricAugment",
    "TorchCheckpoint",
    "TrainConfig",
    "main",
]


