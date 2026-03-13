"""PyTorch model definition for AlphaZero-compatible policy/value inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gomoku_ai.alphazero.specs import ACTION_SIZE, STATE_CHANNELS
from gomoku_ai.env import BOARD_SIZE

CHECKPOINT_FORMAT_VERSION = 1
MODEL_TYPE = "PolicyValueNet"


@dataclass(frozen=True)
class PolicyValueNetConfig:
    """Configuration for the baseline Gomoku policy/value network."""

    in_channels: int = STATE_CHANNELS
    board_size: int = BOARD_SIZE
    trunk_channels: int = 64
    num_trunk_layers: int = 3
    value_hidden_dim: int = 128
    policy_head_channels: int = 2
    value_head_channels: int = 1
    use_batch_norm: bool = True

    def __post_init__(self) -> None:
        """Validate that the config matches the fixed Gomoku AlphaZero contract."""

        if self.in_channels != STATE_CHANNELS:
            raise ValueError(f"in_channels must be {STATE_CHANNELS}, got {self.in_channels}.")
        if self.board_size != BOARD_SIZE:
            raise ValueError(f"board_size must be {BOARD_SIZE}, got {self.board_size}.")
        for field_name in (
            "trunk_channels",
            "num_trunk_layers",
            "value_hidden_dim",
            "policy_head_channels",
            "value_head_channels",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive.")

    @property
    def action_size(self) -> int:
        """Return the flattened board action space size."""

        return self.board_size * self.board_size


class ConvBlock(nn.Module):
    """Simple Conv-BN-ReLU block used across the network trunk and heads."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        padding: int,
        use_batch_norm: bool,
    ) -> None:
        """Initialize a convolutional block."""

        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional block."""

        return self.block(x)


def _validate_input_tensor(x: torch.Tensor, config: PolicyValueNetConfig) -> None:
    """Validate the model input against the fixed AlphaZero tensor contract."""

    if not isinstance(x, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor.")
    if x.ndim != 4:
        raise ValueError(f"Input must have shape (N, {config.in_channels}, {config.board_size}, {config.board_size}).")
    if x.shape[1] != config.in_channels:
        raise ValueError(f"Input channel dimension must be {config.in_channels}, got {x.shape[1]}.")
    if x.shape[2] != config.board_size or x.shape[3] != config.board_size:
        raise ValueError(
            f"Input spatial dimensions must be ({config.board_size}, {config.board_size}), got {tuple(x.shape[2:])}."
        )
    if x.dtype != torch.float32:
        raise ValueError(f"Input dtype must be torch.float32, got {x.dtype}.")
    if x.shape[0] <= 0:
        raise ValueError("Batch size must be positive.")
    if not torch.isfinite(x).all():
        raise ValueError("Input tensor must contain only finite values.")


class PolicyValueNet(nn.Module):
    """Baseline CNN that predicts raw policy logits and a scalar value."""

    def __init__(self, config: PolicyValueNetConfig | None = None) -> None:
        """Initialize the policy/value network using a validated config."""

        super().__init__()
        self.config = config or PolicyValueNetConfig()

        trunk_layers: list[nn.Module] = []
        in_channels = self.config.in_channels
        for _ in range(self.config.num_trunk_layers):
            trunk_layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=self.config.trunk_channels,
                    kernel_size=3,
                    padding=1,
                    use_batch_norm=self.config.use_batch_norm,
                )
            )
            in_channels = self.config.trunk_channels
        self.trunk = nn.Sequential(*trunk_layers)

        self.policy_head = nn.Sequential(
            ConvBlock(
                in_channels=self.config.trunk_channels,
                out_channels=self.config.policy_head_channels,
                kernel_size=1,
                padding=0,
                use_batch_norm=self.config.use_batch_norm,
            ),
            nn.Flatten(),
            nn.Linear(self.config.policy_head_channels * self.config.action_size, self.config.action_size),
        )

        self.value_head = nn.Sequential(
            ConvBlock(
                in_channels=self.config.trunk_channels,
                out_channels=self.config.value_head_channels,
                kernel_size=1,
                padding=0,
                use_batch_norm=self.config.use_batch_norm,
            ),
            nn.Flatten(),
            nn.Linear(self.config.value_head_channels * self.config.action_size, self.config.value_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.value_hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass and return raw policy logits and bounded values."""

        _validate_input_tensor(x, self.config)
        features = self.trunk(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value

    def predict_single(
        self,
        state_np: Any,
        device: torch.device | str | None = None,
        *,
        move_model: bool = False,
    ) -> tuple[Any, float]:
        """Run single-state inference using the shared AlphaZero inference helper."""

        from gomoku_ai.alphazero.inference import predict_single

        return predict_single(self, state_np, device=device, move_model=move_model)


def count_parameters(model: nn.Module) -> int:
    """Return the total number of parameters in a model."""

    return sum(parameter.numel() for parameter in model.parameters())


def save_policy_value_net(model: PolicyValueNet, path: str | Path) -> None:
    """Save the network state and config to disk."""

    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_type": MODEL_TYPE,
            "state_dict": model.state_dict(),
            "config": asdict(model.config),
        },
        resolved_path,
    )


def load_policy_value_net(path: str | Path, device: torch.device | str | None = None) -> PolicyValueNet:
    """Load a `PolicyValueNet` checkpoint and reconstruct its config."""

    map_location = None if device is None else torch.device(device)
    checkpoint = torch.load(Path(path), map_location=map_location)
    required_fields = {"format_version", "model_type", "state_dict", "config"}
    if not required_fields.issubset(checkpoint):
        raise ValueError(
            "Checkpoint must contain 'format_version', 'model_type', 'state_dict', and 'config'."
        )
    if checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format_version {checkpoint['format_version']!r}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}."
        )
    if checkpoint["model_type"] != MODEL_TYPE:
        raise ValueError(f"Unsupported model_type {checkpoint['model_type']!r}; expected {MODEL_TYPE!r}.")

    config = PolicyValueNetConfig(**dict(checkpoint["config"]))
    model = PolicyValueNet(config=config)
    model.load_state_dict(checkpoint["state_dict"])
    if device is not None:
        model = model.to(torch.device(device))
    return model
