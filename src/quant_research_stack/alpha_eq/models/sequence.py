"""1D-CNN base learner for S1-EQ (spec §4.3).

Expects 3-D input shape (n_samples, lookback, feature_channels).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


@dataclass(frozen=True)
class Conv1DEqConfig:
    lookback: int = 20
    feature_channels: int = 8
    hidden_channels: int = 32
    kernel_size: int = 3
    learning_rate: float = 1.0e-3
    batch_size: int = 256
    max_epochs: int = 20
    seed: int = 42


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Net(nn.Module):
    def __init__(
        self,
        *,
        lookback: int,
        feature_channels: int,
        hidden_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                feature_channels, hidden_channels,
                kernel_size=kernel_size, padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x.permute(0, 2, 1)).squeeze(-1)
        return self.head(z).squeeze(-1)


class Conv1DEqModel:
    def __init__(self, config: Conv1DEqConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self._net: _Net | None = None

    def fit(self, *, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        if x.ndim != 3:
            raise ValueError(f"Conv1D requires 3-D input (B,T,C); got shape={x.shape}")
        device = _device()
        self._net = _Net(
            lookback=self.config.lookback,
            feature_channels=self.config.feature_channels,
            hidden_channels=self.config.hidden_channels,
            kernel_size=self.config.kernel_size,
        ).to(device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        bs = self.config.batch_size
        for _ in range(self.config.max_epochs):
            perm = torch.randperm(x_t.shape[0], device=device)
            for i in range(0, x_t.shape[0], bs):
                idx = perm[i : i + bs]
                opt.zero_grad()
                out = self._net(x_t[idx])
                loss = loss_fn(out, y_t[idx])
                loss.backward()
                opt.step()

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._net is None:
            raise RuntimeError("model not fit")
        device = next(self._net.parameters()).device
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = self._net(x_t).cpu().numpy()
        return out.astype(np.float64)

    def save(self, path: Path) -> None:
        if self._net is None:
            raise RuntimeError("model not fit")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self._net.state_dict(), "config": asdict(self.config)},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> Conv1DEqModel:
        device = _device()
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg = Conv1DEqConfig(**payload["config"])
        m = cls(cfg)
        m._net = _Net(
            lookback=cfg.lookback,
            feature_channels=cfg.feature_channels,
            hidden_channels=cfg.hidden_channels,
            kernel_size=cfg.kernel_size,
        ).to(device)
        m._net.load_state_dict(payload["state_dict"])
        m._net.eval()
        return m
