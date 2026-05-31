"""Compact PyTorch MLP for the S1-EQ stack (spec §4.4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


@dataclass(frozen=True)
class MLPEqConfig:
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.3
    learning_rate: float = 1.0e-3
    batch_size: int = 1024
    max_epochs: int = 50
    seed: int = 42


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Net(nn.Module):
    def __init__(self, in_features: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPEqModel:
    def __init__(self, config: MLPEqConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self._net: _Net | None = None
        self._n_features: int | None = None

    def fit(self, *, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        device = _device()
        self._n_features = x.shape[1]
        self._net = _Net(
            in_features=self._n_features,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        bs = self.config.batch_size
        for _ in range(self.config.max_epochs):
            perm = torch.randperm(x_t.shape[0], device=device)
            for i in range(0, x_t.shape[0], bs):
                idx = perm[i : i + bs]
                optimizer.zero_grad()
                out = self._net(x_t[idx])
                loss = loss_fn(out, y_t[idx])
                loss.backward()
                optimizer.step()

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._net is None:
            raise RuntimeError("model not fit")
        device = next(self._net.parameters()).device
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        was_training = self._net.training
        self._net.eval()
        try:
            with torch.no_grad():
                out = self._net(x_t).cpu().numpy()
        finally:
            if was_training:
                self._net.train()
        return out.astype(np.float64)

    def save(self, path: Path) -> None:
        if self._net is None:
            raise RuntimeError("model not fit")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "config": asdict(self.config),
                "n_features": self._n_features,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> MLPEqModel:
        device = _device()
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg = MLPEqConfig(**payload["config"])
        m = cls(cfg)
        m._n_features = int(payload["n_features"])
        m._net = _Net(
            in_features=m._n_features,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        ).to(device)
        m._net.load_state_dict(payload["state_dict"])
        m._net.eval()
        return m
