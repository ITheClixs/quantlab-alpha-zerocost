from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class MLPConfig:
    hidden_dims: list[int]
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 1024
    max_epochs: int = 50
    patience: int = 5
    mixed_precision: bool = True
    device: str = "auto"
    random_state: int = 42


class _Net(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class MLPAlphaModel:
    def __init__(self, config: MLPConfig) -> None:
        self.config = config
        self._net: _Net | None = None
        self._device = _resolve_device(config.device)
        torch.manual_seed(config.random_state)

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        w_train: NDArray[np.float64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        w_val: NDArray[np.float64],
    ) -> None:
        in_dim = x_train.shape[1]
        self._net = _Net(in_dim, list(self.config.hidden_dims), self.config.dropout).to(self._device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.config.learning_rate)
        train_ds = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(w_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        val_x = torch.tensor(x_val, dtype=torch.float32).to(self._device)
        val_y = torch.tensor(y_val, dtype=torch.float32).to(self._device)
        val_w = torch.tensor(w_val, dtype=torch.float32).to(self._device)
        best_val = float("inf")
        stalls = 0
        for _epoch in range(self.config.max_epochs):
            self._net.train()
            for xb, yb, wb in train_loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                wb = wb.to(self._device)
                opt.zero_grad()
                pred = self._net(xb)
                loss = (wb * (pred - yb) ** 2).mean()
                loss.backward()
                opt.step()
            self._net.eval()
            with torch.no_grad():
                vp = self._net(val_x)
                vloss = float((val_w * (vp - val_y) ** 2).mean().item())
            if vloss < best_val - 1e-6:
                best_val = vloss
                stalls = 0
            else:
                stalls += 1
                if stalls >= self.config.patience:
                    break

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._net is None:
            raise RuntimeError("call fit() first")
        self._net.eval()
        with torch.no_grad():
            out = self._net(torch.tensor(x, dtype=torch.float32).to(self._device))
        return out.detach().cpu().numpy().astype(np.float64)
