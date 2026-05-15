from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Conv1DConfig:
    n_filters: int = 64
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 5, 7])
    dropout: float = 0.2
    learning_rate: float = 5e-4
    batch_size: int = 256
    max_epochs: int = 30
    patience: int = 4
    device: str = "auto"
    random_state: int = 42


class _Conv1DNet(nn.Module):
    def __init__(self, channels: int, n_filters: int, kernel_sizes: list[int], dropout: float) -> None:
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv1d(channels, n_filters, k, padding=k // 2) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(n_filters * len(kernel_sizes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len) for Conv1d
        x = x.transpose(1, 2)
        outs = [torch.relu(b(x)).mean(dim=-1) for b in self.branches]
        cat = torch.cat(outs, dim=-1)
        return self.head(self.drop(cat)).squeeze(-1)


def _device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return torch.device(device)


class Conv1DAlphaModel:
    def __init__(self, config: Conv1DConfig) -> None:
        self.config = config
        self._net: _Conv1DNet | None = None
        self._dev = _device(config.device)
        torch.manual_seed(config.random_state)

    def fit(
        self,
        x_train: NDArray[np.float32],
        y_train: NDArray[np.float32],
        w_train: NDArray[np.float32],
        x_val: NDArray[np.float32],
        y_val: NDArray[np.float32],
        w_val: NDArray[np.float32],
    ) -> None:
        channels = x_train.shape[-1]
        self._net = _Conv1DNet(channels, self.config.n_filters, list(self.config.kernel_sizes), self.config.dropout).to(
            self._dev
        )
        opt = torch.optim.Adam(self._net.parameters(), lr=self.config.learning_rate)
        train_ds = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(w_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        vx = torch.tensor(x_val, dtype=torch.float32).to(self._dev)
        vy = torch.tensor(y_val, dtype=torch.float32).to(self._dev)
        vw = torch.tensor(w_val, dtype=torch.float32).to(self._dev)
        best, stalls = float("inf"), 0
        for _ in range(self.config.max_epochs):
            self._net.train()
            for xb, yb, wb in train_loader:
                xb = xb.to(self._dev)
                yb = yb.to(self._dev)
                wb = wb.to(self._dev)
                opt.zero_grad()
                pred = self._net(xb)
                loss = (wb * (pred - yb) ** 2).mean()
                loss.backward()
                opt.step()
            self._net.eval()
            with torch.no_grad():
                vp = self._net(vx)
                vloss = float((vw * (vp - vy) ** 2).mean().item())
            if vloss < best - 1e-6:
                best = vloss
                stalls = 0
            else:
                stalls += 1
                if stalls >= self.config.patience:
                    break

    def predict(self, x: NDArray[np.float32]) -> NDArray[np.float64]:
        if self._net is None:
            raise RuntimeError("call fit() first")
        self._net.eval()
        with torch.no_grad():
            out = self._net(torch.tensor(x, dtype=torch.float32).to(self._dev))
        return out.detach().cpu().numpy().astype(np.float64)
