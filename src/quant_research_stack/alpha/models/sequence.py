from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
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


def _scale_3d(scaler: StandardScaler, x: NDArray[np.float64]) -> NDArray[np.float32]:
    """Apply a per-channel StandardScaler to a 3-D array (batch, seq_len, channels)."""
    batch, seq_len, channels = x.shape
    flat = x.reshape(-1, channels).astype(np.float64)
    scaled = scaler.transform(flat)
    return scaled.reshape(batch, seq_len, channels).astype(np.float32)


class Conv1DAlphaModel:
    def __init__(self, config: Conv1DConfig) -> None:
        self.config = config
        self._net: _Conv1DNet | None = None
        self._dev = _device(config.device)
        self._scaler: StandardScaler | None = None
        self._channels: int | None = None
        self._seq_len: int | None = None
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
        x_train = np.asarray(x_train, dtype=np.float64)
        x_val = np.asarray(x_val, dtype=np.float64)

        self._seq_len = x_train.shape[-2]
        self._channels = x_train.shape[-1]

        # Fit scaler per-channel on training data only; apply to train and val.
        # Reshape (batch, seq_len, channels) -> (batch*seq_len, channels) for sklearn.
        flat_train = x_train.reshape(-1, self._channels)
        self._scaler = StandardScaler()
        self._scaler.fit(flat_train)

        x_train_scaled = _scale_3d(self._scaler, x_train)
        x_val_scaled = _scale_3d(self._scaler, x_val)

        self._net = _Conv1DNet(
            self._channels, self.config.n_filters, list(self.config.kernel_sizes), self.config.dropout
        ).to(self._dev)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.config.learning_rate)
        train_ds = TensorDataset(
            torch.tensor(x_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(w_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        vx = torch.tensor(x_val_scaled, dtype=torch.float32).to(self._dev)
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

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._net is None:
            raise RuntimeError("call fit() first")
        if self._scaler is None:
            raise RuntimeError("call fit() first")
        x_scaled = _scale_3d(self._scaler, np.asarray(x, dtype=np.float64))
        self._net.eval()
        with torch.no_grad():
            out = self._net(torch.tensor(x_scaled, dtype=torch.float32).to(self._dev))
        return out.detach().cpu().numpy().astype(np.float64)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._net is None or self._scaler is None or self._channels is None:
            raise RuntimeError("cannot save un-fitted Conv1DAlphaModel")
        payload = {
            "state_dict": self._net.state_dict(),
            "arch": {
                "channels": int(self._channels),
                "n_filters": int(self.config.n_filters),
                "kernel_sizes": list(self.config.kernel_sizes),
                "dropout": float(self.config.dropout),
            },
            "scaler": {
                "mean": np.asarray(self._scaler.mean_, dtype=np.float64).tolist(),
                "scale": np.asarray(self._scaler.scale_, dtype=np.float64).tolist(),
            },
            "config": asdict(self.config),
        }
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path: Path) -> "Conv1DAlphaModel":
        path = Path(path)
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        inst = cls(Conv1DConfig(**payload["config"]))
        arch = payload["arch"]
        inst._channels = arch["channels"]
        inst._net = _Conv1DNet(
            channels=arch["channels"],
            n_filters=arch["n_filters"],
            kernel_sizes=arch["kernel_sizes"],
            dropout=arch["dropout"],
        )
        inst._net.load_state_dict(payload["state_dict"])
        inst._net.eval()
        inst._dev = torch.device("cpu")
        inst._scaler = StandardScaler()
        inst._scaler.mean_ = np.asarray(payload["scaler"]["mean"], dtype=np.float64)
        inst._scaler.scale_ = np.asarray(payload["scaler"]["scale"], dtype=np.float64)
        inst._scaler.var_ = inst._scaler.scale_ ** 2
        inst._scaler.n_features_in_ = inst._scaler.mean_.size
        return inst
