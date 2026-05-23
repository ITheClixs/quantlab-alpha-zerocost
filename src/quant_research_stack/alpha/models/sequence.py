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
        self._n_features: int | None = None
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

        # Accept 2D input (n, n_features); reshape to 3D (n, n_features, 1) for Conv1d.
        assert x_train.ndim == 2, f"Expected 2D input (n, n_features), got shape {x_train.shape}"
        assert x_val.ndim == 2, f"Expected 2D input (n, n_features), got shape {x_val.shape}"

        self._n_features = x_train.shape[1]

        # Fit scaler per-feature on 2D training data; scale train and val.
        self._scaler = StandardScaler()
        self._scaler.fit(x_train)

        x_train_scaled = self._scaler.transform(x_train).astype(np.float32)
        x_val_scaled = self._scaler.transform(x_val).astype(np.float32)

        # Reshape 2D (n, n_features) -> 3D (n, n_features, 1) for Conv1d.
        x_train_3d = x_train_scaled[:, :, np.newaxis]
        x_val_3d = x_val_scaled[:, :, np.newaxis]

        self._net = _Conv1DNet(
            channels=1,
            n_filters=self.config.n_filters,
            kernel_sizes=list(self.config.kernel_sizes),
            dropout=self.config.dropout
        ).to(self._dev)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.config.learning_rate)
        train_ds = TensorDataset(
            torch.tensor(x_train_3d, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(w_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        vx = torch.tensor(x_val_3d, dtype=torch.float32).to(self._dev)
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
        if self._n_features is None:
            raise RuntimeError("call fit() first")

        x = np.asarray(x, dtype=np.float64)
        assert x.ndim == 2, f"Expected 2D input (n, n_features), got shape {x.shape}"
        assert x.shape[1] == self._n_features, (
            f"Expected {self._n_features} features, got {x.shape[1]}"
        )

        # Scale and reshape 2D (n, n_features) -> 3D (n, n_features, 1).
        x_scaled = self._scaler.transform(x).astype(np.float32)
        x_3d = x_scaled[:, :, np.newaxis]

        self._net.eval()
        with torch.no_grad():
            out = self._net(torch.tensor(x_3d, dtype=torch.float32).to(self._dev))
        return out.detach().cpu().numpy().astype(np.float64)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._net is None or self._scaler is None or self._n_features is None:
            raise RuntimeError("cannot save un-fitted Conv1DAlphaModel")
        payload = {
            "state_dict": self._net.state_dict(),
            "arch": {
                "n_features": int(self._n_features),
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
    def load(cls, path: Path) -> Conv1DAlphaModel:
        path = Path(path)
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        inst = cls(Conv1DConfig(**payload["config"]))
        arch = payload["arch"]
        inst._n_features = arch["n_features"]
        # Conv1d always has 1 channel after 2D->3D reshape.
        inst._net = _Conv1DNet(
            channels=1,
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
