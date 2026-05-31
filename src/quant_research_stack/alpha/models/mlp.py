from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_TORCH_THREADS_CONFIGURED = False


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


def _configure_torch_cpu_threads() -> None:
    global _TORCH_THREADS_CONFIGURED
    if _TORCH_THREADS_CONFIGURED:
        return
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only permits inter-op thread configuration before parallel work
        # starts. In that case the intra-op cap above is still the important guard.
        pass
    _TORCH_THREADS_CONFIGURED = True


class MLPAlphaModel:
    def __init__(self, config: MLPConfig) -> None:
        self.config = config
        _configure_torch_cpu_threads()
        self._net: _Net | None = None
        self._device = _resolve_device(config.device)
        self._scaler: StandardScaler | None = None
        self._input_dim: int | None = None
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
        self._input_dim = x_train.shape[1]

        # Fit scaler on training data only; apply to both train and val.
        self._scaler = StandardScaler()
        x_train_scaled = self._scaler.fit_transform(x_train.astype(np.float64)).astype(np.float32)
        x_val_scaled = self._scaler.transform(x_val.astype(np.float64)).astype(np.float32)

        self._net = _Net(self._input_dim, list(self.config.hidden_dims), self.config.dropout).to(self._device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.config.learning_rate)
        train_ds = TensorDataset(
            torch.tensor(x_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(w_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        val_ds = TensorDataset(
            torch.tensor(x_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
            torch.tensor(w_val, dtype=torch.float32),
        )
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)
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
            loss_sum = 0.0
            n_obs = 0
            with torch.no_grad():
                for xb, yb, wb in val_loader:
                    xb = xb.to(self._device)
                    yb = yb.to(self._device)
                    wb = wb.to(self._device)
                    vp = self._net(xb)
                    loss_sum += float((wb * (vp - yb) ** 2).sum().item())
                    n_obs += int(yb.numel())
            vloss = loss_sum / max(n_obs, 1)
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
        if self._scaler is None:
            raise RuntimeError("call fit() first")
        x_scaled = self._scaler.transform(np.asarray(x, dtype=np.float64)).astype(np.float32)
        self._net.eval()
        pred_loader = DataLoader(
            TensorDataset(torch.tensor(x_scaled, dtype=torch.float32)),
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for (xb,) in pred_loader:
                out = self._net(xb.to(self._device))
                outs.append(out.detach().cpu().numpy().astype(np.float64))
        return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float64)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._net is None or self._scaler is None or self._input_dim is None:
            raise RuntimeError("cannot save un-fitted MLPAlphaModel")
        payload = {
            "state_dict": self._net.state_dict(),
            "arch": {
                "input_dim": int(self._input_dim),
                "hidden_dims": list(self.config.hidden_dims),
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
    def load(cls, path: Path) -> MLPAlphaModel:
        path = Path(path)
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        inst = cls(MLPConfig(**payload["config"]))
        inst._input_dim = payload["arch"]["input_dim"]
        inst._net = _Net(
            in_dim=payload["arch"]["input_dim"],
            hidden=payload["arch"]["hidden_dims"],
            dropout=payload["arch"]["dropout"],
        )
        inst._net.load_state_dict(payload["state_dict"])
        inst._net.eval()
        inst._device = torch.device("cpu")
        inst._scaler = StandardScaler()
        inst._scaler.mean_ = np.asarray(payload["scaler"]["mean"], dtype=np.float64)
        inst._scaler.scale_ = np.asarray(payload["scaler"]["scale"], dtype=np.float64)
        inst._scaler.var_ = inst._scaler.scale_ ** 2
        inst._scaler.n_features_in_ = inst._scaler.mean_.size
        return inst
