"""Configurable feed-forward network + a hand-written training loop.

Shared by Step 2 (baseline) and Step 3 (tuning sweep).
"""
from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def set_seed(seed: int = C.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


@dataclass
class TrainConfig:
    """One experiment: architecture + optimisation + regularisation."""
    name: str = "baseline"
    hidden_sizes: tuple = (128, 64)
    dropout: float = 0.3
    batch_norm: bool = False
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    max_epochs: int = 100
    early_stopping: bool = True
    patience: int = 10
    optimizer: str = "adam"
    note: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["hidden_sizes"] = list(self.hidden_sizes)
        return d


class MLP(nn.Module):
    """Deep feed-forward binary classifier.

    Each hidden block is Linear -> [BatchNorm] -> ReLU -> [Dropout].
    The head emits a single logit (BCEWithLogitsLoss applies the sigmoid).
    """

    def __init__(self, in_features: int, hidden_sizes=(128, 64),
                 dropout: float = 0.3, batch_norm: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def architecture_table(model: MLP, in_features: int) -> list[dict]:
    """Layer-by-layer table: type, output units, activation, parameter count."""
    rows: list[dict] = []
    rows.append({"layer": "Input", "type": "-", "output_units": in_features,
                 "activation": "-", "params": 0})
    current = in_features
    idx = 0
    for module in model.net:
        if isinstance(module, nn.Linear):
            idx += 1
            p = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            is_head = module.out_features == 1
            rows.append({
                "layer": "Output (Linear)" if is_head else f"Hidden {idx} (Linear)",
                "type": f"Linear({module.in_features}, {module.out_features})",
                "output_units": module.out_features,
                "activation": "Sigmoid (via BCEWithLogitsLoss)" if is_head else "ReLU",
                "params": int(p),
            })
            current = module.out_features
        elif isinstance(module, nn.BatchNorm1d):
            rows.append({"layer": f"  BatchNorm1d({current})", "type": "BatchNorm1d",
                         "output_units": current, "activation": "-",
                         "params": int(sum(q.numel() for q in module.parameters()))})
        elif isinstance(module, nn.Dropout):
            rows.append({"layer": f"  Dropout(p={module.p})", "type": "Dropout",
                         "output_units": current, "activation": "-", "params": 0})
    return rows


@dataclass
class TrainResult:
    config: dict
    history: dict = field(default_factory=dict)
    best_epoch: int = 0
    epochs_run: int = 0
    train_time_sec: float = 0.0
    n_params: int = 0
    best_val_loss: float = float("inf")


def train_model(cfg: TrainConfig, X_train, y_train, X_val, y_val,
                device: str = "cpu", verbose: bool = True):
    """Explicit training loop: forward, loss, zero_grad, backward, step.

    No high-level fit() helper is used anywhere - this is the from-scratch loop
    required by Step 2.
    """
    set_seed(C.SEED)
    device_t = torch.device(device)

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32).to(device_t)
    yva = torch.tensor(y_val, dtype=torch.float32).to(device_t)

    loader = DataLoader(
        TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(C.SEED), drop_last=False,
    )

    model = MLP(Xtr.shape[1], cfg.hidden_sizes, cfg.dropout, cfg.batch_norm).to(device_t)
    criterion = nn.BCEWithLogitsLoss()
    if cfg.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                                    momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0

    t0 = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        # training pass
        model.train()
        run_loss, run_correct, run_n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device_t), yb.to(device_t)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * xb.size(0)
            run_correct += ((torch.sigmoid(logits) >= 0.5).float() == yb).sum().item()
            run_n += xb.size(0)
        tr_loss, tr_acc = run_loss / run_n, run_correct / run_n

        # validation pass
        model.eval()
        with torch.no_grad():
            v_logits = model(Xva)
            v_loss = criterion(v_logits, yva).item()
            v_acc = (((torch.sigmoid(v_logits) >= 0.5).float() == yva).float().mean().item())

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(v_acc)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d} | train {tr_loss:.4f}/{tr_acc:.4f} "
                  f"| val {v_loss:.4f}/{v_acc:.4f}")

        if v_loss < best_val_loss - 1e-5:
            best_val_loss, best_epoch, bad_epochs = v_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if cfg.early_stopping and bad_epochs >= cfg.patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    elapsed = time.perf_counter() - t0

    # Only put the best weights back if early stopping is on. Otherwise the
    # "no early stopping" run would still get the benefit of keeping the best
    # checkpoint, and the comparison would not be fair.
    if cfg.early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    result = TrainResult(
        config=cfg.as_dict(), history=history, best_epoch=best_epoch,
        epochs_run=len(history["train_loss"]), train_time_sec=round(elapsed, 3),
        n_params=count_parameters(model), best_val_loss=best_val_loss,
    )
    return model, result


@torch.no_grad()
def predict_proba(model: nn.Module, X, device: str = "cpu", batch_size: int = 4096) -> np.ndarray:
    model.eval()
    device_t = torch.device(device)
    Xt = torch.tensor(X, dtype=torch.float32)
    out = []
    for i in range(0, len(Xt), batch_size):
        out.append(torch.sigmoid(model(Xt[i:i + batch_size].to(device_t))).cpu().numpy())
    return np.concatenate(out) if out else np.array([])
