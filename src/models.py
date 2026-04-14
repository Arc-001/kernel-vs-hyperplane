"""Phase 2 — MLP and RBF classifier module definitions."""

import math

import numpy as np
import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, d: int, hidden_layers: tuple[int, ...], k: int, activation: str = "relu"):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU}[activation]
        layers: list[nn.Module] = []
        prev = d
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            prev = h
        layers.append(nn.Linear(prev, k))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RBFClassifier(nn.Module):
    def __init__(self, d: int, n_centers: int, k: int, sigma_learnable: bool = True):
        super().__init__()
        self.centers = nn.Parameter(torch.zeros(n_centers, d))
        if sigma_learnable:
            self.log_sigma = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("log_sigma", torch.tensor(0.0))
        self.output = nn.Linear(n_centers, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(self.log_sigma)
        dists = torch.cdist(x, self.centers)  # (batch, n_centers)
        phi = torch.exp(-dists.pow(2) / (2 * sigma.pow(2)))
        return self.output(phi)

    def load_centers(self, centers_np: np.ndarray, sigma_val: float):
        """Load KMeans centers and initial sigma. Called by Phase 3A."""
        with torch.no_grad():
            self.centers.copy_(torch.tensor(centers_np, dtype=torch.float32))
            self.log_sigma.copy_(torch.tensor(math.log(sigma_val), dtype=torch.float32))


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
