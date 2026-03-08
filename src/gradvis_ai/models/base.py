"""Base classes and shared helpers for gradient-descent models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class TrainingHistory:
    """Keeps full optimization traces for educational visualizations."""

    losses: List[float] = field(default_factory=list)
    weights: List[np.ndarray] = field(default_factory=list)
    biases: List[float] = field(default_factory=list)

    def append(self, loss: float, weight: np.ndarray, bias: float) -> None:
        self.losses.append(float(loss))
        self.weights.append(weight.copy())
        self.biases.append(float(bias))

    def as_dict(self) -> Dict[str, List]:
        return {"losses": self.losses, "weights": self.weights, "biases": self.biases}
