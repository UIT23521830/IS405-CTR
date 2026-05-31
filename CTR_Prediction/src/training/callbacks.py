from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int = 2, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        improved = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

