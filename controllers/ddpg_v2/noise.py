import numpy as np
from typing import Tuple

class OUNoise:
    def __init__(
        self,
        action_dim: int,
        mu: float = 0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ) -> None:
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
        
    def reset(self) -> None:
        self.state = np.ones(self.action_dim) * self.mu
        
    def __call__(self) -> np.ndarray:
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def __repr__(self) -> str:
        return f"OUNoise(mu={self.mu}, theta={self.theta}, sigma={self.sigma})"