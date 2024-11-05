import torch
import torch.nn as nn
from typing import Tuple

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Actor(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: Tuple[int, int], output_dim: int
    ) -> None:
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], output_dim),
            nn.Softmax(dim=-1),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

    def _initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=0, nonlinearity="relu")
                nn.init.zeros_(layer.bias)


class Critic(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: Tuple[int, int], output_dim: int
    ) -> None:
        super(Critic, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], output_dim),
        )

        self._initialize_weights()

    def forward(
        self, state_vec: torch.Tensor, action_vec: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat((state_vec, action_vec), dim=1))

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

    def _initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=0, nonlinearity="relu")
                nn.init.zeros_(layer.bias)