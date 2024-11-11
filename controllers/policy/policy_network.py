import torch
import os
from typing import List


class PolicyNetwork(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        save_dir: str,
    ) -> None:
        super(PolicyNetwork, self).__init__()

        self.input_layer = torch.nn.Linear(input_dim, hidden_dims[0])

        self.middle_layers = torch.nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.middle_layers.append(
                torch.nn.Linear(hidden_dims[i - 1], hidden_dims[i])
            )

        self.output_layer = torch.nn.Linear(hidden_dims[-1], output_dim)

        self.save_dir = save_dir

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        for layer in self.middle_layers:
            x = torch.relu(layer(x))
        x = torch.nn.functional.softmax(self.output_layer(x), dim=-1)  # Fixed this line
        return x

    def save(self, name) -> None:
        torch.save(self.state_dict(), os.path.join(self.save_dir, name))

    def load(self, name) -> None:
        torch.load_state_dict(torch.load(os.path.join(self.save_dir, name)))
