import torch
from typing import Tuple


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        action_dim: int,
    ):
        self.memory_size = buffer_size
        self.memory_counter = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_memory = torch.zeros((self.memory_size, state_dim))
        self.action_memory = torch.zeros((self.memory_size, action_dim))
        self.new_state_memory = torch.zeros((self.memory_size, state_dim))
        self.reward_memory = torch.zeros(self.memory_size)

    def store_transition(
        self,
        state: torch.Tensor,
        reward: float,
        new_state: torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.memory_counter += 1

    def sample_buffer(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_memory = min(self.memory_counter, self.memory_size)
        batch = torch.randint(0, max_memory, (batch_size,))
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        action = self.action_memory[batch]
        return states, action, rewards, new_states

    def __len__(self) -> int:
        return self.memory_counter

    def reset(self) -> None:
        self.memory_counter = 0
        self.state_memory = torch.zeros((self.memory_size, self.state_dim))
        self.new_state_memory = torch.zeros((self.memory_size, self.state_dim))
        self.reward_memory = torch.zeros(self.memory_size)
        self.action_memory = torch.zeros((self.memory_size, self.action_dim))
