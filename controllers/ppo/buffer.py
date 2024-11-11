import torch
from typing import Tuple, List
import random


class Buffer:
    def __init__(
        self,
        batch_size: int,
    ) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_actions = []
        self.values = []
        self.terminals = []
        self.memory_counter = 0
        self.batch_size = batch_size

    def store(
        self,
        data: Tuple[torch.Tensor, int, float, float, float, bool],
    ) -> None:
        # Unpack the data tuple
        (
            state,
            action,
            reward,
            log_action,
            value,
            terminal,
        ) = data
        
        idx = self.memory_counter

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.log_actions[idx] = log_action
        self.values[idx] = value
        self.terminals[idx] = terminal
        
        self.memory_counter += 1

    def get_batch(
        self,
    ) -> Tuple[
        List[torch.Tensor],
        List[int],
        List[float],
        List[float],
        List[float],
        List[bool],
    ]:
        # Get random indices for the batch
        indices = random.sample(range(self.memory_counter), self.batch_size)

        # Retrieve corresponding data for each index
        batch_states = [self.states[i] for i in indices]
        batch_actions = [self.actions[i] for i in indices]
        batch_rewards = [self.rewards[i] for i in indices]
        batch_log_actions = [self.log_actions[i] for i in indices]
        batch_values = [self.values[i] for i in indices]
        batch_terminals = [self.terminals[i] for i in indices]

        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_log_actions,
            batch_values,
            batch_terminals,
        )
