import numpy
from typing import List, Tuple
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size: int, input_shape: int, n_actions: int) -> None:
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = numpy.zeros(
            (self.mem_size, *input_shape), dtype=numpy.float32
        )
        self.new_state_memory = numpy.zeros(
            (self.mem_size, *input_shape), dtype=numpy.float32
        )
        self.action_memory = numpy.zeros(
            (self.mem_size, n_actions), dtype=numpy.float32
        )
        self.reward_memory = numpy.zeros(self.mem_size, dtype=numpy.float32)
        self.terminal_memory = numpy.zeros(self.mem_size, dtype=numpy.bool)

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        state_: np.ndarray,
        done: bool,
    ) -> None:
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)
        state = self.state_memory[batch]
        action = self.action_memory[batch]
        reward = self.reward_memory[batch]
        new_state = self.new_state_memory[batch]
        done = self.terminal_memory[batch]

        return state, action, reward, new_state, done
    
    
