import numpy as np
import random

class Buffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer.pop(0)
            self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*sample))
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)