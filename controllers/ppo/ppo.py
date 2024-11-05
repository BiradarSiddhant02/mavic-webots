from actor_critic import Actor, Critic

from mavic import Mavic
from controller import Robot    # type: ignore

import numpy as np
import os
import torch

# --- Actor Network Parameters --- #
ACTOR_INPUT_DIM = 9
ACTOR_HIDDEN_DIM = (512, 512)
ACTOR_OUTPUT_DIM = 8

ACTOR_LR = 0.001

ACTOR_SAVE_DIR = 'actor_weights'
os.makedirs(ACTOR_SAVE_DIR, exist_ok=True)

# --- Critic Network Parameters --- #
CRITIC_INPUT_DIM = 17
CRITIC_HIDDEN_DIM = (512, 512)
CRITIC_OUTPUT_DIM = 1

CRITIC_LR = 0.01
CRITIC_SAVE_DIR = 'critic_weights'
os.makedirs(CRITIC_SAVE_DIR, exist_ok=True)

# --- Actor Network --- #