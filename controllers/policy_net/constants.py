import os


# --- Policy Network Parameters --- #
INPUT_DIM = 10
HIDDEN_DIMS = [1024, 1024]
OUTPUT_DIM = 8
LEARNING_RATE = 0.0015

REWARD_SCALE = 10

TRAIN_MODE = True

# --- PPO Parameters --- #
NUM_EPISODES = 1000
NUM_STEPS = 1000

DEVICE = "cpu"

DESIRED_STATE = [0.0, 0.0, 3.14, 1.0, 1.0, 1.5, 0.0, 0.0, 0.0]

GAMMA = 0.99
CLIP_EPSILON = 0.2
VALUE_LOSS_COEFF = 0.5
ENTROPY_BONUS_COEFF = 0.01


# --- PID and Drone Parameters --- #
Kp = 10
Ki = 0.25
Kd = 5

Kroll = 50
Kpitch = 30

MAX_ROTOR_SPEED = 576.0
MIN_THRUST = 68.5

ERROR_RESET = 200

INITIAL_STATE = [0.0, 0.0, 3.14, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]

# --- Buffer Parameters --- #
BATCH_SIZE = 64
