import os


# --- Actor Network Parameters --- #
ACTOR_INPUT_DIM = 9
ACTOR_HIDDEN_DIM = (512, 512)
ACTOR_OUTPUT_DIM = 8

ACTOR_LR = 0.001

ACTOR_SAVE_DIR = "actor_weights"
os.makedirs(ACTOR_SAVE_DIR, exist_ok=True)


# --- Critic Network Parameters --- #
CRITIC_INPUT_DIM = 9
CRITIC_HIDDEN_DIM = (512, 512)
CRITIC_OUTPUT_DIM = 1

CRITIC_LR = 0.01
CRITIC_SAVE_DIR = "critic_weights"
os.makedirs(CRITIC_SAVE_DIR, exist_ok=True)


# --- PPO Parameters --- #
NUM_EPISODES = 1000
NUM_STEPS = 100

DEVICE = "cpu"

DESIRED_STATE = [0.0, 0.0, 3.14, 1.0, 1.0, 1.5, 0.0, 0.0, 0.0]

GAMMA = 0.99
CLIP_EPSILON = 0.2
VALUE_LOSS_COEFF = 0.5
ENTROPY_BONUS_COEFF = 0.01


# --- PID and Drone Parameters --- #
Kp = 10.0
Ki = 0.25
Kd = 5.0

Kroll = 50
Kpitch = 30

MAX_ROTOR_SPEED = 576.0
MIN_THRUST = 68.5

INITIAL_STATE = [0.0, 0.0, 3.14, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]

# --- Buffer Parameters --- #
BATCH_SIZE = 64
