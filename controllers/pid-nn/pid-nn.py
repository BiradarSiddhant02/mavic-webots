import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from mavic import Mavic
from utils import clamp
from typing import Tuple
from controller import Robot  # type: ignore

# Existing parameters
STATE_DIM = 9
ACTION_DIM = 4

# Actor Network Constants
ACTOR_HIDDEN_DIM1 = 16
ACTOR_HIDDEN_DIM2 = 16

# Critic Network Constants
CRITIC_HIDDEN_DIM1 = 16
CRITIC_HIDDEN_DIM2 = 16

# DDPG Hyperparameters
BUFFER_SIZE = 100000  # Replay buffer size
BATCH_SIZE = 64  # Minibatch size
GAMMA = 0.99  # Discount factor
TAU = 0.005  # For soft update of target parameters
ACTOR_LR = 0.001  # Learning rate for actor
CRITIC_LR = 0.001  # Learning rate for critic
EXPLORATION_NOISE = 0.2  # Increased Gaussian noise added to action for exploration
NUM_EPISODES = 1000  # Number of episodes to train
MAX_STEPS = 1000  # Maximum number of steps per episode
UPDATE_EVERY = 1  # How often to update the network

# Target state (example, adjust as needed)
TARGET_STATE = np.array([3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Drone-specific constants (ensure these are defined elsewhere or add them here)
MAX_EPISODE_TIME = 80  # Maximum time for an episode (in seconds)
MAX_MOTOR_VELOCITY = 76  # Example maximum motor velocity
K_VERTICAL_THRUST = 68.5  # Example vertical offset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, ACTOR_HIDDEN_DIM1),
            nn.ReLU(),
            nn.Linear(ACTOR_HIDDEN_DIM1, ACTOR_HIDDEN_DIM2),
            nn.ReLU(),
            nn.Linear(ACTOR_HIDDEN_DIM2, ACTION_DIM),
            nn.Tanh()
        )

    def forward(self, state: torch.tensor) -> torch.tensor:
        return self.net(state) * MAX_MOTOR_VELOCITY

class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, CRITIC_HIDDEN_DIM1),
            nn.ReLU(),
            nn.Linear(CRITIC_HIDDEN_DIM1, CRITIC_HIDDEN_DIM2),
            nn.ReLU(),
            nn.Linear(CRITIC_HIDDEN_DIM2, 1),
        )

    def forward(self, state_action: torch.tensor) -> torch.tensor:
        return self.net(state_action)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def read_sensors(drone: Mavic) -> np.ndarray:
    imu = drone.get_imu_values()
    gps = drone.get_gps_values()
    gyro = drone.get_gyro_values()
    return np.array(imu + gps + gyro)

def normalize_state(state: np.ndarray) -> np.ndarray:
    return (state - np.mean(state)) / (np.std(state) + 1e-8)

def normalize_reward(reward: float) -> float:
    return (reward - np.mean(reward)) / (np.std(reward) + 1e-8)

def main():
    robot = Robot()
    drone = Mavic(robot)

    actor = ActorNet().to(device)
    actor_target = ActorNet().to(device)
    actor_target.load_state_dict(actor.state_dict())
    actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)

    critic = CriticNet().to(device)
    critic_target = CriticNet().to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    start_time = drone.get_time()  # Record the start time
    while drone.get_time() - start_time < 1.0:  # Wait until 1 second has passed
        drone.step_robot()  # Continue stepping the simulation

    episode = 0
    episode_reward = 0

    while episode < NUM_EPISODES:
        state = read_sensors(drone)
        episode_reward = 0
        
        # Initialize loss accumulators as tensors with gradients enabled
        total_critic_loss = torch.zeros(1, device=device, requires_grad=True)
        total_actor_loss = torch.zeros(1, device=device, requires_grad=True)
        
        while True:
            if drone.get_time() >= 5:
                print(f"Episode {episode}, Reward: {episode_reward}")
                drone.reset()
                state = read_sensors(drone)
                episode_reward = 0
                episode += 1
                break  # Exit the inner loop to start a new episode

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = actor(state_tensor).squeeze(0).cpu().detach().numpy()

            # Apply action to drone with exploration noise
            noise = np.random.normal(0, EXPLORATION_NOISE, size=ACTION_DIM)
            action += noise
            
            rotor_speeds = (
                clamp(
                    K_VERTICAL_THRUST + (float(action[0])),
                    -MAX_MOTOR_VELOCITY,
                    MAX_MOTOR_VELOCITY,
                ),
                clamp(
                    K_VERTICAL_THRUST + (float(action[1])),
                    -MAX_MOTOR_VELOCITY,
                    MAX_MOTOR_VELOCITY,
                ),
                clamp(
                    K_VERTICAL_THRUST + (float(action[2])),
                    -MAX_MOTOR_VELOCITY,
                    MAX_MOTOR_VELOCITY,
                ),
                clamp(
                    K_VERTICAL_THRUST + (float(action[3])),
                    -MAX_MOTOR_VELOCITY,
                    MAX_MOTOR_VELOCITY,
                ),
            )

            drone.set_rotor_speed(rotor_speeds)

            # Step the simulation
            drone.step_robot()

            # Get new state and calculate reward
            next_state = read_sensors(drone)

            reward = np.mean((next_state - TARGET_STATE) ** 2)  # Negative MSE as reward

            done = drone.get_time() >= MAX_EPISODE_TIME

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if len(replay_buffer) > BATCH_SIZE:
                # Sample a batch
                batch = replay_buffer.sample(BATCH_SIZE)
                (
                    state_batch,
                    action_batch,
                    reward_batch,
                    next_state_batch,
                    done_batch,
                ) = zip(*batch)

                state_batch = torch.FloatTensor(state_batch).to(device)
                action_batch = torch.FloatTensor(action_batch).to(device)
                reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
                next_state_batch = torch.FloatTensor(next_state_batch).to(device)
                done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(device)

                # Update Critic
                next_actions = actor_target(next_state_batch)
                next_state_actions = torch.cat([next_state_batch, next_actions], 1)
                next_q_values = critic_target(next_state_actions)

                q_targets = reward_batch + GAMMA * next_q_values * (1 - done_batch)
                q_values = critic(torch.cat([state_batch, action_batch], 1))
                critic_loss = nn.MSELoss()(q_values, q_targets.detach())
                total_critic_loss = total_critic_loss + critic_loss  # Accumulate critic loss as tensor

                # Update Actor
                actor_loss = -critic(torch.cat([state_batch, actor(state_batch)], 1)).mean()
                total_actor_loss = total_actor_loss + actor_loss  # Accumulate actor loss as tensor

        # Perform optimization steps after the episode
        if episode > 0:  # Avoid stepping the optimizers before the first episode
            with torch.autograd.set_detect_anomaly(True):
                critic_optimizer.zero_grad()
                total_critic_loss.backward()
                critic_optimizer.step()

                actor_optimizer.zero_grad()
                total_actor_loss.backward()
                actor_optimizer.step()

                # Soft update target networks
                soft_update(actor_target, actor, TAU)
                soft_update(critic_target, critic, TAU)

if __name__ == "__main__":
    main()


