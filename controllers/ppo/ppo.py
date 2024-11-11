from actor_critic import Actor, Critic, set_seed
from buffer import Buffer
from constants import *

from mavic import Mavic
from controller import Robot  # type: ignore

import numpy as np
from typing import Tuple, List, Dict
import os
import torch
from collections import deque


# --- Set seed --- #
set_seed(69)

# --- Actor Network --- #
actor = Actor(
    input_dim=ACTOR_INPUT_DIM,
    hidden_dim=ACTOR_HIDDEN_DIM,
    output_dim=ACTOR_OUTPUT_DIM,
)

actor_optim = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)

# --- Critic Network --- #
critic = Critic(
    input_dim=CRITIC_INPUT_DIM,
    hidden_dim=CRITIC_HIDDEN_DIM,
    output_dim=CRITIC_OUTPUT_DIM,
)

critic_optim = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

# --- Load Actor and Critic Weights --- #
resume_training = False

actor_weights = os.listdir(ACTOR_SAVE_DIR)
if actor_weights and resume_training:
    actor.load(os.path.join(ACTOR_SAVE_DIR, actor_weights[-1]))

critic_weights = os.listdir(CRITIC_SAVE_DIR)
if critic_weights and resume_training:
    critic.load(os.path.join(CRITIC_SAVE_DIR, critic_weights[-1]))

# --- Experience Buffer --- #
buffer = Buffer(BATCH_SIZE)

# --- State Deque --- #
state_buffer = deque(maxlen=2)
state_buffer.append(INITIAL_STATE)

# --- Define Agent --- #
robot = Robot()
mavic = Mavic(robot)


def clamp(value: float, low: float, high: float) -> float:
    return max(min(value, high), low)


def get_state():
    imu_values = mavic.get_imu_values()
    gps_values = mavic.get_gps_values()
    gyro_values = mavic.get_gyro_values()
    return np.concatenate([imu_values, gps_values, gyro_values])


def altitude_PID(
    current: float, desired: float, integral: float, prev_error: float, dt: int
) -> Tuple[float, float, float]:
    error = desired - current
    integral += error * dt
    derivative = (error - prev_error) / dt

    control = (Kp * error) + (Ki * integral) + (Kd * derivative)

    prev_error = error

    return control, integral, prev_error


def execute_action(
    action: int,
    target_altitude: float,
    roll_disturbance: float,
    pitch_disturbance: float,
    yaw_disturbance: float,
    integral_error: float,
    previous_error: float,
) -> Tuple[float, float, float, float, float, float]:
    # --- Determine action effects --- #
    if action == 0:
        target_altitude += 0.05
    elif action == 1:
        target_altitude = max(0, target_altitude - 0.05)
    elif action in {2, 3}:
        roll_disturbance = 1 if action == 2 else -1
    elif action in {4, 5}:
        pitch_disturbance = 2 if action == 4 else -2
    elif action in {6, 7}:
        yaw_disturbance = 1.3 if action == 6 else -1.3
    else:
        raise ValueError(f"Invalid action: {action}. Expected values are 0 to 7.")

    # --- Get state values --- #
    roll, pitch, yaw, x, y, z, roll_rate, pitch_rate, yaw_rate = get_state()

    # --- Compute control inputs --- #
    roll_input = Kroll * clamp(roll, -1, 1) + roll_rate + roll_disturbance
    pitch_input = Kpitch * clamp(pitch, -1, 1) + pitch_rate + pitch_disturbance
    yaw_input = yaw_disturbance

    # --- Altitude control using PID --- #
    control_effort, integral_error, previous_error = altitude_PID(
        z, target_altitude, integral_error, previous_error, mavic.timestep
    )

    # --- Motor inputs --- #
    front_left_motor_input = clamp(
        MIN_THRUST + control_effort - roll_input + pitch_input - yaw_input,
        -MAX_ROTOR_SPEED,
        MAX_ROTOR_SPEED,
    )
    front_right_motor_input = clamp(
        MIN_THRUST + control_effort + roll_input + pitch_input + yaw_input,
        -MAX_ROTOR_SPEED,
        MAX_ROTOR_SPEED,
    )
    rear_left_motor_input = clamp(
        MIN_THRUST + control_effort - roll_input - pitch_input + yaw_input,
        -MAX_ROTOR_SPEED,
        MAX_ROTOR_SPEED,
    )
    rear_right_motor_input = clamp(
        MIN_THRUST + control_effort + roll_input - pitch_input - yaw_input,
        -MAX_ROTOR_SPEED,
        MAX_ROTOR_SPEED,
    )

    # --- Set motor speeds --- #
    mavic.set_rotor_speed(
        (
            front_left_motor_input,
            -front_right_motor_input,
            -rear_left_motor_input,
            rear_right_motor_input,
        )
    )

    return (
        target_altitude,
        roll_disturbance,
        pitch_disturbance,
        yaw_disturbance,
        integral_error,
        previous_error,
    )


def calculate_reward(
    state_buffer: deque,
) -> Tuple[float, bool]:
    state_error = np.linalg.norm(np.array(DESIRED_STATE) - np.array(state_buffer[1]))
    state_reward = 1 / (state_error + 1e-3)

    gps0 = state_buffer[0][3:6]
    gps1 = state_buffer[1][3:6]

    gps_desired = DESIRED_STATE[3:6]

    d_vec = np.array(gps_desired) - np.array(gps0)
    v_vec = np.array(gps1) - np.array(gps0)

    d_dot_v = np.dot(d_vec, v_vec)
    vector_seperation = d_dot_v / (np.linalg.norm(d_vec) * np.linalg.norm(v_vec))

    return float(state_reward * vector_seperation), state_error < 0.1


def learn(experiences: Dict) -> None:
    states = torch.tensor(experiences["states"], dtype=torch.float32).to(DEVICE)
    actions = torch.tensor(experiences["actions"]).to(DEVICE)
    rewards = torch.tensor(experiences["rewards"]).to(DEVICE)
    old_log_probs = torch.tensor(experiences["log_probs"]).to(DEVICE)
    values = torch.tensor(experiences["values"]).to(DEVICE)
    dones = torch.tensor(experiences["dones"]).float().to(DEVICE)

    # Discounted rewards and advantages
    returns = []
    G = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        G = reward + (GAMMA * G * (1 - done))
        returns.insert(0, G)
    returns = torch.tensor(returns).to(DEVICE)
    advantages = returns - values

    # Compute current log probabilities using the actor
    current_log_probs = actor(states).log_prob(actions)

    # Calculate the ratio of current to old probabilities
    ratio = torch.exp(current_log_probs - old_log_probs)

    # Clipped surrogate objective
    epsilon = CLIP_EPSILON  # PPO clipping parameter
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    policy_loss = -torch.min(surrogate1, surrogate2).mean()

    # Value loss (mean squared error between returns and predicted values)
    value_loss = torch.nn.functional.mse_loss(values, returns)

    # Entropy bonus for exploration (encourages exploration by making the policy less deterministic)
    entropy_bonus = actor(states).entropy().mean()

    # Total loss with weighting factors for value and entropy loss terms
    c1 = VALUE_LOSS_COEFF  # Weighting factor for value loss
    c2 = ENTROPY_BONUS_COEFF  # Weighting factor for entropy bonus
    total_loss = policy_loss + c1 * value_loss - c2 * entropy_bonus

    # Backpropagate the loss
    actor_optim.zero_grad()
    critic_optim.zero_grad()
    total_loss.backward()
    actor_optim.step()
    critic_optim.step()


def main() -> None:

    integral_error = 0
    previous_error = 0

    target_altitude = 0
    roll_disturbance = 0
    pitch_disturbance = 0
    yaw_disturbance = 0

    for episode in range(NUM_EPISODES):
        mavic.reset()
        
        state = get_state()

        integral_error = 0
        previous_error = 0

        episode_reward = 0

        experiences = {
            "states": [],
            "actions": [],
            "rewards": [],
            "log_probs": [],
            "values": [],
            "dones": [],
        }

        for step in range(NUM_STEPS):
            ## Get action
            action_dist = actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE))
            state_buffer.append(state)

            ## Choose action from distribution
            action = action_dist.sample()

            ## Calculate log probability for the action
            action_log_prob = action_dist.log_prob(action)

            ## Apply Action
            (
                target_altitude,
                roll_disturbance,
                pitch_disturbance,
                yaw_disturbance,
                integral_error,
                previous_error,
            ) = execute_action(
                float(action),
                target_altitude,
                roll_disturbance,
                pitch_disturbance,
                yaw_disturbance,
                integral_error,
                previous_error,
            )

            ## Calculate Value
            value = critic(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE))

            ## Calculate Reward and Terminal state
            reward, done = calculate_reward(state_buffer)
            episode += reward

            experiences["states"].append(state)
            experiences["actions"].append(action)
            experiences["rewards"].append(reward)
            experiences["log_probs"].append(action_log_prob)
            experiences["values"].append(value)
            experiences["dones"].append(done)
            
            mavic.step_robot()

        learn(experiences)

main()