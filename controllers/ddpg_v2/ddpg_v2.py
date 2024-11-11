## Webots imports
from mavic import Mavic
from controller import Robot  # type: ignore

## python imports
import torch
import numpy as np
from loguru import logger
from datetime import datetime
import os
from collections import deque

## local imports
from actor_critic import Actor, Critic

robot = Robot()
mavic = Mavic(robot)

# Actor-Critic Hyperparameters
state_dim = 9
action_dim = 4
actor_hidden_dim1 = 512
actor_hidden_dim2 = 512
critic_hidden_dim1 = 512
critic_hidden_dim2 = 512
actor_lr = 0.0005
critic_lr = 0.005
gamma = 0.99
tau = 0.005

# Initialize Actor and Critic Networks
actor_local = Actor(state_dim, actor_hidden_dim1, actor_hidden_dim2, action_dim)
critic_local = Critic(state_dim + action_dim, critic_hidden_dim1, critic_hidden_dim2, 1)

actor_target = Actor(state_dim, actor_hidden_dim1, actor_hidden_dim2, action_dim)
actor_target.load_state_dict(actor_local.state_dict())

critic_target = Critic(
    state_dim + action_dim, critic_hidden_dim1, critic_hidden_dim2, 1
)
critic_target.load_state_dict(critic_local.state_dict())

# Optimizers
actor_optimizer = torch.optim.Adam(actor_local.parameters(), lr=actor_lr)
critic_optimizer = torch.optim.Adam(critic_local.parameters(), lr=critic_lr)

# Training Hyperparameters
num_episodes = 1000
num_steps = 150
noise_std_dev = 0.3
desired_state = np.array([0, 0, 3.14, 0, 0, 5.0, 0, 0, 0])

# State Transition Buffer
state_buffer = deque(maxlen=2)
state_buffer.append(np.array([0, 0, 3.14, 0, 0, 0.0, 0, 0, 0]))
state_buffer.append(np.array([0, 0, 3.14, 0, 0, 0.0, 0, 0, 0])) 

def main():
    current_episode = 0
    total_steps = 0

    while current_episode < num_episodes:
        mavic.reset()
        state = get_state()
        state_buffer.pop()
        state_buffer.pop()
        state_buffer.append(state)

        # Reset buffers
        episode_reward = 0
        episode_experiences = list()
        episode_state_errors = list()
        episode_control_efforts = list()
        episode_rewards = list()

        # Reset variables
        prev_action = np.zeros(action_dim)

        for step in range(num_steps):
            # Sample state
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # f-prop state through actor network
            action = actor_local(state_tensor)
            action_np = action.detach().numpy().flatten()

            # add noise to action
            noise = np.random.normal(0, noise_std_dev, size=action_dim)
            action_np = np.clip(action_np + noise, 0, 576.0)

            # apply action to environment
            action_np[1], action_np[2] = -action_np[1], -action_np[2]
            mavic.set_rotor_speed(action_np)

            # Wait for environment to update
            for _ in range(32 // mavic.timestep):
                mavic.step_robot()

            # get new state
            next_state = get_state()

            # calculate reward
            reward = calculate_reward(state_buffer[0], state_buffer[1], action_np, prev_action)

            # store experiences
            state_error = float(np.linalg.norm(next_state - desired_state))
            control_effort = float(np.sum(np.square(action_np - prev_action)))
            prev_action = action_np

            episode_experiences.append((state, action_np, reward, next_state))
            episode_state_errors.append(state_error)
            episode_control_efforts.append(control_effort)
            episode_rewards.append(reward)

            episode_reward += reward
            total_steps += 1
            mavic.step_robot()

        # update actor and critic networks
        actor_loss, critic_loss = update_networks(episode_experiences)
        logger.info(
            f"Episode {current_episode} | "
            f"Reward: {episode_reward:.2f} | "
            f"Actor Loss: {actor_loss:.4f} | "
            f"Critic Loss: {critic_loss:.4f} | "
            f"Avg State Error: {np.mean(episode_state_errors):.4f}"
        )

        # update episode metrics
        current_episode += 1


def get_state():
    imu_values = mavic.get_imu_values()
    gps_values = mavic.get_gps_values()
    gyro_values = mavic.get_gyro_values()
    return np.concatenate([imu_values, gps_values, gyro_values])


def calculate_reward(state, next_state, action, prev_action):
    # Reward for getting closer to the desired state
    state_error = np.linalg.norm(next_state - desired_state)
    state_error_reward = 1 / (
        1 + state_error
    )  # Positive reward that increases as state_error decreases

    # Penalty for aggressive control effort
    control_effort_penalty = np.sum(np.abs(action - prev_action))

    # Reward for consistent and meaningful state change
    state_diff = np.sum(np.abs(next_state - state) / 0.08)
    state_diff_reward = 0.1 * state_diff

    # Combine the components
    return state_error_reward - 0.02 * control_effort_penalty + state_diff_reward


def update_networks(episode_experiences):
    states, actions, rewards, next_states = zip(*episode_experiences)
    states, actions = torch.tensor(states, dtype=torch.float32), torch.tensor(
        actions, dtype=torch.float32
    )
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)

    with torch.no_grad():
        next_actions = actor_target(next_states)
        next_q_values = critic_target(next_states, next_actions)
        target_q_values = rewards + gamma * next_q_values

    current_q_values = critic_local(states, actions)
    critic_loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actor_loss = -critic_local(states, actor_local(states)).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    soft_update(actor_target, actor_local, tau)
    soft_update(critic_target, critic_local, tau)

    return actor_loss.item(), critic_loss.item()


def soft_update(target_net, local_net, tau):
    for target_param, local_param in zip(
        target_net.parameters(), local_net.parameters()
    ):
        target_param.data.copy_(
            tau * local_param.data + (1.0 - tau) * target_param.data
        )


main()
