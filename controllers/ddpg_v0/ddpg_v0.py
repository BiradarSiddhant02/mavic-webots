from actor_critic import Actor, Critic
from mavic import Mavic
from controller import Robot  # type: ignore
import torch
import numpy as np
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

robot = Robot()
mavic = Mavic(robot)

# Actor-Critic Hyperparameters
state_dim = 9
action_dim = 4
actor_hidden_dim1 = 512
actor_hidden_dim2 = 512
critic_hidden_dim1 = 512
critic_hidden_dim2 = 512
actor_lr = 0.00022
critic_lr = 0.0022
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
desired_state = np.array([0, 0, 3.0, 0, 0, 10.0, 0, 0, 0])

# TensorBoard Setup
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("runs", "drone_training", current_time)
writer = SummaryWriter(log_dir=log_dir)
logger.info(f"TensorBoard logs will be saved to: {log_dir}")


def main():
    current_episode = 0
    total_steps = 0  # Global step counter

    while current_episode < num_episodes:
        mavic.reset()
        state = get_state()
        episode_reward = 0
        episode_experiences = []

        # Lists to store episode metrics
        episode_state_errors = []
        episode_control_efforts = []
        episode_rewards = []

        for step in range(num_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = actor_local(state_tensor).detach().numpy().flatten()
            noise = np.random.normal(0, noise_std_dev, size=action.shape)
            action = np.clip(action + noise, 0, 576)
            action[1], action[2] = -action[1], -action[2]

            mavic.set_rotor_speed(action)
            for _ in range(30 // mavic.timestep):
                mavic.step_robot()

            next_state = get_state()
            state_error = float(np.linalg.norm(next_state - desired_state))
            control_effort = float(np.sum(np.square(next_state - state)))
            reward = calculate_reward(state, next_state)

            episode_experiences.append((state, action, reward, next_state))
            state = next_state
            episode_reward += reward

            # Store metrics for this step
            episode_state_errors.append(state_error)
            episode_control_efforts.append(control_effort)
            episode_rewards.append(reward)

            # Log immediate step metrics
            writer.add_scalar("Step/State Error", state_error, total_steps)
            writer.add_scalar("Step/Control Effort", control_effort, total_steps)
            writer.add_scalar("Step/Reward", reward, total_steps)

            total_steps += 1
            mavic.step_robot()

        # Update networks and get losses
        actor_loss, critic_loss = update_networks(episode_experiences)

        # Log episode-level metrics
        writer.add_scalar("Episode/Total Reward", episode_reward, current_episode)
        writer.add_scalar(
            "Episode/Average State Error",
            np.mean(episode_state_errors),
            current_episode,
        )
        writer.add_scalar(
            "Episode/Average Control Effort",
            np.mean(episode_control_efforts),
            current_episode,
        )
        writer.add_scalar("Episode/Actor Loss", actor_loss, current_episode)
        writer.add_scalar("Episode/Critic Loss", critic_loss, current_episode)

        # Log histograms for this episode
        writer.add_histogram(
            "Episode/State Errors", np.array(episode_state_errors), current_episode
        )
        writer.add_histogram(
            "Episode/Control Efforts",
            np.array(episode_control_efforts),
            current_episode,
        )
        writer.add_histogram(
            "Episode/Rewards", np.array(episode_rewards), current_episode
        )

        # Ensure metrics are written to disk
        writer.flush()

        logger.info(
            f"Episode {current_episode} | "
            f"Reward: {episode_reward:.2f} | "
            f"Actor Loss: {actor_loss:.4f} | "
            f"Critic Loss: {critic_loss:.4f} | "
            f"Avg State Error: {np.mean(episode_state_errors):.4f}"
        )

        current_episode += 1

    writer.close()
    logger.info("Training completed. TensorBoard logs saved.")


def get_state():
    imu_values = mavic.get_imu_values()
    gps_values = mavic.get_gps_values()
    gyro_values = mavic.get_gyro_values()
    return np.concatenate([imu_values, gps_values, gyro_values])


def calculate_reward(state, next_state):
    state_error = np.linalg.norm(next_state - desired_state)
    control_effort_penalty = np.sum(np.square(next_state - state))
    return -state_error - 0.01 * control_effort_penalty


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
