import torch
import numpy as np
import os
from typing import List, Tuple
from collections import deque, Counter
from loguru import logger
import random
from datetime import datetime
import pandas as pd

from mavic import Mavic
from policy_network import PolicyNetwork
from controller import Robot  # type: ignore
from constants import *


mavic = Mavic(Robot())
logger.info("Mavic initialized.")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

policy_net = PolicyNetwork(
    INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM, "policy_net_checkpoints"
).to(DEVICE)
policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
logger.info("Policy network initialized.")

os.makedirs("policy_net_checkpoints", exist_ok=True)
os.makedirs("reward_histories", exist_ok=True)


def clamp(value: float, low: float, high: float) -> float:
    return max(min(value, high), low)


def get_state():
    imu_values = mavic.get_imu_values()
    gps_values = mavic.get_gps_values()
    gyro_values = mavic.get_gyro_values()
    return np.concatenate([imu_values, gps_values, gyro_values])


def altitude_PID(target, current, timestep, integral, prev_error):
    error = target - current
    integral += error * timestep
    derivative = (error - prev_error) / timestep
    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral, error


def calculate_reward(
    state_buffer: deque,
    action_buffer: deque,
) -> Tuple[float, bool]:
    position_error = np.linalg.norm(np.array(DESIRED_STATE) - np.array(state_buffer[1]))
    orientation_error = np.linalg.norm(
        np.array(DESIRED_STATE[0:4]) - np.array(state_buffer[1][0:4])
    )
    state_error = position_error + orientation_error
    state_reward = 1 / np.pow((state_error + 1e-3), 0.2)

    gps0 = state_buffer[0][3:6]
    gps1 = state_buffer[1][3:6]

    gps_desired = DESIRED_STATE[3:6]

    d_vec = np.array(gps_desired) - np.array(gps0)
    v_vec = np.array(gps1) - np.array(gps0)

    d_dot_v = np.dot(d_vec, v_vec)
    vector_seperation = (
        2 * d_dot_v / ((np.linalg.norm(d_vec) * np.linalg.norm(v_vec)) + 1e-12) - 1
    )

    opposite_actions = {
        1: 0,
        0: 1,
        3: 2,
        2: 3,
        5: 4,
        4: 5,
        7: 6,
        6: 7,
    }

    control_effort_penalty = 0
    for i in range(1, len(action_buffer)):
        if action_buffer[i] == opposite_actions[action_buffer[i - 1]]:
            control_effort_penalty += 1
        else:
            control_effort_penalty -= 1

    control_effort_penalty /= NUM_STEPS - 1

    return (
        float(
            REWARD_SCALE * (state_reward * vector_seperation * control_effort_penalty)
        ),
        state_error < 0.1,
    )


def calculate_returns(rewards: List[np.float32]) -> List[np.float32]:
    returns = []
    R = 0
    for r in rewards[::-1]:
        R = r + GAMMA * R
        returns.insert(0, R)
    return returns


def calculate_loss(
    log_probs: List[torch.Tensor], returns: List[np.float32]
) -> torch.Tensor:
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
        # print(-log_prob * R)
    return torch.stack(policy_loss).sum()


def main() -> None:
    if TRAIN_MODE:
        reward_history = []
        best_reward = float("-inf")

        policy_net.train()

        for episode in range(NUM_EPISODES):
            mavic.reset()

            done = True
            state_buffer = deque(maxlen=2)
            action_buffer = deque(maxlen=NUM_STEPS)
            action_buffer.append(0)
            state_buffer.append(np.array(INITIAL_STATE))

            log_probs = []
            rewards = []
            episode_reward = 0

            current_step = 0

            altitude_integral_error = 0
            altitude_previous_error = 0

            while mavic.step_robot() != -1 and current_step < NUM_STEPS:
                target_altitude = 1
                roll_disturbance = 0
                pitch_disturbance = 0
                yaw_disturbance = 0

                # for _ in range(512 // mavic.timestep):
                #     mavic.step_robot()

                state_vector = get_state()
                state_buffer.append(state_vector)

                roll, pitch, yaw, x, y, z, roll_rate, pitch_rate, yaw_rate = (
                    state_vector
                )

                if current_step % ERROR_RESET == 0:
                    altitude_integral_error = 0
                    altitude_previous_error = 0

                action_probs = policy_net(
                    torch.tensor(
                        np.concat((state_vector, (action_buffer[-1],))),
                        dtype=torch.float32,
                    ).to(DEVICE)
                )
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_probs.append(action_dist.log_prob(action))
                action_buffer.append(action.item())
                action = action.item()

                if action == 0:
                    target_altitude += 0.1
                elif action == 1:
                    target_altitude -= 0.1
                elif action in {2, 3}:
                    roll_disturbance = 1 if action == 2 else -1
                elif action in {4, 5}:
                    pitch_disturbance = 2 if action == 4 else -2
                elif action in {6, 7}:
                    yaw_disturbance = 1.3 if action == 6 else -1.3
                else:
                    raise ValueError(
                        f"Invalid action: {action}. Expected values are 0 to 7."
                    )

                control_effort, altitude_integral_error, altitude_previous_error = (
                    altitude_PID(
                        target_altitude,
                        z,
                        mavic.timestep / 1e3,
                        altitude_integral_error,
                        altitude_previous_error,
                    )
                )

                roll_input = Kroll * np.clip(roll, -1, 1) + roll_rate + roll_disturbance
                pitch_input = (
                    Kpitch * np.clip(pitch, -1, 1) + pitch_rate + pitch_disturbance
                )
                yaw_input = yaw_disturbance

                front_left_motor_input = np.clip(
                    MIN_THRUST + control_effort - roll_input + pitch_input - yaw_input,
                    -MAX_ROTOR_SPEED,
                    MAX_ROTOR_SPEED,
                )

                front_right_motor_input = np.clip(
                    MIN_THRUST + control_effort + roll_input + pitch_input + yaw_input,
                    -MAX_ROTOR_SPEED,
                    MAX_ROTOR_SPEED,
                )

                rear_left_motor_input = np.clip(
                    MIN_THRUST + control_effort - roll_input - pitch_input + yaw_input,
                    -MAX_ROTOR_SPEED,
                    MAX_ROTOR_SPEED,
                )

                rear_right_motor_input = np.clip(
                    MIN_THRUST + control_effort + roll_input - pitch_input - yaw_input,
                    -MAX_ROTOR_SPEED,
                    MAX_ROTOR_SPEED,
                )

                mavic.set_rotor_speed(
                    (
                        front_left_motor_input,
                        -front_right_motor_input,
                        -rear_left_motor_input,
                        rear_right_motor_input,
                    )
                )

                reward, done = calculate_reward(state_buffer, action_buffer)
                rewards.append(reward)
                reward_history.append(reward)

                episode_reward += reward

                current_step += 1

            most_common_action = Counter(action_buffer).most_common(1)[0][0]
            logger.info(
                f"Episode: {episode:04d}, Reward: {episode_reward:.4f}, Most common action: {most_common_action}"
            )

            if episode % 20 == 0:
                policy_net.save(f"policy_{episode}.pth")

                df = pd.DataFrame(
                    {
                        "Reward": reward_history,
                    }
                ).to_csv(
                    os.path.join(
                        "reward_histories",
                        datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv",
                    ),
                    index=True,
                )

            returns = calculate_returns(rewards)
            loss = calculate_loss(log_probs, returns)
            loss.backward()
            policy_optimizer.step()

    else:
        weights = os.listdir("policy_net_checkpoints")
        weights.sort()
        latest_weights = weights[-1]

        policy_net.load_state_dict(
            torch.load(
                os.path.join("policy_net_checkpoints", latest_weights),
                weights_only=True,
            )
        )

        policy_net.eval()
        state_buffer = deque(maxlen=2)
        action_buffer = deque(maxlen=NUM_STEPS)
        action_buffer.append(0)
        state_buffer.append(np.array(INITIAL_STATE))
        
        target_altitude = 1
        roll_disturbance = 0
        pitch_disturbance = 0
        yaw_disturbance = 0

        altitude_integral_error = 0
        altitude_previous_error = 0

        state_buffer = deque(maxlen=2)
        action_buffer = deque(maxlen=NUM_STEPS)
        previous_action = 0

        current_step = 0

        while mavic.step_robot() != -1:
            state_vector = get_state()
            state_buffer.append(state_vector)

            roll, pitch, yaw, x, y, z, roll_rate, pitch_rate, yaw_rate = state_vector

            if current_step % ERROR_RESET == 0:
                altitude_integral_error = 0
                altitude_previous_error = 0

            action_probs = policy_net(
                torch.tensor(
                    np.concatenate((state_vector, (previous_action,))),
                    dtype=torch.float32,
                ).to(DEVICE)
            )

            action = torch.argmax(action_probs).item()
            action = action
            
            previous_action = action

            if action == 0:
                target_altitude += 0.1
            elif action == 1:
                target_altitude -= 0.1
            elif action in {2, 3}:
                roll_disturbance = 1 if action == 2 else -1
            elif action in {4, 5}:
                pitch_disturbance = 2 if action == 4 else -2
            elif action in {6, 7}:
                yaw_disturbance = 1.3 if action == 6 else -1.3
            else:
                raise ValueError(
                    f"Invalid action: {action}. Expected values are 0 to 7."
                )

            control_effort, altitude_integral_error, altitude_previous_error = (
                altitude_PID(
                    target_altitude,
                    z,
                    mavic.timestep / 1e3,
                    altitude_integral_error,
                    altitude_previous_error,
                )
            )

            roll_input = Kroll * np.clip(roll, -1, 1) + roll_rate + roll_disturbance
            pitch_input = (
                Kpitch * np.clip(pitch, -1, 1) + pitch_rate + pitch_disturbance
            )
            yaw_input = yaw_disturbance

            front_left_motor_input = np.clip(
                MIN_THRUST + control_effort - roll_input + pitch_input - yaw_input,
                -MAX_ROTOR_SPEED,
                MAX_ROTOR_SPEED,
            )

            front_right_motor_input = np.clip(
                MIN_THRUST + control_effort + roll_input + pitch_input + yaw_input,
                -MAX_ROTOR_SPEED,
                MAX_ROTOR_SPEED,
            )

            rear_left_motor_input = np.clip(
                MIN_THRUST + control_effort - roll_input - pitch_input + yaw_input,
                -MAX_ROTOR_SPEED,
                MAX_ROTOR_SPEED,
            )

            rear_right_motor_input = np.clip(
                MIN_THRUST + control_effort + roll_input - pitch_input - yaw_input,
                -MAX_ROTOR_SPEED,
                MAX_ROTOR_SPEED,
            )

            mavic.set_rotor_speed(
                (
                    front_left_motor_input,
                    -front_right_motor_input,
                    -rear_left_motor_input,
                    rear_right_motor_input,
                )
            )

            current_step += 1


main()
