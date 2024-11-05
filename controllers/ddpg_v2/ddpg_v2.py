from mavic import Mavic
from controller import Robot    # type: ignore

from actor_critic import Actor, Critic

from noise import OUNoise

from buffer import ReplayBuffer

from constants import *

import torch
from typing import Tuple, List
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## ---- Actor Networks ---- ##
actor_local = Actor(
    STATE_DIM,
    ACTOR_HIDDEN_DIM1,
    ACTOR_HIDDEN_DIM2,
    ACTION_DIM
).to(device)

actor_target = Actor(
    STATE_DIM,
    ACTOR_HIDDEN_DIM1,
    ACTOR_HIDDEN_DIM2,
    ACTION_DIM
).to(device)
actor_target.load_state_dict(actor_local.state_dict())

actor_local_optimizer = torch.optim.Adam(actor_local.parameters(), lr=ACTOR_LEARNING_RATE)

## ---- Critic Networks ---- ##
critic_local = Critic(
    STATE_DIM + ACTION_DIM,
    CRITIC_HIDDEN_DIM1,
    CRITIC_HIDDEN_DIM2,
    1
).to(device)

critic_target = Critic(
    STATE_DIM + ACTION_DIM,
    CRITIC_HIDDEN_DIM1,
    CRITIC_HIDDEN_DIM2,
    1
).to(device)
critic_target.load_state_dict(critic_local.state_dict())

critic_local_optimizer = torch.optim.Adam(critic_local.parameters(), lr=CRITIC_LEARNING_RATE)

## ---- Noise ---- ##
noise = OUNoise(ACTION_DIM)

## ---- Replay Buffer ---- ##
buffer = ReplayBuffer(BUFFER_SIZE, STATE_DIM, ACTION_DIM)

## ---- Drone PID Variables ---- ##
integral_error = 0
prev_error = 0

## ---- Agent ---- ##
robot = Robot()
mavic = Mavic(robot)

## ---- Helper Functions ---- ##
mse_loss = torch.nn.MSELoss()

def get_state(mavic: Mavic) -> np.ndarray:
    imu = np.array(mavic.get_imu_values())
    gps = np.array(mavic.get_gps_values())
    gyro = np.array(mavic.get_gyro_values())
    
    return np.concatenate((imu, gps, gyro), axis=0)

def calculate_reward(state: List[float], next_state: List[float]) -> float:
    pass

def update_networks() -> None:
    state, action, reward, next_state = buffer.sample_buffer(BATCH_SIZE)
    with torch.no_grad():
        y = reward + GAMMA * critic_target(next_state, actor_target(next_state), dim=1)

    ## Update the Critic Network
    critic_loss = mse_loss(critic_local(state, action).squeeze(1), y)
    
    critic_local_optimizer.zero_grad()
    critic_loss.backward()
    critic_local_optimizer.step()
    
    ## Update the Actor Network
    policy_loss = -critic_local(state, actor_local(state)).mean()
    
    actor_local_optimizer.zero_grad()
    policy_loss.backward()
    actor_local_optimizer.step()
    
    ## Update the Target Networks
    for target_param, local_param in zip(actor_target.parameters(), actor_local.parameters()):
        target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)
        
    ## Update the Critic Target Network
    for target_param, local_param in zip(critic_target.parameters(), critic_local.parameters()):
        target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)
        
def apply_action(
    mavic: Mavic, 
    state: torch.Tensor, 
    target_state: List[float], 
    integral_error: float, 
    prev_error: float
) -> None:
    def pid(
        state: List, 
        target: List, 
        integral_error: float,
        prev_error: float
    ) -> Tuple[float, float, float]:
        current_altitude = state[5]
        target_altitude = target[5]
        
        altitude_error = target_altitude - current_altitude
        
        integral_error += altitude_error * mavic.timestep
        
        derivative_error = (altitude_error - prev_error) / mavic.timestep
        
        output = Kp * altitude_error + Ki * integral_error + Kd * derivative_error
        
        return output, integral_error, altitude_error

    roll, pitch, yaw = state[0], state[1], state[2]
    
        
    
    def clamp(value: float, low: float, high: float) -> float:
        return max(min(value, high), low)
    
    
    
def main() -> None:
    current_episode = 0
    total_steps = 0
    
    while current_episode < NUM_EPISODES:
        mavic.reset()
        state = get_state(mavic)
        episode_reward = 0
        prev_action = torch.zeros(ACTION_DIM)
        
        for step in range(NUM_STEPS):
            
            ## Get the action from the Actor Network
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action = actor_local(state_tensor)
            
            ## Perform the action
            apply_action(mavic, action, step)
            for _ in range(32 // mavic.timestep):
                mavic.step_robot()
                
            ## Get the next state
            next_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            reward = calculate_reward(state, next_state, action, prev_action)
            prev_action = action
            
            ## Store the experience in the replay buffer
            buffer.store_transition(state, reward, next_state, action)
            
            ## Update the Actor and Critic Networks
            if len(buffer) >= BATCH_SIZE:
                update_networks()
                
            ## Update the episode reward
            episode_reward += reward
            
            ## Update the simulation
            mavic.step_robot()

