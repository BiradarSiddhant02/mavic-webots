from controller import Robot, Supervisor    #   type: ignore

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import timedelta
import time
import os

robot = Robot()
start_time = robot.getTime()

# Seed Everything
seed = 42
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Environment(Supervisor):
    def __init__(self):
        super().__init__()

        # Drone-specific parameters
        self.max_rotor_speed = 576
        self.desired_state = np.array([3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Target position and orientation
        self.reach_threshold = 0.45  # Adjusted threshold for reaching target

        # Initialize Motors
        self.front_left_motor = robot.getDevice("front left propeller")
        self.front_right_motor = robot.getDevice("front right propeller")
        self.rear_left_motor = robot.getDevice("rear left propeller")
        self.rear_right_motor = robot.getDevice("rear right propeller")

        # Enable infinite rotation and set initial velocity to zero
        for motor in [self.front_left_motor, self.front_right_motor, self.rear_left_motor, self.rear_right_motor]:
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)

        # Enable GPS, IMU, and Gyro sensors
        self.gps = robot.getDevice("gps")
        self.gps.enable(32)  # Sampling period in ms

        self.imu = robot.getDevice("inertial unit")
        self.imu.enable(32)

        self.gyro = robot.getDevice("gyro")
        self.gyro.enable(32)

        # Reset the simulation
        self.simulationReset()
        self.simulationResetPhysics()

    def get_position(self):
        """Retrieve the current position from the GPS sensor."""
        position = np.array(self.gps.getValues())
        return position

    def get_orientation(self):
        """Retrieve the current orientation from the IMU sensor."""
        orientation = np.array(self.imu.getRollPitchYaw())
        return orientation

    def get_angular_velocity(self):
        """Retrieve the angular velocity from the Gyro sensor."""
        angular_velocity = np.array(self.gyro.getValues())
        return angular_velocity

    def get_state(self):
        """Combines position, orientation, and angular velocity into the current state vector."""
        position = self.get_position()
        orientation = self.get_orientation()
        angular_velocity = self.get_angular_velocity()
        state = np.concatenate((position, orientation, angular_velocity))
        return state

    def apply_action(self, rotor_speeds):
        """
        Apply given propeller speeds to the drone's motors.

        Parameters:
        - rotor_speeds: List of four floats representing each motor speed.
        """
        rotor_speeds = np.clip(rotor_speeds, 0, self.max_rotor_speed)
        self.front_left_motor.setVelocity(rotor_speeds[0])
        self.front_right_motor.setVelocity(rotor_speeds[1])
        self.rear_left_motor.setVelocity(rotor_speeds[2])
        self.rear_right_motor.setVelocity(rotor_speeds[3])

    def get_reward(self):
        """
        Calculate and return the reward based on position, orientation, and angular velocity.

        Returns:
        - reward (float): The computed reward based on sensor data.
        - done (bool): Indicates if the goal has been reached or a failure condition met.
        """
        # 1. Distance to goal
        current_position = self.get_position()
        distance_to_goal = np.linalg.norm(self.desired_state[:3] - current_position)
        
        # Reward for being closer to the target
        distance_reward = -distance_to_goal  # Larger reward as distance decreases
        
        # Check if within threshold
        if distance_to_goal < self.reach_threshold:
            distance_reward += 10.0  # Bonus for reaching the target

        # 2. Orientation reward
        current_orientation = self.get_orientation()
        target_orientation = self.desired_state[3:6]  # Desired roll, pitch, yaw
        orientation_error = np.linalg.norm(target_orientation - current_orientation)
        orientation_reward = -orientation_error  # Penalize deviation from target orientation

        # 3. Stability reward (minimal angular velocity)
        angular_velocity = self.get_angular_velocity()
        stability_reward = -np.linalg.norm(angular_velocity)  # Penalize high angular velocity
        
        # 4. Efficiency penalty (penalize excessive propeller speeds)
        rotor_speeds = np.array([self.front_left_motor.getVelocity(),
                                self.front_right_motor.getVelocity(),
                                self.rear_left_motor.getVelocity(),
                                self.rear_right_motor.getVelocity()])
        efficiency_penalty = -np.sum(rotor_speeds ** 2) * 0.0001  # Small penalty for high speeds

        # Total reward
        reward = distance_reward + orientation_reward + stability_reward + efficiency_penalty
        
        # Check if episode is done
        done = distance_to_goal < self.reach_threshold or distance_to_goal > 10
        
        return reward, done


    def reset(self):
        """Reset the environment and return the initial state."""
        self.simulationReset()
        self.simulationResetPhysics()
        return self.get_state()

    def step(self, rotor_speeds):
        """
        Take an action by setting propeller speeds and retrieve the next state and reward.
        
        Parameters:
        - rotor_speeds: List of four motor speeds.
        
        Returns:
        - state (numpy.ndarray): The new state after the action.
        - reward (float): The reward received.
        - done (bool): Whether the episode has ended.
        """
        self.apply_action(rotor_speeds)
        robot.step(int(self.getBasicTimeStep()))  # Execute one step in the simulation
        reward, done = self.get_reward()
        state = self.get_state()
        return state, reward, done
    
class Policy_Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy_Network, self).__init__()
        # Define the neural network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        print(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.output_layer(x), dim=-1)  # Changed to log_softmax
        print(x)
        return x

class Agent_REINFORCE_Drone():
    """Agent implementing the REINFORCE algorithm for drone control."""

    def __init__(self, save_path, load_path, num_episodes, max_steps, learning_rate, gamma, hidden_size, clip_grad_norm, baseline):
        self.save_path = save_path
        self.load_path = load_path

        # Hyper-parameters
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.clip_grad_norm = clip_grad_norm
        self.baseline = baseline

        # Initialize Network
        self.network = Policy_Network(input_size=9, hidden_size=self.hidden_size, output_size=4).to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

        # Drone environment
        self.env = Environment()

    def save(self, path):
        """Save the trained model parameters."""
        torch.save(self.network.state_dict(), os.path.join(self.save_path, path))

    def load(self):
        """Load pre-trained model parameters."""
        self.network.load_state_dict(torch.load(self.load_path))

    def compute_returns(self, rewards):
        """Compute discounted returns."""
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        if self.baseline:
            returns -= returns.mean()
        return returns

    def compute_loss(self, log_probs, returns):
        """Compute REINFORCE loss."""
        loss = -torch.sum(torch.stack(log_probs) * returns)
        return loss

    def train(self):
        start_time = robot.getTime()
        while robot.getTime() - start_time < 1.0:
            self.env.step((0, 0, 0, 0))
        """Train the drone with REINFORCE."""
        start_time = time.time()
        reward_history = []
        best_score = -np.inf

        for episode in range(1, self.num_episodes + 1):
            done = False
            state = self.env.reset()
            log_probs = []
            rewards = []
            ep_reward = 0

            for step in range(self.max_steps):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                action_probs = self.network(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                log_probs.append(log_prob)

                # Step in environment
                next_state, reward, done = self.env.step(action.cpu())
                rewards.append(reward)
                ep_reward += reward

                if done:
                    break

                state = next_state

            # Calculate returns and loss, backpropagation
            returns = self.compute_returns(rewards)
            loss = self.compute_loss(log_probs, returns)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            reward_history.append(ep_reward)

            if ep_reward > best_score:
                best_score = ep_reward
                self.save("best_weights.pt")

            print(f"Episode {episode}: Score = {ep_reward:.3f}")

        # Save final weights
        self.save("final_weights.pt")
        elapsed_time = time.time() - start_time
        print(f"Training completed in {timedelta(seconds=int(elapsed_time))}.")

# Training parameters
save_path = "./results"
load_path = "./results/final_weights.pt"
train_mode = True
num_episodes = 2000
max_steps = 500
learning_rate = 2.5e-4
gamma = 0.99
hidden_size = 6
clip_grad_norm = 5
baseline = True

if __name__ == "__main__":
    agent = Agent_REINFORCE_Drone(save_path, load_path, num_episodes, max_steps, learning_rate, gamma, hidden_size, clip_grad_norm, baseline)
    if train_mode:
        agent.train()