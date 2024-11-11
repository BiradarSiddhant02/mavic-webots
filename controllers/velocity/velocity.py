import numpy as np
import pandas as pd
import torch.cuda as cuda
from torch.cuda.amp import autocast
from mavic import Mavic
from environment import Environment
from controller import Robot  # type: ignore
from constants import K_VERTICAL_THRUST, K_ROLL_P, K_PITCH_P, MAX_SIMULATION_TIME
from loguru import logger
from utils import calculate_pid_altitude, velocity_vector, clamp_motor_speed
from model import DepthModel
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Initialize a pool of threads and lock for buffer access
thread_pool = ThreadPoolExecutor(max_workers=8)
lock = Lock()

# Buffer to store depth outputs
depth_buffer = []
image_buffer = []
facings = []
batch_size = 8  # Set batch size for processing images in batches


def main():
    robot = Robot()
    mavic = Mavic(robot)
    logger.info("Drone object created")

    altitudes = [1 + i / 2 for i in range(10)]
    current_altitude_idx = 0
    target_altitude = altitudes[current_altitude_idx]
    change_time = mavic.get_time()

    integral_altitude_error = 0.0
    previous_altitude_error = 0.0
    timestep_seconds = mavic.timestep / 1000.0

    points = (
        (0, 0, 10),
        (0, 0, -10)
    )

    prev_position = mavic.get_gps_values()

    while mavic.step_robot() != -1:
        current_time = mavic.get_time()
        if current_time >= MAX_SIMULATION_TIME:
            break

        roll, pitch, _ = mavic.get_imu_values()
        x_1, y_1, z_1 = mavic.get_gps_values()
        roll_velocity, pitch_velocity, _ = mavic.get_gyro_values()

        roll_input = K_ROLL_P * np.clip(roll, -1.0, 1.0) + roll_velocity
        pitch_input = K_PITCH_P * np.clip(pitch, -1.0, 1.0) + pitch_velocity
        yaw_input = 0.0

        vertical_input, integral_altitude_error, previous_altitude_error = (
            calculate_pid_altitude(
                target_altitude,
                z_1,
                timestep_seconds,
                integral_altitude_error,
                previous_altitude_error,
            )
        )

        base_thrust = K_VERTICAL_THRUST + vertical_input
        motor_speeds = [
            clamp_motor_speed(base_thrust - roll_input + pitch_input - yaw_input),
            clamp_motor_speed(base_thrust + roll_input + pitch_input + yaw_input),
            clamp_motor_speed(base_thrust - roll_input - pitch_input + yaw_input),
            clamp_motor_speed(base_thrust + roll_input - pitch_input - yaw_input),
        ]
        # print(motor_speeds)

        mavic.set_rotor_speed(
            (motor_speeds[0], -motor_speeds[1], -motor_speeds[2], motor_speeds[3])
        )

        vel = velocity_vector(
            prev_position, mavic.get_gps_values(), mavic.timestep / 1000.0
        )

        prev_position = mavic.get_gps_values()
        
        facing = []

        for p in points:
            O = (x_1, y_1, z_1)
            d = np.array(p) - np.array(O)
            
            # Calculate norms and handle small magnitudes
            vel_norm = np.linalg.norm(vel)
            d_norm = np.linalg.norm(d)

            # Normalize vectors
            vel_unit = vel / vel_norm
            d_unit = d / d_norm if d_norm > 0 else np.zeros_like(d)

            # Calculate dot product
            vel_dot_d = np.dot(vel_unit, d_unit)
            facing.append(1 - ((vel_dot_d + 1) / 2))

        # Append the list of dot products for this timestep to facings
        facings.append(facing)

    # Convert facings to a numpy array for easier analysis
    facings_df = pd.DataFrame(facings)
    facings_df.to_csv("facings.csv", index=False)

if __name__ == "__main__":
    main()
