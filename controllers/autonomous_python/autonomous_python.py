import numpy as np
import cv2
import torch
import torch.cuda as cuda
from torch.cuda.amp import autocast
from mavic import Mavic
from environment import Environment
from controller import Robot   # type: ignore
from constants import (
    K_VERTICAL_THRUST,
    K_ROLL_P, 
    K_PITCH_P, 
    MAX_SIMULATION_TIME
)
from loguru import logger
from utils import calculate_pid_altitude, process_image, clamp_motor_speed, depth_inference_batch
from model import DepthModel
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Initialize a pool of threads and lock for buffer access
thread_pool = ThreadPoolExecutor(max_workers=8)
lock = Lock()

# Buffer to store depth outputs
depth_buffer = []
image_buffer = []
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

    # Initialize device and load depth model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_model = DepthModel().to(device)
    depth_model.load_state_dict(torch.load("weights/TFL_5_3.pth", weights_only=True))
    depth_model.eval()
    logger.info("Depth Model Loaded")

    while mavic.step_robot() <= MAX_SIMULATION_TIME:
        current_time = mavic.get_time()
        if current_time >= MAX_SIMULATION_TIME:
            mavic.reset()
            integral_altitude_error = 0.0
            previous_altitude_error = 0.0

        # Update target altitude every 10 seconds
        if current_time - change_time >= 10.0:
            current_altitude_idx = (current_altitude_idx + 1) % len(altitudes)
            target_altitude = max(altitudes[current_altitude_idx], 0.5)
            change_time = current_time
            logger.info(f"New altitude: {target_altitude}")

        roll, pitch, _ = mavic.get_imu_values()
        _, _, z = mavic.get_gps_values()
        roll_velocity, pitch_velocity, _ = mavic.get_gyro_values()

        roll_input = K_ROLL_P * np.clip(roll, -1.0, 1.0) + roll_velocity
        pitch_input = K_PITCH_P * np.clip(pitch, -1.0, 1.0) + pitch_velocity
        yaw_input = 0.0

        vertical_input, integral_altitude_error, previous_altitude_error = calculate_pid_altitude(
            target_altitude, z, timestep_seconds, integral_altitude_error, previous_altitude_error
        )

        base_thrust = K_VERTICAL_THRUST + vertical_input
        motor_speeds = [
            clamp_motor_speed(base_thrust - roll_input + pitch_input - yaw_input),
            clamp_motor_speed(base_thrust + roll_input + pitch_input + yaw_input),
            clamp_motor_speed(base_thrust - roll_input - pitch_input + yaw_input),
            clamp_motor_speed(base_thrust + roll_input - pitch_input - yaw_input)
        ]
        # print(motor_speeds)

        mavic.set_rotor_speed((motor_speeds[0], -motor_speeds[1], -motor_speeds[2], motor_speeds[3]))

        # Fetch and process image asynchronously
        raw_image = mavic.camera.getImage()
        image_width, image_height = mavic.camera.getWidth(), mavic.camera.getHeight()
        img_tensor = process_image(raw_image, image_width, image_height, device)
        image_buffer.append(img_tensor)

        # If buffer is full, process images in a batch
        if len(image_buffer) >= batch_size:
            thread_pool.submit(depth_inference_batch, depth_model, image_buffer.copy(), depth_buffer, lock)
            image_buffer.clear()  # Clear buffer after submitting to the thread pool

if __name__ == "__main__":
    main()
