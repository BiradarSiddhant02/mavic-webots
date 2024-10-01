import numpy as np
import cv2
import torch
from mavic import Mavic
from constants import (
    K_VERTICAL_THRUST,
    K_ROLL_P, 
    K_PITCH_P, 
    MAX_SIMULATION_TIME
)
from loguru import logger
from utils import calculate_pid_altitude, process_image, clamp_motor_speed
from model import DepthModel

def main():
    mavic = Mavic()
    logger.info("Drone object created")

    altitudes = [1 + i / 2 for i in range(5)]
    current_altitude_idx = 0
    target_altitude = altitudes[current_altitude_idx]

    change_time = mavic.get_time()

    # PID control state
    integral_altitude_error = 0.0
    previous_altitude_error = 0.0
    timestep_seconds = mavic.timestep / 1000.0  # Convert to seconds once

    # Move DepthModel and constants to GPU once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_model = DepthModel().to(device)
    depth_model.load_state_dict(torch.load("weights/TFL_5_3.pth", weights_only=True, map_location=device))
    depth_model.eval()  # Set to evaluation mode to avoid dropout/batchnorm training behavior
    logger.info("Depth Model Loaded")

    while mavic.drone.step(mavic.timestep) <= MAX_SIMULATION_TIME:
        current_time = mavic.get_time()
        if current_time >= MAX_SIMULATION_TIME:
            break

        # Change altitude at intervals
        if current_time - change_time >= 10.0:
            current_altitude_idx = (current_altitude_idx + 1) % len(altitudes)
            target_altitude = max(altitudes[current_altitude_idx], 0.5)  # Ensure altitude does not go below 0.5
            change_time = current_time
            logger.info(f"New altitude: {target_altitude}")

        # Read sensor values once per loop
        roll, pitch, yaw = mavic.get_imu_values()
        x, y, z = mavic.get_gps_values()
        roll_velocity, pitch_velocity, yaw_velocity = mavic.get_gyro_values()

        # Calculate roll, pitch, and yaw inputs once
        roll_input = K_ROLL_P * max(min(roll, 1.0), -1.0) + roll_velocity
        pitch_input = K_PITCH_P * max(min(pitch, 1.0), -1.0) + pitch_velocity
        yaw_input = 0.0  # Assuming no yaw disturbances

        # PID altitude control
        vertical_input, integral_altitude_error, previous_altitude_error = calculate_pid_altitude(
            target_altitude, z, timestep_seconds, integral_altitude_error, previous_altitude_error
        )

        # Precompute common motor input values
        base_thrust = K_VERTICAL_THRUST + vertical_input
        roll_adjustment = roll_input
        pitch_adjustment = pitch_input
        yaw_adjustment = yaw_input

        # Calculate motor speeds and clamp them
        motor_speeds = [
            clamp_motor_speed(base_thrust - roll_adjustment + pitch_adjustment - yaw_adjustment),
            clamp_motor_speed(base_thrust + roll_adjustment + pitch_adjustment + yaw_adjustment),
            clamp_motor_speed(base_thrust - roll_adjustment - pitch_adjustment + yaw_adjustment),
            clamp_motor_speed(base_thrust + roll_adjustment - pitch_adjustment - yaw_adjustment)
        ]

        # Set rotor speeds (Negatives for some to correct orientation)
        mavic.set_rotor_speed((
            motor_speeds[0],
            -motor_speeds[1],
            -motor_speeds[2],
            motor_speeds[3]
        ))

        # Capture and process image
        raw_image = mavic.camera.getImage()  # Get raw bytes from the camera
        image_width, image_height = mavic.camera.getWidth(), mavic.camera.getHeight()

        # Process image (resize and normalize)
        img_array = process_image(raw_image, image_width, image_height)
        img_resized = cv2.resize(img_array, (120, 200))  # Resize once, outside the tensor conversion
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(device)

        # Perform depth inference
        with torch.no_grad():  # Disable gradients for inference
            depth = depth_model(img_tensor)

if __name__ == "__main__":
    main()
