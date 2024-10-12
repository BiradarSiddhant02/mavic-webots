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
import threading

def depth_inference(depth_model, img_tensor, depth_output):
    with torch.no_grad():
        depth = depth_model(img_tensor)
        depth_output.append(depth.cpu().numpy())

def main():
    mavic = Mavic()
    logger.info("Drone object created")

    altitudes = [1 + i / 2 for i in range(5)]
    current_altitude_idx = 0
    target_altitude = altitudes[current_altitude_idx]
    change_time = mavic.get_time()
    
    integral_altitude_error = 0.0
    previous_altitude_error = 0.0
    timestep_seconds = mavic.timestep / 1000.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_model = DepthModel().to(device)
    depth_model.load_state_dict(torch.load("weights/TFL_5_3.pth", weights_only=True, map_location=device))
    depth_model.eval()
    logger.info("Depth Model Loaded")

    depth_output = []

    while mavic.step_robot() <= MAX_SIMULATION_TIME:
        current_time = mavic.get_time()
        if current_time >= MAX_SIMULATION_TIME:
            break

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

        mavic.set_rotor_speed((motor_speeds[0], -motor_speeds[1], -motor_speeds[2], motor_speeds[3]))

        raw_image = mavic.camera.getImage()
        image_width, image_height = mavic.camera.getWidth(), mavic.camera.getHeight()
        img_tensor = process_image(raw_image, image_width, image_height, device)

        # Start a new thread for depth inference
        depth_thread = threading.Thread(target=depth_inference, args=(depth_model, img_tensor, depth_output))
        depth_thread.start()

        # Join the thread if you need to wait for it to finish
        # depth_thread.join()  # Uncomment if you want to wait for depth inference to complete

if __name__ == "__main__":
    main()
