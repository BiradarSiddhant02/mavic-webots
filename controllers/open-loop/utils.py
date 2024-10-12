from constants import K_VERTICAL_P, K_VERTICAL_I, K_VERTICAL_D, MAX_MOTOR_VELOCITY

import numpy as np

def calculate_pid_altitude(target, current, timestep, integral, prev_error):
    error = target - current
    integral += error * timestep
    derivative = (error - prev_error) / timestep

    pid_output = (K_VERTICAL_P * error) + (K_VERTICAL_I * integral) + (K_VERTICAL_D * derivative)
    return pid_output, integral, error

def clamp_motor_speed(input):
    return max(min(input, MAX_MOTOR_VELOCITY), -MAX_MOTOR_VELOCITY)

def process_image(raw_image, width, height):
    # Convert raw image bytes into a NumPy array and reshape to (height, width, 3) for RGB
    img_array = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 4))  # RGBA format
    img_array = img_array[:, :, :3]  # Strip the alpha channel if needed (convert RGBA to RGB)
    return img_array