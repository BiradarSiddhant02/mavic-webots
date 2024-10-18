from constants import *

import numpy as np
import torch
import cv2

def calculate_pid_altitude(target, current, timestep, integral, prev_error):
    # Calculates the PID control output for altitude adjustment
    error = target - current
    integral += error * timestep
    derivative = (error - prev_error) / timestep
    pid_output = (K_VERTICAL_P * error) + (K_VERTICAL_I * integral) + (K_VERTICAL_D * derivative)
    return pid_output, integral, error

def clamp_motor_speed(input_speed):
    # Clamps the motor speed within the allowable range to prevent excessive values
    return max(min(input_speed, MAX_MOTOR_VELOCITY), -MAX_MOTOR_VELOCITY)

def process_image(raw_image, width, height, device):
    # Converts a raw image to a PyTorch tensor after resizing and moving to the specified device
    img_array = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
    img_resized = cv2.resize(img_array, (60 * 4, 80 * 4), interpolation=cv2.INTER_AREA)
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(device)
    return img_tensor

# Function to perform depth inference asynchronously
def depth_inference_batch(depth_model, img_tensors, buffer, lock):
    # Perform depth inference on a batch of images with mixed precision
    with torch.no_grad():
        with torch.amp.autocast():
            depths = depth_model(torch.stack(img_tensors))
        with lock:
            buffer.extend([depth.cpu().numpy() for depth in depths])
