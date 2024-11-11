from controller import Robot  # type: ignore
from constants import K_VERTICAL_THRUST, K_ROLL_P, K_PITCH_P, MAX_SIMULATION_TIME
from loguru import logger
from utils import calculate_pid_altitude, velocity_vector, clamp_motor_speed
from mavic import Mavic
import numpy as np

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
        print(mavic.get_gps_values(), [x_1, y_1, z_1])

if __name__ == "__main__":
    main()
