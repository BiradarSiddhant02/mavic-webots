from mavic import Mavic
from controller import Robot, Keyboard  # type: ignore
import csv

# Initialize the Robot
robot = Robot()


def calc_pid(target, current, timestep, integral, prev_error, kp, ki, kd):
    error = target - current
    integral += error * timestep
    derivative = (error - prev_error) / timestep
    output = kp * error + ki * integral + kd * derivative
    return output, integral, error


def clamp(value, low, high):
    return max(min(value, high), low)


def main():
    integral_error = 0
    prev_error = 0

    stabilization_count = 0
    stabilization_threshold = 100

    target_altitude = 1.0

    # Initialize the Mavic drone
    mavic = Mavic(robot)
    keyboard = Keyboard()
    keyboard.enable(mavic.timestep)

    # Initialize the PID constants
    kp = 10
    ki = 0
    kd = 5.0

    with open("flight_log.csv", mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(
            [
                "time",
                "imu_roll",
                "imu_pitch",
                "imu_yaw",
                "gyro_roll_rate",
                "gyro_pitch_rate",
                "gyro_yaw_rate",
                "gps_x",
                "gps_y",
                "gps_altitude",
                "motor_fl",
                "motor_fr",
                "motor_rl",
                "motor_rr",
                "target_altitude",
                "current_altitude",
                "roll_disturbance",
                "pitch_disturbance",
                "yaw_disturbance",
            ]
        )

        while mavic.step_robot() != -1:
            imu = mavic.get_imu_values()
            roll, pitch, yaw = imu

            gyro = mavic.get_gyro_values()
            roll_rate, pitch_rate, yaw_rate = gyro

            gps = mavic.get_gps_values()
            x, y, z = gps

            # Control disturbances based on keyboard input
            roll_disturbance = 0
            pitch_disturbance = 0
            yaw_disturbance = 0

            key = keyboard.getKey()
            if key == ord("W"):
                pitch_disturbance = -2
            elif key == ord("S"):
                pitch_disturbance = 2
            elif key == ord("A"):
                roll_disturbance = 1
            elif key == ord("D"):
                roll_disturbance = -1
            elif key == ord("Q"):
                target_altitude += 0.005
            elif key == ord("E"):
                target_altitude -= 0.005
            elif key == Keyboard.RIGHT:
                yaw_disturbance = -1.3
            elif key == Keyboard.LEFT:
                yaw_disturbance = 1.3

            # PID controller for altitude
            altitude_output, integral_error, prev_error = calc_pid(
                target_altitude,
                z,
                mavic.timestep / 1000,
                integral_error,
                prev_error,
                kp,
                ki,
                kd,
            )

            # Compute the roll, pitch, and yaw inputs
            roll_input = 50 * clamp(roll, -1, 1) + roll_rate + roll_disturbance
            pitch_input = 30 * clamp(pitch, -1, 1) + pitch_rate + pitch_disturbance
            yaw_input = yaw_disturbance

            # Rotor speed calculations with clamping between -576 and 576
            front_left_motor_input = clamp(
                68.5 + altitude_output - roll_input + pitch_input - yaw_input, -576, 576
            )
            front_right_motor_input = clamp(
                68.5 + altitude_output + roll_input + pitch_input + yaw_input, -576, 576
            )
            rear_left_motor_input = clamp(
                68.5 + altitude_output - roll_input - pitch_input + yaw_input, -576, 576
            )
            rear_right_motor_input = clamp(
                68.5 + altitude_output + roll_input - pitch_input - yaw_input, -576, 576
            )

            mavic.set_rotor_speed(
                (
                    front_left_motor_input,
                    -front_right_motor_input,  # Negative sign for opposite rotation
                    -rear_left_motor_input,  # Negative sign for opposite rotation
                    rear_right_motor_input,
                )
            )

            # Check if drone is within the stabilization threshold
            if abs(target_altitude - z) < 0.01:  # Adjust tolerance as needed
                stabilization_count += 1
            else:
                stabilization_count = 0  # Reset if out of tolerance

            # Reset integral and previous errors if stabilized for a threshold count
            if stabilization_count >= stabilization_threshold:
                integral_error = 0
                prev_error = 0

            writer.writerow(
                [
                    mavic.get_time(),
                    roll,
                    pitch,
                    yaw,
                    roll_rate,
                    pitch_rate,
                    yaw_rate,
                    x,
                    y,
                    z,
                    front_left_motor_input,
                    front_right_motor_input,
                    rear_left_motor_input,
                    rear_right_motor_input,
                    target_altitude,
                    z,
                    roll_disturbance,
                    pitch_disturbance,
                    yaw_disturbance,
                ]
            )


main()
