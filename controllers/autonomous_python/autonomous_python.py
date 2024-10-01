from mavic import Mavic

from constants import (K_VERTICAL_THRUST, K_VERTICAL_OFFSET,
                       K_VERTICAL_P, K_VERTICAL_I, K_VERTICAL_D,
                       K_ROLL_P, K_PITCH_P, MAX_SIMULATION_TIME, MAX_MOTOR_VELOCITY)

from loguru import logger

def calculate_pid_altitude(
        target, 
        current, 
        timestep, 
        integral_altitude, 
        previous_altitude_error
):
    # Compute the error
    error = target - current

    # Proportional term
    p_term = K_VERTICAL_P * error

    # Integral term (accumulated over time)
    integral_altitude += error * timestep
    i_term = K_VERTICAL_I * integral_altitude

    # Derivative term (change in error)
    derivative = (error - previous_altitude_error) / timestep
    d_term = K_VERTICAL_D * derivative

    # Store the error for the next timestep
    previous_altitude_error = error

    # Return the total PID output and updated values
    return p_term + i_term + d_term, integral_altitude, previous_altitude_error

def main():
    mavic = Mavic()
    logger.info("Drone object created")

    current_altitude = 0
    altitudes = [1 + i/2 for i in range(5)]
    target_altitude = altitudes[current_altitude]

    change_time = mavic.get_time()

    integral_altitude_error = 0.0
    previous_altitude_error = 0.0
    
    while mavic.drone.step(mavic.timestep) <= MAX_SIMULATION_TIME:

        current_time = mavic.get_time()  # in seconds
        if current_time >= MAX_SIMULATION_TIME:
            break
        
        ## Check if it is time to change altitude
        if current_time - change_time >= 10.0:
            current_altitude = (current_altitude + 1) % len(altitudes)
            target_altitude = max(altitudes[current_altitude], 0.5)  # Ensure altitude does not go below 0.5
            change_time = current_time  # Update the change time
            logger.info(f"New altitude: {target_altitude}")

        ## Read sensor values
        roll, pitch, yaw = mavic.get_imu_values()
        logger.info("roll, pitch, yaw read")
        x, y, z = mavic.get_gps_values()
        logger.info("x, y, z read")
        roll_velocity, pitch_velocity, yaw_velocity = mavic.get_gyro_values()
        logger.info("roll velocity, pitch velocity, yaw velocity read")

        ## Calculate roll, pitch, yaw inputs
        roll_input = K_ROLL_P * max(min(roll, 1.0), -1.0) + roll_velocity
        pitch_input = K_PITCH_P * max(min(pitch, 1.0), -1.0) + pitch_velocity
        yaw_input = 0.0  # Assuming no yaw disturbances
        logger.info(f"Inputs calculated")

        # PID controller for altitude
        vertical_input, integral_altitude_error, previous_altitude_error = calculate_pid_altitude(
            target_altitude,
            z,
            mavic.timestep/ 1000.0,
            integral_altitude_error,
            previous_altitude_error
        )
        logger.info("Control effort calculated")

        ## Calculate Rotor speeds
        front_left_motor_input = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input
        front_right_motor_input = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input
        rear_left_motor_input = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input
        rear_right_motor_input = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input

        # Clamp motor velocities to the maximum value
        front_left_motor_input = max(min(front_left_motor_input, MAX_MOTOR_VELOCITY), -MAX_MOTOR_VELOCITY)
        front_right_motor_input = max(min(front_right_motor_input, MAX_MOTOR_VELOCITY), -MAX_MOTOR_VELOCITY)
        rear_left_motor_input = max(min(rear_left_motor_input, MAX_MOTOR_VELOCITY), -MAX_MOTOR_VELOCITY)
        rear_right_motor_input = max(min(rear_right_motor_input, MAX_MOTOR_VELOCITY), -MAX_MOTOR_VELOCITY)
        logger.info("Rotor speeds calulated")

        mavic.set_rotor_speed(
            (
                front_left_motor_input,
                -front_right_motor_input,
                -rear_left_motor_input,
                rear_right_motor_input
            )
        )
        logger.info("Rotor speed set")

main()