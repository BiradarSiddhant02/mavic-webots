import csv
from mavic import Mavic
from constants import (
    K_VERTICAL_THRUST,
    K_ROLL_P, 
    K_PITCH_P, 
    MAX_SIMULATION_TIME,
    MAX_MOTOR_VELOCITY
)
from loguru import logger
from utils import calculate_pid_altitude, clamp_motor_speed

def main():
    mavic = Mavic()
    target_altitude = 1.0

    integral_altitude_error = 0.0
    previous_altitude_error = 0.0
    timestep_seconds = mavic.timestep / 1000.0  # Convert to seconds once

    csv_file = open("unit_step_response.csv", "w")
    writer = csv.writer(csv_file)
    writer.writerow(["time", "target_altitude", "current_altitude"])

    while mavic.drone.step(mavic.timestep) <= MAX_SIMULATION_TIME:
        
        current_time = mavic.get_time()
        if current_time >= MAX_SIMULATION_TIME:
            break

        # Read sensor values once per loop
        x, y, z = mavic.get_gps_values()
        
        mavic.set_rotor_speed((
            MAX_MOTOR_VELOCITY,
            -MAX_MOTOR_VELOCITY,
            -MAX_MOTOR_VELOCITY,
            MAX_MOTOR_VELOCITY
        ))

        writer.writerow([f"{current_time:.4f}", f"{target_altitude:.4f}", f"{z:.4f}"])
        
if __name__ == "__main__":
    main()
