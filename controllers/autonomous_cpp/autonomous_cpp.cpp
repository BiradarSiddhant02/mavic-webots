#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm> // For std::clamp

#include "Mavic.h"
#include "constants.h"

double altitude_pid(
    double target_altitude, 
    double current_altitude,
    double timestep,
    double* integral_altitude_error,
    double* previous_altitude_error
) {
    // Calculate error
    double altitude_error = target_altitude - current_altitude;

    // Calculate integral of error
    *integral_altitude_error += altitude_error * timestep;

    // Calculate derivative of error
    double derivative_error = (altitude_error - *previous_altitude_error) / timestep;

    // Calculate PID output
    double pid_output = K_VERTICAL_P * altitude_error +
                        K_VERTICAL_I * (*integral_altitude_error) +
                        K_VERTICAL_D * derivative_error;

    // Update previous error
    *previous_altitude_error = altitude_error;

    return pid_output;
}

int main() {
    mavic = new Mavic();

    // Declare variables outside the loop
    int current_altitude_index = 0;
    double altitudes[4] = {1., 2., 3., 4.};
    double target_altitude = altitudes[current_altitude_index];

    double change_time = mavic.get_time();

    // PID control variables
    double integral_altitude_error = 0.0;
    double previous_altitude_error = 0.0;

    // Sensor values
    std::tuple<double, double, double> imu_values;
    std::tuple<double, double, double> gps_values;

    // Control inputs
    double roll, pitch, roll_input, pitch_input, yaw_input;
    double current_altitude;
    double control_signal;

    while (true) {
        double current_time = mavic.get_time();
        if (current_time >= MAX_SIMULATION_TIME)
            break;

        // Check if it is time to change altitude
        if (current_time - change_time >= MAX_STEP_LENGTH) {
            current_altitude_index = (current_altitude_index + 1) % (sizeof(altitudes) / sizeof(double));
            target_altitude = altitudes[current_altitude_index];
            change_time = current_time;
        }

        // Read Sensor values
        imu_values = mavic.get_imu_values();
        gps_values = mavic.get_gps_values();

        // Calculate roll, pitch and yaw inputs
        roll = std::get<0>(imu_values);
        pitch = std::get<1>(imu_values);
        roll_input = K_ROLL_P * std::clamp(roll, -1.0, 1.0);
        pitch_input = K_PITCH_P * std::clamp(pitch, -1.0, 1.0);
        yaw_input = 0.0; // Assuming yaw input remains constant or is calculated elsewhere

        current_altitude = std::get<2>(gps_values); // Get the altitude from GPS data

        control_signal = altitude_pid(
            target_altitude,
            current_altitude,
            mavic.timestep / 1000.0,
            &integral_altitude_error,
            &previous_altitude_error
        );

        // Motor control calculations
        double front_left_motor_input = std::clamp(
            K_VERTICAL_THRUST + control_signal - roll_input + pitch_input - yaw_input,
            -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED
        );

        double front_right_motor_input = std::clamp(
            K_VERTICAL_THRUST + control_signal + roll_input + pitch_input + yaw_input,
            -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED
        );

        double rear_left_motor_input = std::clamp(
            K_VERTICAL_THRUST + control_signal - roll_input - pitch_input + yaw_input,
            -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED
        );

        double rear_right_motor_input = std::clamp(
            K_VERTICAL_THRUST + control_signal + roll_input - pitch_input - yaw_input,
            -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED
        );

        std::tuple<double, double, double, double> rotor_speeds = std::make_tuple(
            front_left_motor_input,
            -front_right_motor_input,
            -rear_left_motor_input,
            rear_right_motor_input
        );

        // Further processing for rotor speeds can be added here
    }

    // Clean up
    delete mavic;

    return 0;
}
