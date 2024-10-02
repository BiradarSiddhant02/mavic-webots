#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm> // For std::clamp

#include <Mavic.hpp>
#include <constants.hpp>

#include <webots/Robot.hpp>
#include <webots/Camera.hpp>
#include <webots/GPS.hpp>
#include <webots/Gyro.hpp>
#include <webots/Motor.hpp>
#include <webots/InertialUnit.hpp>

double altitude_pid(
    double target_altitude,
    double current_altitude,
    double delta_time,
    double* integral_altitude_error,
    double* previous_altitude_error
) {

    // Compute the error between the target altitude and the current altitude
    double altitude_error = target_altitude - current_altitude;

    // Update the integral of the error
    *integral_altitude_error += altitude_error * delta_time;

    // Compute the derivative of the error
    double derivative_altitude_error = (altitude_error - *previous_altitude_error) / delta_time;

    // Compute the control signal using PID formula
    double control_signal = K_VERTICAL_P * altitude_error + K_VERTICAL_I * (*integral_altitude_error) + K_VERTICAL_D * derivative_altitude_error;

    // Update the previous altitude error
    *previous_altitude_error = altitude_error;

    return control_signal;
}

int main() {
    Mavic* mavic = new Mavic();

    // Declare variables outside the loop
    int current_altitude_index = 0;
    double altitudes[4] = {1., 2., 3., 4.};
    double target_altitude = altitudes[current_altitude_index];

    double change_time = mavic->get_time();

    // PID control variables
    double integral_altitude_error = 0.0;
    double previous_altitude_error = 0.0;

    // Sensor values
    const double* imu_values;
    const double* gps_values;

    // Control inputs
    double roll, pitch, roll_input, pitch_input, yaw_input;
    double current_altitude;
    double control_signal;

    while (true) {
        double current_time = mavic->get_time();

        // Check if it is time to change altitude
        if (current_time - change_time >= 10) {
            current_altitude_index = (current_altitude_index + 1) % (sizeof(altitudes) / sizeof(double));
            target_altitude = altitudes[current_altitude_index];
            change_time = current_time;
        }

        // Read Sensor values
        imu_values = mavic->get_imu_values();
        gps_values = mavic->get_gps_values();

        // Calculate roll, pitch and yaw inputs
        roll = imu_values[0];
        pitch = imu_values[1];
        roll_input = K_ROLL_P * std::clamp(roll, -1.0, 1.0);
        pitch_input = K_PITCH_P * std::clamp(pitch, -1.0, 1.0);
        yaw_input = 0.0; // Assuming yaw input remains constant or is calculated elsewhere

        current_altitude = gps_values[2]; // Get the altitude from GPS data

        control_signal = altitude_pid(
            target_altitude,
            current_altitude,
            mavic->timestep / 1000.0,
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

        const double rotor_speeds[4] = {
            front_left_motor_input,
            front_right_motor_input,
            rear_left_motor_input,
            rear_right_motor_input
        };

        mavic->set_rotor_speed(rotor_speeds);

        // Further processing for rotor speeds can be added here
    }

    // Clean up
    delete mavic;

    return 0;
}
