#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <algorithm>
#include <cstring>

#include <Mavic.hpp>
#include <constants.hpp>

double altitude_pid(double target_altitude, double current_altitude, double delta_time, 
                    double &integral_altitude_error, double &previous_altitude_error)
{
    double altitude_error = target_altitude - current_altitude;
    integral_altitude_error += altitude_error * delta_time;
    double derivative_altitude_error = (altitude_error - previous_altitude_error) / delta_time;

    double control_signal = K_VERTICAL_P * altitude_error + 
                            K_VERTICAL_I * integral_altitude_error + 
                            K_VERTICAL_D * derivative_altitude_error;

    previous_altitude_error = altitude_error;

    return control_signal;
}

void print_vec(const double *vec, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

void print_image(const unsigned char *image, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            std::cout << static_cast<int>(image[i * width + j]) << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    std::cout << "[INFO] Initializing Mavic object." << std::endl;
    Mavic mavic;
    std::cout << "[INFO] Initialized Mavic object." << std::endl;

    while (wb_robot_step(mavic.timestep) != -1)
    {
        if (wb_robot_get_time() > 1.0)
            break;
    }

    const char* filename = "test.png";
    mavic.save_image(filename, 100);

    double target_altitude = 1.0;
    double integral_altitude_error = 0.0;
    double previous_altitude_error = 0.0;

    const double timestep_seconds = mavic.timestep / 1000.0;

    // Main loop
    while (mavic.robot_step() != -1)
    {
        const double time = mavic.get_time();
        if (time >= MAX_SIMULATION_TIME)
        {
            break;
        }

        const double *imu_values = mavic.get_imu_values();
        double roll = imu_values[0];
        double pitch = imu_values[1];

        const double *location = mavic.get_gps_values();
        double altitude = location[2];

        double roll_velocity = mavic.get_gyro_values()[0];
        double pitch_velocity = mavic.get_gyro_values()[1];

        // No yaw disturbances used, so it remains zero
        double yaw_disturbance = 0.0;

        // PID controller for altitude
        double vertical_input = altitude_pid(target_altitude, altitude, timestep_seconds, 
                                             integral_altitude_error, previous_altitude_error);

        // Roll and pitch inputs with clamped values
        double roll_input = K_ROLL_P * std::clamp(roll, -1.0, 1.0) + roll_velocity;
        double pitch_input = K_PITCH_P * std::clamp(pitch, -1.0, 1.0) + pitch_velocity;

        // Motor inputs, simplified calculations
        double base_thrust = K_VERTICAL_THRUST + vertical_input;

        double front_left_motor_input = std::clamp(base_thrust - roll_input + pitch_input - yaw_disturbance, -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED);
        double front_right_motor_input = std::clamp(base_thrust + roll_input + pitch_input + yaw_disturbance, -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED);
        double rear_left_motor_input = std::clamp(base_thrust - roll_input - pitch_input + yaw_disturbance, -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED);
        double rear_right_motor_input = std::clamp(base_thrust + roll_input - pitch_input - yaw_disturbance, -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED);

        const double speeds[4] = { front_left_motor_input, front_right_motor_input, rear_left_motor_input, rear_right_motor_input };

        mavic.set_motor_speed(speeds);

    }

    return 0;
}
