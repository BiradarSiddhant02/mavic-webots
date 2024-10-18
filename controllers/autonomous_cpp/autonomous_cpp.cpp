#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iomanip>

#include <Mavic.hpp>
#include <constants.hpp>

inline double altitude_pid(double target_altitude, double current_altitude, double delta_time,
                           double &integral_altitude_error, double &previous_altitude_error,
                           double &proportional_error, double &derivative_error)
{
    proportional_error = target_altitude - current_altitude;
    integral_altitude_error += proportional_error * delta_time;
    derivative_error = (proportional_error - previous_altitude_error) / delta_time;

    double control_signal = K_VERTICAL_P * proportional_error +
                            K_VERTICAL_I * integral_altitude_error +
                            K_VERTICAL_D * derivative_error;

    previous_altitude_error = proportional_error;
    return control_signal;
}

void read_sensor_values(SensorData* sensor_data, Mavic& mavic)
{
    sensor_data->imu_data = mavic.get_imu_values();
    sensor_data->gps_data = mavic.get_gps_values();
    sensor_data->gyro_data = mavic.get_gyro_values();
}

int main()
{
    Mavic mavic;

    // Open the CSV files
    std::ofstream log_file("simulation_log.csv");
    std::ofstream error_log_file("error_log.csv");

    if (!log_file.is_open() || !error_log_file.is_open())
    {
        std::cerr << "Failed to open file for logging." << std::endl;
        return -1;
    }

    // Write headers for the main log file
    log_file << "Time,IMU_X,IMU_Y,IMU_Z,"
             << "GPS_X,GPS_Y,GPS_Z,Gyro_X,Gyro_Y,Gyro_Z,Motor_0,Motor_1,Motor_2,Motor_3\n";

    // Write headers for the error log file with PID components
    error_log_file << "Time,Altitude_Error,Integral_Altitude_Error,Derivative_Altitude_Error,Control_Signal\n";

    const double target_altitude = 1.0;
    double integral_altitude_error = 0.0;
    double previous_altitude_error = 0.0;
    double proportional_error = 0.0;
    double derivative_error = 0.0;

    const double timestep_seconds = mavic.timestep / 1000.0;
    double last_log_time = 0.0;

    // Main loop
    while (mavic.robot_step() != -1)
    {
        const double time = mavic.get_time();
        if (time >= MAX_SIMULATION_TIME)
        {
            break;
        }

        SensorData sensor_data;
        read_sensor_values(&sensor_data, mavic);

        const double altitude = sensor_data.gps_data[2];

        // PID controller for altitude
        double vertical_input = altitude_pid(target_altitude, altitude, timestep_seconds,
                                             integral_altitude_error, previous_altitude_error,
                                             proportional_error, derivative_error);

        // Log altitude errors and control signal to error log file
        error_log_file << std::fixed << std::setprecision(DATA_PRECISION)
                       << time << "," 
                       << proportional_error << "," 
                       << integral_altitude_error << ","
                       << derivative_error << ","
                       << vertical_input << "\n";

        // Other control inputs and motor calculations
        double roll = sensor_data.imu_data[0];
        double pitch = sensor_data.imu_data[1];
        double roll_velocity = sensor_data.gyro_data[0];
        double pitch_velocity = sensor_data.gyro_data[1];

        double roll_input = K_ROLL_P * std::clamp(roll, -1.0, 1.0) + roll_velocity;
        double pitch_input = K_PITCH_P * std::clamp(pitch, -1.0, 1.0) + pitch_velocity;

        double base_thrust = K_VERTICAL_THRUST + vertical_input;

        double motor_inputs[4];
        motor_inputs[0] = std::clamp(base_thrust - roll_input + pitch_input, -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED);
        motor_inputs[1] = std::clamp(base_thrust + roll_input + pitch_input, -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED);
        motor_inputs[2] = std::clamp(base_thrust - roll_input - pitch_input, -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED);
        motor_inputs[3] = std::clamp(base_thrust + roll_input - pitch_input, -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED);

        mavic.set_motor_speed(motor_inputs);

        // Log all sensor values and motor speeds every second
        if (time - last_log_time >= SAMPLING_PERIOD)
        {
            log_file << std::fixed << std::setprecision(DATA_PRECISION)
                     << time << "," 
                     << sensor_data.imu_data[0] << "," 
                     << sensor_data.imu_data[1] << "," 
                     << sensor_data.imu_data[2] << ","
                     << sensor_data.gps_data[0] << "," 
                     << sensor_data.gps_data[1] << "," 
                     << sensor_data.gps_data[2] << ","
                     << sensor_data.gyro_data[0] << "," 
                     << sensor_data.gyro_data[1] << "," 
                     << sensor_data.gyro_data[2] << ","
                     << motor_inputs[0] << "," 
                     << -motor_inputs[1] << ","
                     << -motor_inputs[2] << "," 
                     << motor_inputs[3] << "\n";

            last_log_time = time;
        }
    }

    log_file.close();
    error_log_file.close();
    return 0;
}
