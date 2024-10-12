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

void read_sensor_values(SensorData* sensor_data, Mavic& mavic)
{
    sensor_data->imu_data = mavic.get_imu_values();
    sensor_data->gps_data = mavic.get_gps_values();
    sensor_data->gyro_data = mavic.get_gyro_values();
}

int main()
{
    Mavic mavic;

    // Open the CSV file
    std::ofstream log_file("simulation_log.csv");
    if (!log_file.is_open())
    {
        std::cerr << "Failed to open file for logging." << std::endl;
        return -1;
    }

    // Write header row
    log_file << "Time,IMU_X,IMU_Y,IMU_Z,"
             << "GPS_X,GPS_Y,GPS_Z,Gyro_X,Gyro_Y,Gyro_Z,Motor_0,Motor_1,Motor_2,Motor_3\n";

    // while (wb_robot_step(mavic.timestep) != -1)
    // {
    //     if (wb_robot_get_time() > 1.0)
    //         break;
    // }

    const double target_altitude = 1.0;
    double integral_altitude_error = 0.0;
    double previous_altitude_error = 0.0;

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

        const double roll = sensor_data.imu_data[0];
        const double pitch = sensor_data.imu_data[1];
        const double altitude = sensor_data.gps_data[2];
        const double roll_velocity = sensor_data.gyro_data[0];
        const double pitch_velocity = sensor_data.gyro_data[1];

        // PID controller for altitude
        double vertical_input = altitude_pid(target_altitude, altitude, timestep_seconds,
                                             integral_altitude_error, previous_altitude_error);

        // Roll and pitch inputs with clamped values
        double roll_input = K_ROLL_P * std::clamp(roll, -1.0, 1.0) + roll_velocity;
        double pitch_input = K_PITCH_P * std::clamp(pitch, -1.0, 1.0) + pitch_velocity;

        // Base thrust calculation
        double base_thrust = K_VERTICAL_THRUST + vertical_input;

        // Motor inputs array
        double motor_inputs[4];
        motor_inputs[0] = std::clamp(base_thrust - roll_input + pitch_input, -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED); // Front left
        motor_inputs[1] = std::clamp(base_thrust + roll_input + pitch_input, -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED); // Front right
        motor_inputs[2] = std::clamp(base_thrust - roll_input - pitch_input, -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED); // Rear left
        motor_inputs[3] = std::clamp(base_thrust + roll_input - pitch_input, -MAX_MOTOR_SPEED, MAX_MOTOR_SPEED); // Rear right

        mavic.set_motor_speed(motor_inputs);

        // Log all sensor values and motor speeds every second
        if (time - last_log_time >= SAMPLING_PERIOD)
        {
            log_file << std::fixed << std::setprecision(DATA_PRECISION) // Set to fixed and 3 decimal places
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
    return 0;
}
