// MAVIC_HPP

#ifndef MAVIC_H
#define MAVIC_H

#include <iostream>
#include <tuple>

#include <webots/robot.h>
#include <webots/camera.h>
#include <webots/gps.h>
#include <webots/gyro.h>
#include <webots/motor.h>
#include <webots/inertial_unit.h>

class Mavic{
    public:

        WbDeviceTag camera;
        WbDeviceTag gps;
        WbDeviceTag gyro;
        WbDeviceTag imu;

        WbDeviceTag front_left_motor;
        WbDeviceTag front_right_motor;
        WbDeviceTag rear_left_motor;
        WbDeviceTag rear_right_motor;

        int timestep;
        double get_time();

        Mavic();

        const double* get_gps_values();
        const double* get_gyro_values();
        const double* get_imu_values();

        const unsigned char* get_image();
        void save_image(const char* filename, int quality);

        void set_motor_speed(const double* speeds);

        int robot_step();
};

typedef struct sensor_data{
    const double* gps_data;
    const double* imu_data;
    const double* gyro_data;
} SensorData;

#endif // MAVIC_H
