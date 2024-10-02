#ifndef MAVIC_H
#define MAVIC_H

#include <iostream>
#include <tuple>

#include <webots/Robot.hpp>
#include <webots/Camera.hpp>
#include <webots/GPS.hpp>
#include <webots/Gyro.hpp>
#include <webots/Motor.hpp>
#include <webots/InertialUnit.hpp>

namespace webots{
    class Robot;
    class Motor;
    class Camera;
    class GPS;
    class Gyro;
    class InertialUnit;
}

class Mavic : public webots::Robot{
    public:

        webots::Camera* camera;
        webots::GPS* gps;
        webots::Gyro* gyro;        
        webots::InertialUnit* imu;

        webots::Motor* front_left_motor;
        webots::Motor* front_right_motor;
        webots::Motor* rear_left_motor;
        webots::Motor* rear_right_motor;

        double timestep;

        Mavic();
        ~Mavic();

        const double* get_imu_values();
        const double* get_gps_values();
        const double* get_gyro_values();
        double get_time();

        void set_rotor_speed(const double* rotor_speeds);
};

#endif // MAVIC_H
