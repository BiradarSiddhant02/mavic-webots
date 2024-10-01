#ifndef MAVIC_H
#define MAVIC_H

#include <iostream>
#include <tuple>

#include <webots/Robot.h>
#include <webots/Camera.h>
#include <webots/GPS.h>
#include <webots/Gyro.h>
#include <webots/Motor.h>
#include <webots/InertialUnit.h>

class Mavic{
    public:

        webots::Robot* drone;  
        webots::Camera* camera;
        webots::GPS* gps;
        webots::Gyro* gyro_sensor;        
        webots::InertialUnit* imu;

        double timestep;

        Mavic();
        ~Mavic();

        std::tuple<double, double, double> get_imu_values();
        std::tuple<double, double, double> get_gps_values();
        std::tuple<double, double, double> get_gyro_values();
        double get_time();

        void set_rotor_speed(const std::tuple<double, double, double, double>& speed_vector);
};

#endif // MAVIC_H
