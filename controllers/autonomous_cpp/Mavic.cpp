#include "Mavic.h"

#include <iostream>
#include <tuple>

#include <webots/Robot.h>
#include <webots/Camera.h>
#include <webots/GPS.h>
#include <webots/Gyro.h>
#include <webots/InertialUnit.h>

Mavic::Mavic() {
    // Initialize Robot
    drone = new Robot();
    timestep = drone->getBasicTimeStep();

    // Get Camera
    camera = drone->getDevice("camera");
    camera->enable(timestep);

    // Get IMU
    imu = drone->getDevice("inertial unit");
    imu->enable(timestep);

    // Get Gyro
    gyro = drone->getDevice("gyro");
    gyro->enable(timestep);

    // Get GPS
    gps = drone->getDevice("gps");
    gps->enable(timestep);

    // Get Propellers
    front_left_motor = drone->getDevice("front left propeller");
    front_right_motor = drone->getDevice("front right propeller");
    rear_left_motor = drone->getDevice("rear left propeller");
    rear_right_motor = drone->getDevice("rear right propeller");

    motors = {
        front_left_motor,
        front_right_motor,
        rear_left_motor,
        rear_right_motor
    };

    for (auto& motor : motors) {
        motor->setPosition(INFINITY);  // Set motors to velocity mode
        motor->setVelocity(1.0);        // Initial velocity for motors
    }
}

std::tuple<double, double, double> Mavic::get_imu_values() {
    return imu->getRollPitchYaw();
}

std::tuple<double, double, double> Mavic::get_gps_values() {
    return gps->getValues();
}

std::tuple<double, double, double> Mavic::get_gyro_values() {
    return gyro->getValues();
}

double Mavic::get_time() {
    return drone->getTime();
}

void Mavic::set_rotor_speed(const std::tuple<double, double, double, double>& speed_vector) {
    double fl_speed, fr_speed, rl_speed, rr_speed;
    std::tie(fl_speed, fr_speed, rl_speed, rr_speed) = speed_vector;

    front_left_motor->setVelocity(fl_speed);
    front_right_motor->setVelocity(fr_speed);
    rear_left_motor->setVelocity(rl_speed);
    rear_right_motor->setVelocity(rr_speed);
}