#include "Mavic.hpp"

#include <iostream>
#include <tuple>
#include <limits>

#include <webots/Robot.hpp>
#include <webots/Camera.hpp>
#include <webots/GPS.hpp>
#include <webots/Gyro.hpp>
#include <webots/InertialUnit.hpp>

Mavic::Mavic() {
    // Initialize Robot
    timestep = getBasicTimeStep();

    // Get Camera
    camera = getCamera("camera");
    camera->enable(timestep);

    // Get IMU
    imu = getInertialUnit("inertial unit");
    imu->enable(timestep);

    // Get Gyro
    gyro = getGyro("gyro");
    gyro->enable(timestep);

    // Get GPS
    gps = getGPS("gps");
    gps->enable(timestep);

    // Get Propellers
    front_left_motor = getMotor("front left propeller");
    front_left_motor->setPosition(std::numeric_limits<double>::infinity());
    front_left_motor->setVelocity(1.);

    front_right_motor = getMotor("front right propeller");
    front_right_motor->setPosition(std::numeric_limits<double>::infinity());
    front_right_motor->setVelocity(1.);

    rear_left_motor = getMotor("rear left propeller");
    rear_left_motor->setPosition(std::numeric_limits<double>::infinity());
    rear_left_motor->setVelocity(1.);

    rear_right_motor = getMotor("rear right propeller");
    rear_right_motor->setPosition(std::numeric_limits<double>::infinity());
    rear_left_motor->setVelocity(1.);

}

const double* Mavic::get_imu_values() {
    return imu->getRollPitchYaw();
}

const double* Mavic::get_gps_values() {
    return gps->getValues();
}

const double* Mavic::get_gyro_values() {
    return gyro->getValues();
}

double Mavic::get_time() {
    return getTime();
}

void Mavic::set_rotor_speed(const double* rotor_speeds) {
    double fl_speed, fr_speed, rl_speed, rr_speed;
    fl_speed = rotor_speeds[0];
    fr_speed = rotor_speeds[1];
    rl_speed = rotor_speeds[2];
    rr_speed = rotor_speeds[3];

    front_left_motor->setVelocity(fl_speed);
    front_right_motor->setVelocity(fr_speed);
    rear_left_motor->setVelocity(rl_speed);
    rear_right_motor->setVelocity(rr_speed);
}