// MAVIC_CPP

#include "Mavic.hpp"

#include <iostream>
#include <tuple>
#include <limits>

#include <webots/robot.h>
#include <webots/camera.h>
#include <webots/gps.h>
#include <webots/gyro.h>
#include <webots/motor.h>
#include <webots/inertial_unit.h>

Mavic::Mavic() {

    // ---Initialize robot and get the timestep---
    wb_robot_init();
    timestep = (int)wb_robot_get_basic_time_step();

    // ---Initialize all the sensors---
    // Camera
    camera = wb_robot_get_device("camera");
    wb_camera_enable(camera, timestep);
    std::cout<<"[INFO] Got Camera"<<std::endl;

    // GPS
    gps = wb_robot_get_device("gps");
    wb_gps_enable(gps, timestep);
    std::cout<<"[INFO] Got GPS"<<std::endl;

    // Gyro
    gyro = wb_robot_get_device("gyro");
    wb_gyro_enable(gyro, timestep);
    std::cout<<"[INFO] Got Gyro"<<std::endl;

    // IMU
    imu = wb_robot_get_device("inertial unit");
    wb_inertial_unit_enable(imu, timestep);
    std::cout<<"[INFO] Got IMU"<<std::endl;

    // ---Initialize all motors---
    front_left_motor = wb_robot_get_device("front left propeller");
    wb_motor_set_position(front_left_motor, INFINITY);
    wb_motor_set_velocity(front_left_motor, 1);
    std::cout<<"[INFO] Got front left motor"<<std::endl;
    
    front_right_motor = wb_robot_get_device("front right propeller");
    wb_motor_set_position(front_right_motor, INFINITY);
    wb_motor_set_velocity(front_right_motor, 1);
    std::cout<<"[INFO] Got front right motor"<<std::endl;
    
    rear_left_motor = wb_robot_get_device("rear left propeller");
    wb_motor_set_position(rear_left_motor, INFINITY);
    wb_motor_set_velocity(rear_left_motor, 1);
    std::cout<<"[INFO] Got rear left motor"<<std::endl;
    
    rear_right_motor = wb_robot_get_device("rear right propeller");
    wb_motor_set_position(rear_right_motor, INFINITY);
    wb_motor_set_velocity(rear_right_motor, 1);
    std::cout<<"[INFO] Got rear right motor"<<std::endl;

}

const double* Mavic::get_gps_values(){
    return wb_gps_get_values(gps);
}

const double* Mavic::get_gyro_values(){
    return wb_gyro_get_values(gyro);
}

const double* Mavic::get_imu_values(){
    return wb_inertial_unit_get_roll_pitch_yaw(imu);
}

const unsigned char* Mavic::get_image(){
    return wb_camera_get_image(camera);
}

void Mavic::set_motor_speed(const double* speeds){
    wb_motor_set_velocity(front_left_motor, speeds[0]);
    wb_motor_set_velocity(front_right_motor, -speeds[1]);
    wb_motor_set_velocity(rear_left_motor, -speeds[2]);
    wb_motor_set_velocity(rear_right_motor, speeds[3]);
}

int Mavic::robot_step(){
    return wb_robot_step(timestep);
}

double Mavic::get_time(){
    return wb_robot_get_time();
}

void Mavic::save_image(const char* filename, int quality){
    int ret = wb_camera_save_image(camera, filename, quality);
}