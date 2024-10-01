#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm> // For std::clamp


#include <webots/Robot.h>
#include <webots/Camera.h>
#include <webots/GPS.h>


#include <webots/Gyro.h>
#include <webots/Motor.h>
#include <webots/InertialUnit.h>

#define K_VERTICAL_THRUST 68.5       // Thrust value to lift the drone
#define K_VERTICAL_OFFSET 0.6        // Vertical offset for stabilization

#define K_VERTICAL_P 10              // Proportional constant for altitude PID
#define K_VERTICAL_I 0.25            // Integral constant for altitude PID
#define K_VERTICAL_D 5.0             // Derivative constant for altitude PID

#define K_ROLL_P 50.0                // Proportional constant for roll control
#define K_PITCH_P 30.0               // Proportional constant for pitch control

#define MAX_SIMULATION_TIME 99.99;   // Maximum Simulation Time
#define MAX_MOTOR_SPEED 576;         // Maximum rotor speed

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

double altitude_pid(
    double target_altitude, 
    double current_altitude,
    double timestep,
    double* integral_altitude_error,
    double* previous_altitude_error
) {
    // Calculate error
    double altitude_error = target_altitude - current_altitude;

    // Calculate integral of error
    *integral_altitude_error += altitude_error * timestep;

    // Calculate derivative of error
    double derivative_error = (altitude_error - *previous_altitude_error) / timestep;

    // Calculate PID output
    double pid_output = K_VERTICAL_P * altitude_error +
                        K_VERTICAL_I * (*integral_altitude_error) +
                        K_VERTICAL_D * derivative_error;

    // Update previous error
    *previous_altitude_error = altitude_error;

    return pid_output;
}

int main() {
    mavic = new Mavic();

    // Declare variables outside the loop
    int current_altitude_index = 0;
    double altitudes[4] = {1., 2., 3., 4.};
    double target_altitude = altitudes[current_altitude_index];

    double change_time = mavic.get_time();

    // PID control variables
    double integral_altitude_error = 0.0;
    double previous_altitude_error = 0.0;

    // Sensor values
    std::tuple<double, double, double> imu_values;
    std::tuple<double, double, double> gps_values;

    // Control inputs
    double roll, pitch, roll_input, pitch_input, yaw_input;
    double current_altitude;
    double control_signal;

    while (true) {
        double current_time = mavic.get_time();
        if (current_time >= MAX_SIMULATION_TIME)
            break;

        // Check if it is time to change altitude
        if (current_time - change_time >= MAX_STEP_LENGTH) {
            current_altitude_index = (current_altitude_index + 1) % (sizeof(altitudes) / sizeof(double));
            target_altitude = altitudes[current_altitude_index];
            change_time = current_time;
        }

        // Read Sensor values
        imu_values = mavic.get_imu_values();
        gps_values = mavic.get_gps_values();

        // Calculate roll, pitch and yaw inputs
        roll = std::get<0>(imu_values);
        pitch = std::get<1>(imu_values);
        roll_input = K_ROLL_P * std::clamp(roll, -1.0, 1.0);
        pitch_input = K_PITCH_P * std::clamp(pitch, -1.0, 1.0);
        yaw_input = 0.0; // Assuming yaw input remains constant or is calculated elsewhere

        current_altitude = std::get<2>(gps_values); // Get the altitude from GPS data

        control_signal = altitude_pid(
            target_altitude,
            current_altitude,
            mavic.timestep / 1000.0,
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

        std::tuple<double, double, double, double> rotor_speeds = std::make_tuple(
            front_left_motor_input,
            -front_right_motor_input,
            -rear_left_motor_input,
            rear_right_motor_input
        );

        // Further processing for rotor speeds can be added here
    }

    // Clean up
    delete mavic;

    return 0;
}
