#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "constants.h"

#include <webots/robot.h>
#include <webots/camera.h>
#include <webots/compass.h>
#include <webots/gps.h>
#include <webots/gyro.h>
#include <webots/inertial_unit.h>
#include <webots/keyboard.h>
#include <webots/led.h>
#include <webots/motor.h>

#define SIGN(x) ((x) > 0) - ((x) < 0)
#define CLAMP(value, low, high) ((value) < (low) ? (low) : ((value) > (high) ? (high) : (value)))
#define MAX_SIMULATION_TIME INT_MAX

// Function to calculate the PID output for altitude
double calculate_pid_altitude(
  double target, double current, double timestep,
  double* integral_altitude, double* previous_altitude_error
) {
  // Compute the error
  double error = target - current;

  // Proportional term
  double p_term = K_VERTICAL_P * error;

  // Integral term (accumulated over time)
  *integral_altitude += error * timestep;
  double i_term = K_VERTICAL_I * (*integral_altitude);

  // Derivative term (change in error)
  double derivative = (error - (*previous_altitude_error)) / timestep;
  double d_term = K_VERTICAL_D * derivative;

  // Store the error for the next timestep
  *previous_altitude_error = error;

  // Return the total PID output
  return p_term + i_term + d_term;
}

int main(int argc, char **argv) {
  wb_robot_init();
  int timestep = (int)wb_robot_get_basic_time_step();

  // Open file for writing
  FILE *file = fopen("response.csv", "w");
  if (file == NULL) {
    printf("Error opening file!\n");
    return EXIT_FAILURE;
  }

  // Write header to the file
  fprintf(file, "a,b\n");

  // Get and enable devices.
  WbDeviceTag camera = wb_robot_get_device("camera");
  wb_camera_enable(camera, timestep);

  WbDeviceTag front_left_led = wb_robot_get_device("front left led");

  WbDeviceTag front_right_led = wb_robot_get_device("front right led");

  WbDeviceTag imu = wb_robot_get_device("inertial unit");
  wb_inertial_unit_enable(imu, timestep);

  WbDeviceTag gps = wb_robot_get_device("gps");
  wb_gps_enable(gps, timestep);

  WbDeviceTag compass = wb_robot_get_device("compass");
  wb_compass_enable(compass, timestep);

  WbDeviceTag gyro = wb_robot_get_device("gyro");
  wb_gyro_enable(gyro, timestep);

  wb_keyboard_enable(timestep);

  WbDeviceTag camera_roll_motor = wb_robot_get_device("camera roll");
  WbDeviceTag camera_pitch_motor = wb_robot_get_device("camera pitch");

  // Get propeller motors and set them to velocity mode.
  WbDeviceTag front_left_motor = wb_robot_get_device("front left propeller");
  WbDeviceTag front_right_motor = wb_robot_get_device("front right propeller");
  WbDeviceTag rear_left_motor = wb_robot_get_device("rear left propeller");
  WbDeviceTag rear_right_motor = wb_robot_get_device("rear right propeller");

  WbDeviceTag motors[4] = {front_left_motor, front_right_motor, rear_left_motor, rear_right_motor};

  for (int m = 0; m < 4; ++m) {
    wb_motor_set_position(motors[m], INFINITY);
    wb_motor_set_velocity(motors[m], 1.0);
  }

  // Wait one second.
  while (wb_robot_step(timestep) != -1) {
    if (wb_robot_get_time() > 1.0)
      break;
  }

  // Variables.
  double target_altitude = 1.0;

  // Terms used for PID calculation
  double integral_altitude_error = 0.0;
  double previous_altitude_error = 0.0;

  // Main loop
  while (wb_robot_step(timestep) != -1) {
    const double time = wb_robot_get_time();  // in seconds.
    if (time >= MAX_SIMULATION_TIME) {
      break;
    }

    // Retrieve robot position using the sensors.
    const double* roll_pitch_yaw = wb_inertial_unit_get_roll_pitch_yaw(imu);
    const double roll = roll_pitch_yaw[0];
    const double pitch = roll_pitch_yaw[1];
    const double yaw = roll_pitch_yaw[1];
    
    const double* location = wb_gps_get_values(gps);
    const double x = location[0];
    const double y = location[1];
    const double altitude = location[2];
    
    const double roll_velocity = wb_gyro_get_values(gyro)[0];
    const double pitch_velocity = wb_gyro_get_values(gyro)[1];

    // Blink the front LEDs alternatively with a 1 second rate.
    const bool led_state = ((int)time) % 2;
    wb_led_set(front_left_led, led_state);
    wb_led_set(front_right_led, !led_state);

    // Stabilize the Camera by actuating the camera motors according to the gyro feedback.
    wb_motor_set_position(camera_roll_motor, -0.115 * roll_velocity);
    wb_motor_set_position(camera_pitch_motor, -0.1 * pitch_velocity);

    // Transform the keyboard input to disturbances on the stabilization algorithm.
    double roll_disturbance = 0.0;
    double pitch_disturbance = 0.0;
    double yaw_disturbance = 0.0;

    int key = wb_keyboard_get_key();
    while (key > 0) {
      switch (key) {
        case WB_KEYBOARD_UP:
          pitch_disturbance = -2.0;
          break;
        case WB_KEYBOARD_DOWN:
          pitch_disturbance = 2.0;
          break;
        case WB_KEYBOARD_RIGHT:
          yaw_disturbance = -1.3;
          break;
        case WB_KEYBOARD_LEFT:
          yaw_disturbance = 1.3;
          break;
        case (WB_KEYBOARD_SHIFT + WB_KEYBOARD_RIGHT):
          roll_disturbance = -1.0;
          break;
        case (WB_KEYBOARD_SHIFT + WB_KEYBOARD_LEFT):
          roll_disturbance = 1.0;
          break;
        case (WB_KEYBOARD_SHIFT + WB_KEYBOARD_UP):
          target_altitude += 0.05;
          break;
        case (WB_KEYBOARD_SHIFT + WB_KEYBOARD_DOWN):
          target_altitude -= 0.05;
          break;
      }
      key = wb_keyboard_get_key();
    }

    // Write data to the file instead of printing
    fprintf(file, "%.4f,%.4f\n", target_altitude, altitude);
    fflush(file);

    // Compute the roll, pitch, yaw and vertical inputs.
    const double roll_input = K_ROLL_P * CLAMP(roll, -1.0, 1.0) + roll_velocity + roll_disturbance;
    const double pitch_input = K_PITCH_P * CLAMP(pitch, -1.0, 1.0) + pitch_velocity + pitch_disturbance;
    const double yaw_input = yaw_disturbance;

    // PID controller for altitude
    const double vertical_input = calculate_pid_altitude(
      target_altitude, 
      altitude, 
      timestep / 1000.0,
      &integral_altitude_error,
      &previous_altitude_error
    );

    // Actuate the motors taking into consideration all the computed inputs.
    const double front_left_motor_input = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input;
    const double front_right_motor_input = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input;
    const double rear_left_motor_input = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input;
    const double rear_right_motor_input = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input;
    wb_motor_set_velocity(front_left_motor, front_left_motor_input);
    wb_motor_set_velocity(front_right_motor, -front_right_motor_input);
    wb_motor_set_velocity(rear_left_motor, -rear_left_motor_input);
    wb_motor_set_velocity(rear_right_motor, rear_right_motor_input);
  }

  // Close the file
  fclose(file);

  wb_robot_cleanup();

  return EXIT_SUCCESS;
}
