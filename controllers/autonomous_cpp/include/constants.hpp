#ifndef CONSTANTS_H
#define CONSTANTS_H

#define K_VERTICAL_THRUST 68.5        // Thrust value to lift the drone
#define K_VERTICAL_OFFSET 0.6         // Vertical offset for stabilization

#define K_VERTICAL_P 10.0             // Proportional constant for altitude PID
#define K_VERTICAL_I 0.25             // Integral constant for altitude PID
#define K_VERTICAL_D 5.0              // Derivative constant for altitude PID

#define K_ROLL_P 50.0                 // Proportional constant for roll control
#define K_PITCH_P 30.0                // Proportional constant for pitch control

#define MAX_SIMULATION_TIME 10.     // Maximum Simulation Time
#define MAX_MOTOR_SPEED 576.           // Maximum rotor speed
#define MAX_STEP_LENGTH 10            // Seconds per altitude

#define SAMPLING_PERIOD .1              // Time period for simulation data sampling
#define DATA_PRECISION 4                // Precision for simulation data

#endif
