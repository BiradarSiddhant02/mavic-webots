# constants.py

# Thrust and stabilization constants
K_VERTICAL_THRUST = 68.5       # Thrust value to lift the drone
K_VERTICAL_OFFSET = 0.6         # Vertical offset for stabilization

# PID constants for altitude
K_VERTICAL_P = 10                 # Proportional constant for altitude PID
K_VERTICAL_I = 0.25              # Integral constant for altitude PID
K_VERTICAL_D = 5                 # Derivative constant for altitude PID

# PID constants for x
K_X_P = 1                 # Proportional constant for x PID
K_X_I = 0.              # Integral constant for x PID
K_X_D = 0.                # Derivative constant for x PID

# PID constants for y
K_Y_P = 1                 # Proportional constant for y PID
K_Y_I = 0.              # Integral constant for y PID
K_Y_D = 0.                 # Derivative constant for y PID

# Roll and pitch control constants
K_ROLL_P = 50.0                  # Proportional constant for roll control
K_PITCH_P = 30.0                 # Proportional constant for pitch control

# Motor and simulation constants
MAX_SIMULATION_TIME = 9.99
MAX_MOTOR_VELOCITY = 576.0      # Maximum motor velocity
