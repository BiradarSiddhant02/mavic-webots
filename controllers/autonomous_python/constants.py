# constants.py

# Thrust and stabilization constants
K_VERTICAL_THRUST = 68.5       # Thrust value to lift the drone
K_VERTICAL_OFFSET = 0.6         # Vertical offset for stabilization

# PID constants for altitude
K_VERTICAL_P = 10                 # Proportional constant for altitude PID
K_VERTICAL_I = 0.25              # Integral constant for altitude PID
K_VERTICAL_D = 5                 # Derivative constant for altitude PID

# Roll and pitch control constants
K_ROLL_P = 50.0                  # Proportional constant for roll control
K_PITCH_P = 30.0                 # Proportional constant for pitch control

# Motor and simulation constants
MAX_SIMULATION_TIME = 99.99
MAX_MOTOR_VELOCITY = 576.0      # Maximum motor velocity
