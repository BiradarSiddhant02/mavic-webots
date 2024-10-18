import pandas as pd
import matplotlib.pyplot as plt

# Load the control log data
control_log_file = pd.read_csv("error_log.csv")

# Extract time and relevant data for plotting
time = control_log_file['Time']
altitude_error = control_log_file['Altitude_Error']
integral_altitude_error = control_log_file['Integral_Altitude_Error']
derivative_altitude_error = control_log_file['Derivative_Altitude_Error']
control_signal = control_log_file['Control_Signal']

# Create a figure and axis for the plots
plt.figure(figsize=(14, 10))

# Plot altitude error
plt.subplot(4, 1, 1)
plt.plot(time, altitude_error, label='Altitude Error', color='red')
plt.title('Altitude Error')
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.legend()
plt.grid()

# Plot integral of altitude error
plt.subplot(4, 1, 2)
plt.plot(time, integral_altitude_error, label='Integral of Altitude Error', color='green')
plt.title('Integral Altitude Error')
plt.xlabel('Time (s)')
plt.ylabel('Integral Error')
plt.legend()
plt.grid()

# Plot derivative of altitude error
plt.subplot(4, 1, 3)
plt.plot(time, derivative_altitude_error, label='Derivative of Altitude Error', color='blue')
plt.title('Derivative Altitude Error')
plt.xlabel('Time (s)')
plt.ylabel('Derivative Error')
plt.legend()
plt.grid()

# Plot control signal
plt.subplot(4, 1, 4)
plt.plot(time, control_signal, label='Control Signal', color='purple')
plt.title('Control Signal')
plt.xlabel('Time (s)')
plt.ylabel('Control Effort')
plt.legend()
plt.grid()

# Adjust layout and show plot
plt.tight_layout()
plt.show()
