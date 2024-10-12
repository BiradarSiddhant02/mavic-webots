import pandas as pd
import matplotlib.pyplot as plt

# Load the simulation log data
log_file = pd.read_csv("simulation_log.csv")

# Extract time and relevant data for plotting
time = log_file['Time']
imu_x = log_file['IMU_X']
imu_y = log_file['IMU_Y']
imu_z = log_file['IMU_Z']
gps_x = log_file['GPS_X']
gps_y = log_file['GPS_Y']
gps_z = log_file['GPS_Z']
gyro_x = log_file['Gyro_X']
gyro_y = log_file['Gyro_Y']
gyro_z = log_file['Gyro_Z']
motor_0 = log_file['Motor_0']
motor_1 = log_file['Motor_1']
motor_2 = log_file['Motor_2']
motor_3 = log_file['Motor_3']

# Create a figure and axis for the plots
plt.figure(figsize=(14, 12))

# Plot IMU data
plt.subplot(4, 1, 1)
plt.plot(time, imu_x, label='IMU X', color='r')
plt.plot(time, imu_y, label='IMU Y', color='g')
plt.plot(time, imu_z, label='IMU Z', color='b')
plt.title('IMU Data')
plt.xlabel('Time (s)')
plt.ylabel('IMU Values')
plt.legend()
plt.grid()

# Plot GPS data
plt.subplot(4, 1, 2)
plt.plot(time, gps_x, label='GPS X', color='c')
plt.plot(time, gps_y, label='GPS Y', color='m')
plt.plot(time, gps_z, label='GPS Z', color='y')
plt.title('GPS Data')
plt.xlabel('Time (s)')
plt.ylabel('GPS Values')
plt.legend()
plt.grid()

# Plot Gyro data
plt.subplot(4, 1, 3)
plt.plot(time, gyro_x, label='Gyro X', color='purple')
plt.plot(time, gyro_y, label='Gyro Y', color='orange')
plt.plot(time, gyro_z, label='Gyro Z', color='brown')
plt.title('Gyro Data')
plt.xlabel('Time (s)')
plt.ylabel('Gyro Values')
plt.legend()
plt.grid()

# Plot Motor signals
plt.subplot(4, 1, 4)
plt.plot(time, motor_0, label='Motor 0', color='pink')
plt.plot(time, motor_1, label='Motor 1', color='teal')
plt.plot(time, motor_2, label='Motor 2', color='navy')
plt.plot(time, motor_3, label='Motor 3', color='lightgreen')
plt.title('Motor Signals')
plt.xlabel('Time (s)')
plt.ylabel('Motor Values')
plt.legend()
plt.grid()

# Adjust layout and show plot
plt.tight_layout()
plt.show()
