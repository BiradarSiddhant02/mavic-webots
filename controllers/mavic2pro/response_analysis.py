import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('response.csv')

# Extract the columns into separate variables
time = df.index * 1/118.65  # Assuming the Webots simulation timestep is 32 ms
target_altitude = df['a']
robot_altitude = df['b']

# Plot the target and robot altitudes
plt.figure(figsize=(10, 6))
plt.plot(time, target_altitude, label='Target Altitude', color='blue', linestyle='--')
plt.plot(time, robot_altitude, label='Robot Altitude', color='green')

# Add labels and title
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.title('Drone Altitude Response')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()
