import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('response.csv')

# Extract the columns into separate variables
time = df.index * 1 / 125.00  # Assuming the Webots simulation timestep is 32 ms
target_altitude = df['target_altitude']
robot_altitude = df['altitude']

# Set the Seaborn style
sns.set_theme(style='darkgrid')

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the target and robot altitudes
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
