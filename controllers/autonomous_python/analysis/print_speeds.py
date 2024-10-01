import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv("Motor_speeds.csv", names=["FL", "FR", "RL", "RR"])

# Create subplots with shared y-axis
fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharey=True)

# Plot each motor speed on a different subplot
axes[0].plot(df["FL"][10:], label="Front Left", color="blue")
axes[0].legend(loc="upper right")
axes[0].set_ylabel("Speed")

axes[1].plot(df["FR"][10:], label="Front Right", color="green")
axes[1].legend(loc="upper right")
axes[1].set_ylabel("Speed")

axes[2].plot(df["RL"][10:], label="Rear Left", color="red")
axes[2].legend(loc="upper right")
axes[2].set_ylabel("Speed")

axes[3].plot(df["RR"][10:], label="Rear Right", color="purple")
axes[3].legend(loc="upper right")
axes[3].set_ylabel("Speed")

# Set the common x-axis label
plt.xlabel("Time Step")

# Display the plot
plt.tight_layout()
plt.show()
