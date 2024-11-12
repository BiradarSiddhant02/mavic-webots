import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# List and sort reward history files
latest_rewards = os.listdir('reward_histories')
latest_rewards.sort()

# Select the most recent file
latest_reward = latest_rewards[-1]

# Read the reward data
reward_df = pd.read_csv(f'reward_histories/{latest_reward}')
print(reward_df.columns)

# Sample every 100th reward
rewards = reward_df['Reward']

# Set Seaborn style for the plot
sns.set_style(style='darkgrid')

# Plot the sampled rewards
plt.figure(figsize=(10, 6))
plt.plot(rewards, color='b', label='Reward')
## Running average of 100 episodes
plt.plot(rewards.rolling(5).mean(), color='r', label='Running Average (5 episodes)')
plt.xlabel('Sampled Episode (every 100 episodes)')
plt.ylabel('Reward')
plt.title('Reward History (Sampled)')
plt.legend()
plt.show()
