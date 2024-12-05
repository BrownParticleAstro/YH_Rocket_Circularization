import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO

# Create and wrap the environment
env = gym.make('CartPole-v1')

# Train the PPO model
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=50_000)

# Reset the environment and unpack the observation
obs, _ = env.reset()
done = False

# Initialize the data storage lists
observations = []
actions = []
values = []
rewards = []
log_probs = []

gamma = model.gamma
gae_lambda = model.gae_lambda

# Initialize data for plotting
timesteps = []
live_values = []
live_returns = []
live_advantages = []

fig, ax = plt.subplots(figsize=(10, 6))
line_value, = ax.plot([], [], label='Predicted Value (Critic)', color='blue')
line_return, = ax.plot([], [], label='Actual Return', color='green', linestyle='--')
line_advantage, = ax.plot([], [], label='Advantage', color='purple')

ax.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Baseline (y=0)')
ax.set_xlim(0, 200)  # Maximum timesteps in CartPole is typically 200
ax.set_ylim(-10, 10)
ax.set_xlabel('Timestep in the Episode')
ax.set_ylabel('Value / Return / Advantage')
ax.set_title('Live Learning Signals')
ax.legend(loc='upper right')

# Update function for animation
def update(frame):
    global obs, done

    if done:
        return line_value, line_return, line_advantage

    # Ensure observation is valid
    if isinstance(obs, np.ndarray) and obs.shape == (4,):  # Ensure obs is valid
        obs_tensor = torch.tensor([obs], dtype=torch.float32)  # Ensure a 2D tensor
    else:
        raise ValueError(f"Unexpected observation format: {obs}")

    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        action = distribution.sample().cpu().numpy().flatten()
        log_prob = distribution.log_prob(torch.tensor(action)).sum().item()
        value = model.policy.predict_values(obs_tensor).item()

    # Step in the environment
    obs_next, reward, done, truncated, info = env.step(int(action[0]))
    done = done or truncated

    # Append data for analysis
    observations.append(obs)
    actions.append(action)
    values.append(value)
    rewards.append(reward)
    log_probs.append(log_prob)

    # Update live data
    t = len(values)  # Current timestep
    timesteps.append(t)
    live_values.append(value)
    live_returns.append(reward + gamma * (live_returns[-1] if live_returns else 0))  # Cumulative return
    live_advantages.append(reward - value)  # Advantage approximation

    # Update plot lines
    line_value.set_data(timesteps, live_values)
    line_return.set_data(timesteps, live_returns)
    line_advantage.set_data(timesteps, live_advantages)
    ax.set_xlim(0, max(10, t + 1))  # Dynamically adjust x-axis as needed

    # Render the CartPole environment
    env.render()

    return line_value, line_return, line_advantage

# Animate the plot
ani = FuncAnimation(fig, update, frames=200, blit=False, interval=50, repeat=False)

# Show the plot
plt.show()

# Close the environment
env.close()
