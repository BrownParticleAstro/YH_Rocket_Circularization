import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.colors as mcolors

# Define the OrbitalEnvironment
class OrbitalEnvironment:
    def __init__(self, GM=1.0, r0=None, v0=1.0, dt=0.01, max_steps=5000, reward_function=None):
        """
        Args:
            GM: Gravitational constant (float).
            r0: Initial orbital radius (float). If None, a random value is generated.
            v0: Initial velocity (float).
            dt: Time step for the simulation (float).
            max_steps: Maximum number of simulation steps (int).
            reward_function: Optional function for calculating rewards. Defaults to exponential radial difference.

        Returns:
            None. Initializes the orbital environment state.
        """
        self.GM = GM
        self.dt = dt
        self.init_r = r0 if r0 is not None else np.random.uniform(0.2, 4.0)
        self.enforce_r = True if r0 is not None else False
        self.x = self.init_r
        self.y = 0.0
        self.vx = 0.0
        self.vy = np.sqrt(self.GM / self.init_r)
        self.max_steps = max_steps
        self.current_step = 0
        self.reward_function = reward_function or self.default_reward
        self.reset()

    def reset(self):
        """
        Resets the environment to the initial state.
        Returns: Initial state as a numpy array (x, y, vx, vy).
        """
        self.x = self.init_r if self.enforce_r else np.random.uniform(0.2, 4.0)
        self.y = 0.0
        self.vx = 0.0
        self.vy = np.sqrt(self.GM / self.init_r)
        self.current_step = 0
        state = np.array([self.x, self.y, self.vx, self.vy])
        return state
    
    def step(self, action):
        """
        Advances the environment state by one timestep using Runge-Kutta (RK4) integration.
        Args:
            action: Tangential thrust value (float).

        Returns:
            Tuple of (new state, reward, done):
            - new state: Updated state (x, y, vx, vy) as a numpy array.
            - reward: Calculated reward based on the current state and action.
            - done: Boolean indicating whether the simulation is complete.
        """
        action = np.array([0, action[0]])  # Tangential thrust only

        def acceleration(state):
            """Helper function to compute the gravitational acceleration."""
            x, y = state[:2]
            dist = np.sqrt(x**2 + y**2)
            dist = np.clip(dist, 1e-5, 5.0)
            rhat = np.array([x, y]) / dist
            return -self.GM / (dist**2) * rhat
        
        # Current state and RK4 position update
        state = np.array([self.x, self.y, self.vx, self.vy])

        # RK4 Integration
        k1_v = self.dt * acceleration(state)
        k1_p = self.dt * state[2:4]

        state_mid = state + 0.5 * np.concatenate([k1_p, k1_v])
        k2_v = self.dt * acceleration(state_mid)
        k2_p = self.dt * state_mid[2:4]

        state_mid = state + 0.5 * np.concatenate([k2_p, k2_v])
        k3_v = self.dt * acceleration(state_mid)
        k3_p = self.dt * state_mid[2:4]

        state_end = state + np.concatenate([k3_p, k3_v])
        k4_v = self.dt * acceleration(state_end)
        k4_p = self.dt * state_end[2:4]

        # Update position and velocity
        delta_p = (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6
        delta_v = (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

        self.x += delta_p[0]
        self.y += delta_p[1]
        self.vx += delta_v[0]
        self.vy += delta_v[1]

        # Apply the tangential thrust to the velocity
        dist = np.sqrt(self.x**2 + self.y**2)
        dist = max(dist, 1e-5)  # Avoid division by zero
        rhat = np.array([self.x, self.y]) / dist
        rotation_matrix = np.array([[rhat[0], -rhat[1]], [rhat[1], rhat[0]]])
        thrust = rotation_matrix @ action
        self.vx += thrust[0] * self.dt
        self.vy += thrust[1] * self.dt

        # Update state and calculate reward
        state = np.array([self.x, self.y, self.vx, self.vy])
        reward = self.reward_function(action[1])

        # Check if the episode is done
        done = dist > 5.0 or dist < 0.1 or self.current_step >= self.max_steps
        self.current_step += 1

        return state, reward, done

    def default_reward(self, action):
        """
        Default reward function based on radial distance from the target orbit and action penalty.
        Args:
            action: Tangential thrust value (float).
        Returns:
            Reward value (float).
        """
        r = np.sqrt(self.x**2 + self.y**2)
        r_err = r - 1.0
        r_max_err = max(abs(self.init_r - 1), 1e-2)
        scaled_r_err = np.clip((r_err / r_max_err) * 2, -2, 2)

        reward = np.exp(-scaled_r_err**2)
        action_penalty = np.exp(-action**2)
        return reward * action_penalty


# Gym wrapper for the OrbitalEnvironment
class OrbitalGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(OrbitalGymEnv, self).__init__()
        self.env = OrbitalEnvironment(r0=1.0)
        # Action is a continuous value, tangential thrust
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Observation is [x, y, vx, vy]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self):
        state = self.env.reset()
        return state.astype(np.float32)

    def step(self, action):
        state, reward, done = self.env.step(action)
        return state.astype(np.float32), reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class PolarObservationWrapper(gym.ObservationWrapper):
    """
    A wrapper that converts observations from Cartesian coordinates (x,y,vx,vy)
    to polar coordinates (r, theta, v_r, v_t).

    r = sqrt(x^2 + y^2)
    theta = arctan2(y, x)
    v_r = (vx*cos(theta) + vy*sin(theta)) (radial velocity)
    v_t = (-vx*sin(theta) + vy*cos(theta)) (tangential velocity)
    """
    def __init__(self, env):
        super(PolarObservationWrapper, self).__init__(env)
        low = np.array([0.0, -np.pi, -np.inf, -np.inf], dtype=np.float32)
        high = np.array([np.inf, np.pi, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        x, y, vx, vy = obs
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        # Compute v_r and v_t
        v_r = vx*np.cos(theta) + vy*np.sin(theta)
        v_t = -vx*np.sin(theta) + vy*np.cos(theta)
        return np.array([r, theta, v_r, v_t], dtype=np.float32)


# Create the base environment
base_env = OrbitalGymEnv()

# Wrap the environment to use polar observations
env = PolarObservationWrapper(base_env)

# Train the PPO model
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=500_000)

# Now we simulate one episode and collect data for plotting in Cartesian coordinates.
# Since the wrapper changes observation, we will separately track the underlying
# Cartesian states from the base environment for visualization.

obs = env.reset()
done = False

# We'll store both polar (for reference) and cartesian (for visualization)
polar_states = []
cartesian_states = []
actions = []
values = []
rewards = []
log_probs = []
advantages = []
actor_losses = []
critic_losses = []
total_losses = []

gamma = model.gamma
gae_lambda = model.gae_lambda

while not done:
    # Convert observation to tensor
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    
    # Get action, log_prob, and value from the model
    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        action = distribution.sample().cpu().numpy().flatten()
        log_prob = distribution.log_prob(torch.tensor(action)).sum().item()
        value = model.policy.predict_values(obs_tensor).item()
    
    # Take a step in the environment
    obs_next, reward, done, info = env.step(action)

    # Store data
    polar_states.append(obs)
    cartesian_state = np.array([base_env.env.x, base_env.env.y, base_env.env.vx, base_env.env.vy], dtype=np.float32)
    cartesian_states.append(cartesian_state)
    actions.append(action)
    values.append(value)
    rewards.append(reward)
    log_probs.append(log_prob)

    obs = obs_next

# Compute advantages and actor/critic losses
values = np.array(values)
rewards = np.array(rewards)
returns = np.zeros_like(rewards)
advantages = np.zeros_like(rewards)

last_gae_lam = 0
for t in reversed(range(len(rewards))):
    if t == len(rewards) - 1:
        next_non_terminal = 1.0 - done
        next_value = 0
    else:
        next_non_terminal = 1.0
        next_value = values[t + 1]
    delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
    advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
returns = advantages + values

if callable(model.clip_range):
    clip_range_value = model.clip_range(1.0)
else:
    clip_range_value = model.clip_range

log_probs = np.array(log_probs)
for t in range(len(rewards)):
    ratio = 1.0  # Assuming no off-policy updates
    surr1 = ratio * advantages[t]
    surr2 = np.clip(ratio, 1 - clip_range_value, 1 + clip_range_value) * advantages[t]
    actor_loss = -np.minimum(surr1, surr2)
    critic_loss = (returns[t] - values[t]) ** 2
    total_loss = actor_loss + model.vf_coef * critic_loss
    actor_losses.append(actor_loss)
    critic_losses.append(critic_loss)
    total_losses.append(total_loss)

# Initialize the plot
fig, ax = plt.subplots(figsize=(8, 6))

game_space_bounds = 5.0
num_points = 100
x_vals = np.linspace(-game_space_bounds, game_space_bounds, num_points)
y_vals = np.linspace(-game_space_bounds, game_space_bounds, num_points)
xv, yv = np.meshgrid(x_vals, y_vals)
values_grid = np.zeros_like(xv)

value_min, value_max = -1, 1
levels = np.linspace(value_min, value_max, 20)

heatmap = ax.contourf(xv, yv, values_grid, levels=levels, cmap='viridis')
colorbar = fig.colorbar(heatmap, ax=ax, label='Value Function')

trajectory_x = []
trajectory_y = []

def animate(i):
    ax.clear()

    # Use cartesian_states for visualization
    obs_cart = cartesian_states[i]
    action = actions[i]
    value = values[i]
    actor_loss = actor_losses[i]
    critic_loss = critic_losses[i]
    total_loss = total_losses[i]

    x_current, y_current, vx_current, vy_current = obs_cart
    trajectory_x.append(x_current)
    trajectory_y.append(y_current)

    # Update the value function predictions on the grid
    for idx in np.ndindex(xv.shape):
        state = np.array([xv[idx], yv[idx], 0, 0], dtype=np.float32)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            predicted_value = model.policy.predict_values(state_tensor).item()
            if np.isnan(predicted_value):
                predicted_value = value_min
            values_grid[idx] = np.clip(predicted_value, value_min, value_max)

    heatmap = ax.contourf(xv, yv, values_grid, levels=levels, cmap='viridis')

    # Plot the trajectory
    ax.plot(trajectory_x, trajectory_y, 'b-', label='Trajectory')

    # Plot current position
    ax.plot(x_current, y_current, 'ro', label='Current Position')

    # Compute thrust vector for plotting
    dist = np.sqrt(x_current**2 + y_current**2)
    dist = max(dist, 1e-5)
    rhat = np.array([x_current, y_current]) / dist
    rotation_matrix = np.array([[rhat[0], -rhat[1]], [rhat[1], rhat[0]]])
    action_vector = rotation_matrix @ np.array([0, action[0]])

    ax.arrow(x_current, y_current, action_vector[0], action_vector[1],
             head_width=0.05, head_length=0.1, fc='r', ec='r', label='Action')

    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-game_space_bounds, game_space_bounds)
    ax.set_ylim(-game_space_bounds, game_space_bounds)
    ax.set_title(f'Timestep {i+1}')

    ax.text(0.05, 0.95, f'Actor Loss: {actor_loss:.4f}\nCritic Loss: {critic_loss:.4f}\nTotal Loss: {total_loss:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ani = animation.FuncAnimation(fig, animate, frames=len(cartesian_states), interval=200)

plt.show()
