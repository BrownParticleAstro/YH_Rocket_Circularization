import numpy as np
import gym
from gym import spaces

# OrbitalEnvironment simulates a 2D gravitational orbital system.
# Takes the gravitational constant (GM), initial radius (r0), initial velocity (v0), time step (dt),
# maximum simulation steps, and an optional reward function.
# Outputs the current state after each step (x, y, vx, vy) and reward.
class OrbitalEnvironment:
    def __init__(self, GM=1.0, r0=None, v0=1.0, dt=0.01, max_steps=5000, hohmann_bottleneck=False, reward_function=None):
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
        self.init_r = r0 if r0 is not None else max(0.01, np.random.uniform(0.2, 4.0))
        self.x = self.init_r
        self.y = 0.0
        self.vx = 0.0
        self.vy = np.sqrt(self.GM / self.init_r)
        self.cumm = 0.0
        self.max_steps = max_steps
        self.reward_function = reward_function or self.default_reward
        self.hohmann_bottleneck = hohmann_bottleneck
        if self.hohmann_bottleneck:
            self.bottleneck_step = self._calculate_bottleneck_timestep(1.0)
        else:
            self.bottleneck_step = max_steps
        
        self.current_step = 0
        self.reset()

    def _calculate_bottleneck_timestep(self, target_r):
        """
        Calculates the adaptive bottleneck timestep based on the Hohmann transfer time.
        Args:
            target_r: Target radius (float), typically 1.0 for the stable orbit.

        Returns:
            bottleneck_step: Number of timesteps before applying the bottleneck condition (int).
        """
        # Semi-major axis of the transfer orbit
        a = (self.init_r + target_r) / 2
        
        # Time for half of the Hohmann transfer orbit
        hohmann_time = np.pi * np.sqrt(((self.init_r + target_r)**3) / (8 * self.GM))
        
        # Convert time into timesteps and apply a 1.25x margin
        bottleneck_timestep = int(1.25 * (hohmann_time / self.dt))
        
        return bottleneck_timestep

    def reset(self):
        """
        Resets the environment to the initial state.
        Returns: Initial state as a numpy array (x, y, vx, vy).
        """
        self.x = self.init_r
        self.y = 0.0
        self.vx = 0.0
        self.vy = np.sqrt(self.GM / self.init_r)
        self.cumm = 0.0
        self.current_step = 0
        if self.hohmann_bottleneck:
            self.bottleneck_step = self._calculate_bottleneck_timestep(1.0)
        else:
            self.bottleneck_step = self.max_steps
        state = np.array([self.x, self.y, self.vx, self.vy])
        return state

    def step(self, action):
        """
        Advances the environment state by one timestep based on the provided action (tangential thrust).
        Args: action: Tangential thrust value (float).

        Returns:
            Tuple of (new state, reward, done):
            - new state: Updated state (x, y, vx, vy) as a numpy array.
            - reward: Calculated reward based on the current state and action.
            - done: Boolean indicating whether the simulation is complete.
        """
        action = np.array([0, action[0]])  # Tangential thrust only
        state = np.array([self.x, self.y, self.vx, self.vy])
        r = state[:2]
        v = state[2:]
        
        dist = np.linalg.norm(r)  # Distance from the center (0,0)
        rhat = r / dist  # Unit vector in the direction of radius
        
        # Rotate thrust vector to align with the radial direction
        rotation_matrix = np.array([[rhat[0], -rhat[1]], [rhat[1], rhat[0]]])
        thrust = rotation_matrix @ action

        dx, dy = thrust
        self.cumm += np.sqrt(dx**2 + dy**2)
        
        # Apply thrust to velocity
        self.vx += dx * self.dt
        self.vy += dy * self.dt
        
        # Gravitational acceleration based on current distance
        dist = np.sqrt(self.x**2 + self.y**2)  # Update the distance from center
        gravitational_acceleration = -self.GM / (dist**2)
        rhat = np.array([self.x, self.y]) / dist  # Recompute the unit vector in the radial direction
        acceleration = gravitational_acceleration * rhat  # Gravitational force
        
        # Update velocities due to gravitational acceleration
        self.vx += acceleration[0] * self.dt
        self.vy += acceleration[1] * self.dt
        
        # Update positions
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

        state = np.array([self.x, self.y, self.vx, self.vy])
        
        # Calculate reward based on the new state
        reward = self.reward_function(action[1])
        
        # Check if the episode is done
        done = dist > 5.0 or dist < 0.1 or \
               self.current_step >= self.max_steps or \
               (abs(1-dist) > 0.05 and self.current_step >= self.bottleneck_step)
        self.current_step += 1
        
        return state, reward, done
    
    def default_reward(self, action):
        """
        Default reward function based on radial distance from the target orbit and action penalty.
        Args: action: Tangential thrust value (float).
        Returns: Reward value (float).
        """
        r = np.sqrt(self.x**2 + self.y**2)
        reward = np.exp(- 100*(r - 1.0)**2)
        action_penalty = np.exp(- 100*action**2)
        return reward * action_penalty
    
    def set_initial_orbit(self, radius):
        """
        Manually set the initial orbital radius for the environment.
        Args: radius: Initial orbital radius (float).
        Returns: None
        """
        self.init_r = radius
        self.x = radius
        self.vy = np.sqrt(self.GM / radius)

class OrbitalEnvWrapper(gym.Env):
    def __init__(self, r0=None, max_steps=5000, bottleneck_step=5000, reward_function=None):
        """
        Args:
            r0: Initial orbital radius (float). If None, a random value is generated.
            max_steps: Maximum number of simulation steps (int).
            bottleneck_step: Number of simulation steps after which if the radial error is sufficiently large the episode is ended.
            reward_function: Optional custom reward function.
        
        Returns:
            None. Initializes the environment and action/observation spaces.
        """
        super(OrbitalEnvWrapper, self).__init__()
        self.env = OrbitalEnvironment(r0=r0, max_steps=max_steps, bottleneck_step=bottleneck_step, reward_function=reward_function)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.state = None
        self.episode_data = []

    def reset(self):
        """
        Resets the environment to the initial state and clears episode data.
        Returns: Initial observation (numpy array) after conversion by `_convert_state()`.
        """
        self.episode_data = []
        self.state = self.env.reset()
        return self._convert_state(self.state)
    
    def step(self, action):
        """
        Performs one step in the environment using the provided action.
        Args: action: Tangential thrust action (float).
        
        Returns:
            Tuple of (observation, reward, done, info):
            - observation: Next state (numpy array).
            - reward: Reward from the current step (float).
            - done: Whether the episode has finished (boolean).
            - info: Additional info dictionary (empty).
        """
        self.state, reward, done = self.env.step(action)
        self.episode_data.append([
            self.state[0],  # x
            self.state[1],  # y
            self.state[2],  # vx
            self.state[3],  # vy
            reward,         # reward
            action[0]       # action
        ])
        return self._convert_state(self.state), reward, done, {}
    
    def get_episode_data(self):
        """
        Returns: List of episode data containing [x, y, vx, vy, reward, action] for each step.
        """
        return self.episode_data
    
    def _convert_state(self, state):
        """
        Converts the raw state into a processed observation for the RL model.
        Args: state: Raw state (numpy array [x, y, vx, vy]).
        
        Returns:
            Processed observation (numpy array [1 - r, v_radial, v_tangential, initial_r, timestep, flag, specific_energy, angular_momentum]).
        """
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        v_radial = (x * vx + y * vy) / r
        v_tangential = (x * vy - y * vx) / r
        initial_r = self.env.init_r
        timestep = self.env.current_step
        flag = 1.0 if np.abs(r - 1.0) < 0.01 else 0.0
        specific_energy = 0.5 * (vx**2 + vy**2) - self.env.GM / r
        angular_momentum = r * v_tangential
        return np.array([1 - r, v_radial, v_tangential, initial_r, timestep, flag, specific_energy, angular_momentum])
