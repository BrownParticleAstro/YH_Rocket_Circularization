import numpy as np
import gym
from gym import spaces

# OrbitalEnvironment simulates a 2D gravitational orbital system.
# Takes the gravitational constant (GM), initial radius (r0), initial velocity (v0), time step (dt),
# maximum simulation steps, and an optional reward function.
# Outputs the current state after each step (x, y, vx, vy) and reward.
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
        self.init_r = r0 if r0 is not None else max(0.01, np.random.uniform(0.2, 4.0))
        self.x = self.init_r
        self.y = 0.0
        self.vx = 0.0
        self.vy = np.sqrt(self.GM / self.init_r)
        self.cumm = 0.0
        self.max_steps = max_steps
        self.current_step = 0
        self.reward_function = reward_function or self.default_reward
        self.reset()

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
            """ Helper function to compute the gravitational acceleration. """
            x, y = state[:2]
            dist = np.sqrt(x**2 + y**2)
            rhat = np.array([x, y]) / dist
            gravitational_acceleration = -self.GM / (dist**2)
            return gravitational_acceleration * rhat

        # RK4 position update
        state = np.array([self.x, self.y, self.vx, self.vy])

        k1_v = self.dt * acceleration(state)
        k1_p = self.dt * np.array([self.vx, self.vy])

        state_mid = state + 0.5 * np.concatenate([k1_p, k1_v])
        k2_v = self.dt * acceleration(state_mid)
        k2_p = self.dt * np.array([self.vx + 0.5 * k1_v[0], self.vy + 0.5 * k1_v[1]])

        state_mid = state + 0.5 * np.concatenate([k2_p, k2_v])
        k3_v = self.dt * acceleration(state_mid)
        k3_p = self.dt * np.array([self.vx + 0.5 * k2_v[0], self.vy + 0.5 * k2_v[1]])

        state_end = state + np.concatenate([k3_p, k3_v])
        k4_v = self.dt * acceleration(state_end)
        k4_p = self.dt * np.array([self.vx + k3_v[0], self.vy + k3_v[1]])

        # Final velocity and position update using RK4 weighted sum
        self.vx += (k1_v[0] + 2 * k2_v[0] + 2 * k3_v[0] + k4_v[0]) / 6
        self.vy += (k1_v[1] + 2 * k2_v[1] + 2 * k3_v[1] + k4_v[1]) / 6

        self.x += (k1_p[0] + 2 * k2_p[0] + 2 * k3_p[0] + k4_p[0]) / 6
        self.y += (k1_p[1] + 2 * k2_p[1] + 2 * k3_p[1] + k4_p[1]) / 6

        # Apply the tangential thrust to the velocity
        dist = np.sqrt(self.x**2 + self.y**2)
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
        Args: action: Tangential thrust value (float).
        Returns: Reward value (float).
        """
        r = np.sqrt(self.x**2 + self.y**2)
        r_err = r - 1.0
        reward = np.exp(-r_err**2)
        action_penalty = np.exp(- action**2)
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
    def __init__(self, r0=None, reward_function=None):
        """
        Args:
            r0: Initial orbital radius (float). If None, a random value is generated.
            reward_function: Optional custom reward function.
        
        Returns:
            None. Initializes the environment and action/observation spaces.
        """
        super(OrbitalEnvWrapper, self).__init__()
        self.env = OrbitalEnvironment(r0=r0, reward_function=reward_function)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.state = None
        self.episode_data = []
        self.prev_r_err = None
        self.integral_r_err = 0.0

    def reset(self):
        """
        Resets the environment to the initial state and clears episode data.
        Returns: Initial observation (numpy array) after conversion by _convert_state().
        """
        self.episode_data = []
        self.state = self.env.reset()
        self.prev_r_err = None
        self.integral_r_err = 0.0
        d_r_err = 0.0
        r = np.sqrt(self.state[0]**2 + self.state[1]**2)
        r_err = 1 - r

        return self._convert_state(self.state, d_r_err, self.integral_r_err)
    
    def step(self, action):
        """
        Performs one step in the environment using the provided action.
        Args: action: Tangential thrust action (float).

        Returns:
            Tuple of (observation, reward, done, info):
            - observation: Next state (numpy array).
            - reward: Reward from the current step (float).
            - done: Whether the episode has finished (boolean).
            - info: Additional info dictionary containing the state.
        """
        # Step through the environment
        self.state, reward, done = self.env.step(action)
        
        # Compute radial error, derivative, and integral error terms
        r = np.sqrt(self.state[0]**2 + self.state[1]**2)
        #r_err = r - 1.0 if r - 1.0 > 0.001 else 0.0
        r_err = 1 - 1.0
        if self.prev_r_err is None:
            d_r_err = 0.0
        else:
            d_r_err = (r_err - self.prev_r_err) / self.env.dt

        expected_err = max(abs(self.env.init_r - 1), 1e-2)
        self.integral_r_err += (r_err / expected_err) * self.env.dt
        self.prev_r_err = r_err

        self.integral_r_err *= 50
        d_r_err *= 50

        # Compute penalties for reward modification
        d_r_err_penalty = 1.0 
        if (r > 1 and d_r_err > 0) or (r < 1 and d_r_err < 0):
            d_r_err_penalty = np.exp(-d_r_err**2)
        #print(f"d_r_err: {d_r_err}")
        #print(f"abs(self.integral_r_err): {abs(self.integral_r_err)}")
        integral_r_err_penalty = np.exp(-abs(self.integral_r_err))
        reward *= d_r_err_penalty * integral_r_err_penalty

        # Save episode data
        self.episode_data.append([
            self.state[0],  # x
            self.state[1],  # y
            self.state[2],  # vx
            self.state[3],  # vy
            reward,         # reward
            action[0]       # action
        ])

        # Create the info dictionary including the state
        info = {
            "state": (self.state[0], self.state[1], self.state[2], self.state[3])  # x, y, vx, vy
        }
        
        # Return the observation, reward, done, and info
        return self._convert_state(self.state, d_r_err, self.integral_r_err), reward, done, info
    
    def get_episode_data(self):
        """
        Returns: List of episode data containing [x, y, vx, vy, reward, action] for each step.
        """
        return self.episode_data
    
    def _convert_state(self, state, d_r_err, integral_r_err):
        """
        Converts the raw state into a processed observation for the RL model.
        Args: state: Raw state (numpy array [x, y, vx, vy]).
              d_r_err: Derivative of radial error.
              integral_r_err: Integral of radial error.
        
        Returns:
            Processed observation (numpy array [1 - r, v_radial, v_tangential, initial_r, timestep, flag, specific_energy, angular_momentum, d_r_err, integral_r_err]).
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

        return np.array([1 - r,              # Radius error
                        v_radial,            # Radial velocity
                        v_tangential,        # Tangential velocity
                        initial_r,           # Starting radius
                        flag,                # Flag of if at one of the Hohmann thrust points
                        specific_energy,     # KE + PE
                        angular_momentum,    # Rotational momentum
                        d_r_err,             # Change in Radial error
                        integral_r_err       # Cumulative Radial error
                        ])