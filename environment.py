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
        self.init_r = r0 if r0 is not None else np.random.uniform(0.2, 4.0)
        self.enforce_r = True if r0 is not None else False
        self.x = self.init_r
        self.y = 0.0
        self.vx = 0.0
        self.vy = np.sqrt(self.GM / self.init_r)
        #self.vy = 0.0
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
        #self.vy = 0.0
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
        
        # dr = dt * (-GM / r**2)
        # Current state and RK4 position update
        state = np.array([self.x, self.y, self.vx, self.vy])

        # RK4 Integration
        k1_v = self.dt * acceleration(state)
        k1_p = self.dt * state[2:4]

        state_mid = state + 0.5 * np.concatenate([k1_p, k1_v])
        k2_v = self.dt * acceleration(state_mid)
        k2_p = self.dt * state_mid[2:4]

        # Similarly, correct k3_p and k4_p
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
        return self._convert_state(self.state, 0.0, 0.0)

    def step(self, action):
        """
        Performs one step in the environment using the provided action.
        Args:
            action: Tangential thrust action (float).

        Returns:
            Tuple of (observation, reward, done, info):
            - observation: Next state (numpy array).
            - reward: Reward from the current step (float).
            - done: Whether the episode has finished (boolean).
            - info: Additional info dictionary containing the state.
        """
        self.state, base_reward, done = self.env.step(action)

        # Extract state variables
        x, y, vx, vy = self.state[0], self.state[1], self.state[2], self.state[3]
        r = np.sqrt(x**2 + y**2)
        t = self.env.current_step * self.env.dt  # Current time

        # Constants for Hohmann transfer
        r_initial = self.env.init_r
        r_final = 1.0
        transfer_time = np.pi * np.sqrt(((r_initial + r_final) / 2)**3 / self.env.GM)

        # Compute expected radius at current time
        if t < transfer_time:
            # On elliptical transfer orbit
            a_transfer = (r_initial + r_final) / 2
            e_transfer = (r_final - r_initial) / (r_final + r_initial)
            # Mean motion
            n_transfer = np.sqrt(self.env.GM / a_transfer**3)
            # Mean anomaly
            M = n_transfer * t
            # Eccentric anomaly approximation
            E = M  # For small eccentricities
            # True anomaly
            theta = 2 * np.arctan2(np.sqrt(1 + e_transfer) * np.sin(E / 2),
                                np.sqrt(1 - e_transfer) * np.cos(E / 2))
            # Expected radius
            r_expected = a_transfer * (1 - e_transfer**2) / (1 + e_transfer * np.cos(theta))
        else:
            # Circularize at r_final
            r_expected = r_final

        # Compute errors
        r_err = r - r_expected

        d_r_err = (r_err - self.prev_r_err) / self.env.dt if self.prev_r_err is not None else 0.0
        self.prev_r_err = r_err

        self.integral_r_err += r_err * self.env.dt

        max_timesteps_passed = self.env.max_steps * self.env.dt

        # Normalize errors
        r_err_norm = r_err / r_expected
        d_r_err_norm = d_r_err / (r_expected / transfer_time)
        int_r_err_norm = self.integral_r_err / (r_expected * max_timesteps_passed)

        # Time-dependent penalty factor
        time_factor = t / max_timesteps_passed

        # Penalty factors using exponential decay
        k1, k2, k3 = 1.0, 1.0, 1.0  # Tunable constants
        penalty_r_err = np.exp(-k1 * abs(r_err_norm) * (1 + time_factor))
        penalty_d_r_err = np.exp(-k2 * abs(d_r_err_norm) * (1 + time_factor))
        penalty_int_r_err = np.exp(-k3 * abs(int_r_err_norm) * (1 + time_factor))

        # Compute reward
        base_reward = 1.0  # Maximum possible reward per step
        reward = base_reward * penalty_r_err * penalty_d_r_err * penalty_int_r_err

        # Ensure reward is not too small
        min_reward = 0.01  # Minimal reward to avoid zero
        reward = max(reward, min_reward)

        # Heavy penalty for divergence
        if r > 2 * r_final or r < r_initial / 2:
            reward = min_reward  # Set reward to minimal value

        # Save episode data
        self.episode_data.append([
            x, y, vx, vy, reward, action[0], r_err_norm, d_r_err_norm, int_r_err_norm
        ])

        # Info dictionary
        info = {
            "state": (x, y, vx, vy),
            "r_err_norm": r_err_norm,
            "d_r_err_norm": d_r_err_norm,
            "int_r_err_norm": int_r_err_norm
        }

        # Prepare next observation
        observation = self._convert_state(self.state, d_r_err_norm, int_r_err_norm)

        return observation, reward, done, info

    def _convert_state(self, state, d_r_err_norm, int_r_err_norm):
        """
        Converts the raw state into a processed observation for the RL model.
        Args:
            state: Raw state (numpy array [x, y, vx, vy]).
            d_r_err_norm: Derivative of radial error.
            int_r_err_norm: Integral of radial error.

        Returns:
            Processed observation (numpy array [scaled_r_err, v_radial, v_tangential, initial_r, timestep, flag,
                                                specific_energy, angular_momentum, d_r_err, scaled_integral_r_err]).
        """
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        r = max(r, 1e-5)          # Avoid division by zero
        v_radial = (x * vx + y * vy) / r
        v_tangential = (x * vy - y * vx) / r
        initial_r = self.env.init_r
        flag = 1.0 if np.abs(r - 1.0) < 0.01 else 0.0
        specific_energy = 0.5 * (vx**2 + vy**2) - self.env.GM / r
        angular_momentum = r * v_tangential

        # Scaling for radial error and integral of radial error
        r_err = r - 1.0
        r_max_err = max(abs(self.env.init_r - 1), 1e-2)
        scaled_r_err = np.clip((r_err / r_max_err) * 2, -2, 2)

        # Pack processed state
        state = np.array([
            scaled_r_err,         # Scaled radius error (=1 when at init_r)
            v_radial,             # Radial velocity
            v_tangential,         # Tangential velocity
            1 - initial_r,        # Initial radial error
            flag,                 # Flag if at one of the Hohmann thrust points
            specific_energy,      # KE + PE
            angular_momentum,     # Rotational momentum
            d_r_err_norm,         # Change in radial error
            int_r_err_norm        # Scaled cumulative radial error
        ], dtype=np.float32)

        return state