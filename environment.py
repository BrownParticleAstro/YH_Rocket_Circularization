import torch
import numpy as np

# =============================
# Set seeds for reproducibility
# =============================
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ===================================================================
# Online Normalization Helper Classes
# ===================================================================
class RunningMeanStd:
    """
    Calculates a running mean and standard deviation for online normalization.
    This is crucial for stabilizing training in environments with varying state scales.
    It uses Welford's algorithm to update statistics in a single pass.
    """
    def __init__(self, shape=()):
        """
        Initializes the running statistics.

        Args:
            shape (tuple): The shape of the data to be normalized.
        """
        # Initialize stats in float64 for better numerical stability during updates.
        self.mean = torch.zeros(shape, dtype=torch.float64)
        self.var = torch.ones(shape, dtype=torch.float64)
        self.count = 1e-4  # Small epsilon to avoid division by zero

    def update(self, x):
        """
        Updates the running mean and variance with a new batch of data.

        Args:
            x (torch.Tensor): A new batch of data.
        """
        # Cast the input tensor to float64 before calculations for precision.
        x_float64 = x.to(torch.float64)
        batch_mean = torch.mean(x_float64, dim=0)
        batch_var = torch.var(x_float64, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates statistics from pre-computed moments of a batch."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        # Combine variances using the parallel axis theorem
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class ObservationNormalizer:
    """
    A wrapper class that applies online normalization to observations using RunningMeanStd.
    """
    def __init__(self, shape, device):
        """
        Initializes the normalizer.

        Args:
            shape (tuple): The shape of the observation space.
            device (torch.device): The device the tensors are on.
        """
        self.rms = RunningMeanStd(shape=shape)
        self.clip = 10.0  # Clip observations to a reasonable range to prevent extreme values.
        self.device = device
        self.epsilon = 1e-8 # Small value to avoid division by zero in normalization.

    def __call__(self, obs, update=True):
        """
        Normalizes observations and optionally updates the running statistics.

        Args:
            obs (torch.Tensor): The observation tensor to normalize.
            update (bool): If True, update the running mean/std. Should be True for training, False for evaluation.

        Returns:
            torch.Tensor: The normalized and clipped observation.
        """
        if update:
            self.rms.update(obs.cpu()) # Statistics are updated on the CPU
        
        # Normalize using existing stats, moving them to the correct device and dtype
        normalized_obs = (obs - self.rms.mean.to(self.device, dtype=torch.float32)) / torch.sqrt(self.rms.var.to(self.device, dtype=torch.float32) + self.epsilon)
        return torch.clamp(normalized_obs, -self.clip, self.clip)

# ===================================================================
# Environment Model
# ===================================================================
class OrbitalEnvironment:
    """
    Simulates a vectorized 2D gravitational orbital system for reinforcement learning.
    This environment runs multiple simulations in parallel on a specified device (e.g., GPU)
    for efficient training. It features a PID-inspired state representation and a curriculum
    learning approach for the initial state distribution.

    Args:
        GM (float): The gravitational constant multiplied by the central mass.
        dt (float): The time step for the simulation.
        max_steps (int): The maximum number of steps per episode before truncation.
        num_envs (int): The number of parallel environments to simulate.
        sim_device (torch.device): The device (CPU or CUDA) to run the simulation on.
    """
    def __init__(self, GM=1.0, dt=0.01, max_steps=1000, num_envs=1, sim_device=None):
        self.num_envs, self.device = num_envs, sim_device or torch.device("cpu")
        self.GM, self.dt, self.max_steps = GM, dt, max_steps
        self.pe_target = -self.GM  # Target potential energy for circular orbit at radius 1.0
        
        # State variables for all environments (position and velocity)
        self.x, self.y, self.vx, self.vy = [torch.zeros(num_envs, device=self.device) for _ in range(4)]
        self.current_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        
        # For curriculum learning
        self.current_episode_in_loop = 0
        
        # State variables for PID-like features
        self.prev_r, self.prev_v_radial, self.prev_eccentricity, self.prev_apoapsis = [torch.zeros(num_envs, device=self.device) for _ in range(4)]
        self.integral_pe_error, self.integral_ecc_error = [torch.zeros(num_envs, device=self.device) for _ in range(2)]
        self.integral_clamp = 5.0
        self.previous_pe_error = torch.zeros(num_envs, device=self.device)
        
        # Tracking episode stats
        self.episode_rewards, self.episode_lengths = [torch.zeros(num_envs, device=self.device) for _ in range(2)]
        self.initial_radii = torch.zeros(num_envs, device=self.device)
        
        self.reset()

    def reset(self, env_indices=None):
        """
        Resets the specified environments to an initial state.

        Args:
            env_indices (torch.Tensor, optional): A tensor of indices for the environments to reset.
                                                  If None, all environments are reset.
        Returns:
            torch.Tensor: The initial observation for the reset environments.
        """
        indices = slice(None) if env_indices is None else torch.tensor(env_indices, device=self.device, dtype=torch.long)
        num_to_reset = self.num_envs if env_indices is None else len(env_indices)
        
        # Reset episode-specific stats
        self.episode_rewards[indices], self.episode_lengths[indices] = 0.0, 0
        self.integral_pe_error[indices], self.integral_ecc_error[indices] = 0.0, 0.0

        # Curriculum learning: Start with initial states close to the target orbit and
        # gradually increase the difficulty by sampling from a wider range of initial radii.
        max_err = min(2.0, 0.1 + ((max(0, self.current_episode_in_loop - 50)) / 800.0) * 1.9)
        init_r = 1.0 + (torch.rand(num_to_reset, device=self.device) * max_err) * torch.sign(torch.randn(num_to_reset, device=self.device))
        init_r = torch.clamp(init_r, 0.2, 4.0)

        # Initialize to a circular orbit at the sampled radius
        self.x[indices], self.y[indices], self.vx[indices] = init_r, 0.0, 0.0
        self.vy[indices] = torch.sqrt(self.GM / torch.clamp(init_r, min=1e-6))
        self.current_step[indices] = 0

        # Reset PID-related state variables
        r, vr, _, apo, ecc, pe = self._get_raw_state()
        self.prev_r[indices], self.prev_v_radial[indices], self.prev_eccentricity[indices], self.prev_apoapsis[indices] = r[indices], vr[indices], ecc[indices], apo[indices]
        self.previous_pe_error[indices] = torch.abs(pe[indices] - self.pe_target)
        self.initial_radii[indices] = init_r
        
        return self._get_observation()

    def _acceleration(self, x, y):
        """Helper function to compute gravitational acceleration."""
        dist_sq = x**2 + y**2
        dist_cubed = torch.clamp(dist_sq, min=1e-9)**1.5
        inv_dist_cubed = 1.0 / dist_cubed
        ax = -self.GM * x * inv_dist_cubed
        ay = -self.GM * y * inv_dist_cubed
        return torch.stack([ax, ay], dim=-1)

    def _get_raw_state(self):
        """Computes key orbital parameters from the Cartesian state."""
        r = torch.sqrt(self.x**2 + self.y**2)
        r_safe = torch.clamp(r, min=1e-6)
        v2 = self.vx**2 + self.vy**2
        
        # Radial and tangential velocity
        vr = (self.x * self.vx + self.y * self.vy) / r_safe
        vt = (self.x * self.vy - self.y * self.vx) / r_safe
        
        # Orbital energy and angular momentum
        energy = 0.5 * v2 - self.GM / r_safe
        h = self.x * self.vy - self.y * self.vx
        
        # Potential energy
        pe = -self.GM / r_safe
        
        # Semi-major axis (a), apoapsis, and eccentricity (e)
        a = torch.full_like(energy, float('inf'))
        is_ellip = energy < 0
        a[is_ellip] = -self.GM / (2 * energy[is_ellip])
        ecc = torch.sqrt(torch.clamp(1 + 2 * energy * h**2 / (self.GM**2), min=0))
        apo = a * (1 + ecc)
        apo[~is_ellip] = float('inf') # Apoapsis is infinite for non-elliptical orbits
        
        return r, vr, vt, torch.clamp(apo, 0.0, 10.0), torch.clamp(ecc, 0.0, 2.0), pe

    def _get_observation(self):
        """
        Constructs the observation tensor from the raw state. This state is inspired by
        PID controllers, including proportional (error), derivative (change in error),
        and integral (accumulated error) terms.
        """
        r, vr, vt, apo, ecc, pe = self._get_raw_state()
        
        # Proportional terms (current error) - using potential energy instead of radius
        pe_err = pe - self.pe_target  # Potential energy error
        apo_err = apo - 1.0
        
        # Derivative terms (rate of change of error)
        # The derivative of potential energy with respect to time is related to radial velocity
        # dU/dt = d(-GM/r)/dt = GM/r^2 * dr/dt = GM/r^2 * vr
        pe_err_deriv = self.GM / torch.clamp(r**2, min=1e-6) * vr
        vr_deriv = (vr - self.prev_v_radial) / self.dt
        ecc_deriv = (ecc - self.prev_eccentricity) / self.dt
        
        # Integral terms (accumulated error)
        self.integral_pe_error += pe_err * self.dt
        self.integral_ecc_error += ecc * self.dt
        self.integral_pe_error.clamp_(-self.integral_clamp, self.integral_clamp) # Anti-windup
        self.integral_ecc_error.clamp_(-self.integral_clamp, self.integral_clamp)
        
        # Update previous values for the next step's derivative calculation
        self.prev_r, self.prev_v_radial, self.prev_eccentricity = r.clone(), vr.clone(), ecc.clone()
        
        # The final observation vector fed to the agent
        return torch.stack([
            pe_err,              # Proportional potential energy error
            vr,                  # Radial velocity (derivative of radius)
            ecc,                 # Eccentricity (proportional error for circularity)
            apo_err,             # Apoapsis error
            pe_err_deriv,        # Derivative of potential energy error
            vr_deriv,            # Derivative of radial velocity
            ecc_deriv,           # Derivative of eccentricity
            self.integral_pe_error, # Integral of potential energy error
            self.integral_ecc_error, # Integral of eccentricity error
            vt                   # Tangential velocity
        ], dim=-1)

    def _compute_reward(self, r, vr, apo, ecc, terminated, truncated):
        """
        Calculates the reward based on the current state. The goal is to incentivize
        the agent to reach and maintain a circular orbit with the target potential energy.
        """
        pe = -self.GM / torch.clamp(r, min=1e-6)
        pe_err_abs = torch.abs(pe - self.pe_target)
        
        # 1. Progress Reward: Dense reward for reducing the potential energy error.
        pe_prog = self.previous_pe_error - pe_err_abs
        self.previous_pe_error = pe_err_abs
        
        # 2. State Penalties: Penalize deviations from the target state (circular, target potential energy).
        apo_rew = -0.5 * torch.abs(apo - 1.0) # Penalty for incorrect apoapsis
        ecc_rew = -0.5 * ecc                 # Penalty for non-zero eccentricity
        vr_pen = -0.1 * torch.abs(vr)        # Penalty for radial velocity
        
        # 3. Action Penalty: A small constant penalty to encourage efficiency.
        act_pen = -0.01
        
        # Combine reward components with weights
        total_rew = (150.0 * pe_prog) + vr_pen + act_pen + 0.01 + apo_rew + ecc_rew
        
        # 4. Success Bonus: A large bonus if the episode ends in a good state.
        good_state = (pe_err_abs < 0.05) & (torch.abs(vr) < 0.05) & (torch.abs(apo - 1.0) < 0.05) & (ecc < 0.05)
        total_rew[truncated & good_state] += 5.0
        
        # 5. Termination Penalty: A large penalty for crashing or flying away.
        total_rew[terminated] = -2.0
        
        reward_components = {'pe_prog': pe_prog.mean().item(), 'apo_rew': apo_rew.mean().item(), 'ecc_rew': ecc_rew.mean().item()}
        return total_rew, reward_components
    
    def step(self, action):
        """
        Advances the environment state by one timestep using Runge-Kutta (RK4) integration.

        Args:
            action (torch.Tensor): The tangential thrust value for each environment.

        Returns:
            tuple: A tuple containing (observation, reward, done, info, reward_components).
        """
        action = action.squeeze(-1)
        
        # RK4 Integration for gravitational forces
        pos, vel = torch.stack([self.x, self.y], dim=-1), torch.stack([self.vx, self.vy], dim=-1)
        k1_v = self._acceleration(pos[:, 0], pos[:, 1]); k1_p = vel
        k2_v = self._acceleration(pos[:, 0] + 0.5*self.dt*k1_p[:, 0], pos[:, 1] + 0.5*self.dt*k1_p[:, 1]); k2_p = vel + 0.5*self.dt*k1_v
        k3_v = self._acceleration(pos[:, 0] + 0.5*self.dt*k2_p[:, 0], pos[:, 1] + 0.5*self.dt*k2_p[:, 1]); k3_p = vel + 0.5*self.dt*k2_v
        k4_v = self._acceleration(pos[:, 0] + self.dt*k3_p[:, 0], pos[:, 1] + self.dt*k3_p[:, 1]); k4_p = vel + self.dt*k3_v
        vel += (self.dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        pos += (self.dt/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        self.x, self.y, self.vx, self.vy = pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1]
        
        # Apply tangential thrust to the velocity
        dist = torch.sqrt(self.x**2 + self.y**2)
        safe_dist = torch.clamp(dist, min=1e-6)
        self.vx += (-self.y/safe_dist * action) * self.dt
        self.vy += (self.x/safe_dist * action) * self.dt
        
        self.current_step += 1
        obs = self._get_observation()
        r, vr, _, apo, ecc, pe = self._get_raw_state()
        
        # Check termination conditions
        terminated = (r > 10.0) | (r < 0.1) # Flew away or crashed
        truncated = self.current_step >= self.max_steps # Ran out of time
        
        reward, reward_components = self._compute_reward(r, vr, apo, ecc, terminated, truncated)
        dones = terminated | truncated
        
        self.episode_rewards += reward
        self.episode_lengths += 1

        info = {}
        # If any environments are done, collect their final stats and reset them
        if torch.any(dones):
            done_indices = torch.where(dones)[0]
            info = {
                'final_rewards': self.episode_rewards[done_indices].cpu().numpy(),
                'final_lengths': self.episode_lengths[done_indices].cpu().numpy(),
                'final_radius': r[done_indices].cpu().numpy(),
                'final_vr': vr[done_indices].cpu().numpy(),
                'final_apoapsis': apo[done_indices].cpu().numpy(),
                'final_eccentricity': ecc[done_indices].cpu().numpy(),
                'final_potential_energy': pe[done_indices].cpu().numpy()
            }
            self.reset(done_indices)
            
        return obs, reward, dones, info, reward_components