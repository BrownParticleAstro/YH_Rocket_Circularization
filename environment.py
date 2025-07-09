import numpy as np
import torch
import math

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
    """
    def __init__(self, shape=(), device=torch.device("cpu")):
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.var = torch.ones(shape, dtype=torch.float64, device=device)
        self.count = 1e-4
        self.device = device

    def update(self, x):
        x_float64 = x.to(torch.float64)
        batch_mean = torch.mean(x_float64, dim=0)
        batch_var = torch.var(x_float64, dim=0, unbiased=False)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class ObservationNormalizer:
    """
    A wrapper class that applies online normalization to observations.
    """
    def __init__(self, shape, device):
        self.rms = RunningMeanStd(shape=shape, device=device)
        self.clip = 10.0
        self.device = device
        self.epsilon = 1e-8

    def __call__(self, obs, update=True):
        if update:
            self.rms.update(obs)
        normalized_obs = (obs - self.rms.mean.to(torch.float32)) / torch.sqrt(self.rms.var.to(torch.float32) + self.epsilon)
        return torch.clamp(normalized_obs, -self.clip, self.clip)

# ===================================================================
# Environment Model
# ===================================================================
class OrbitalEnvironment:
    """
    Simulates a vectorized 2D gravitational orbital system for reinforcement learning.
    MODIFIED: Implements a *single* alternating curriculum (shared across all
    usages) that first increases the allowable radius deviation, then (once every
    30 update calls) increases the allowable thrust-angle range, alternating
    between them until both reach their respective maxima.  This removes the
    need for multiple environment instances with different curricula – one
    environment object can now be progressed through training and later used at
    its final, hardest setting for evaluation/plotting.
    """
    # ============================= NEW CURRICULUM CONSTANTS =============================
    _STEPS_PER_STAGE = 30           # number of *training updates* per curriculum stage
    _MAX_RADIUS_ERR = 2.0           # maximum |r-1.0| that can be used during reset()
    _BASE_RADIUS_ERR = 0.1          # starting radius error window
    _MAX_ANGLE_RANGE = 1.5          # rad (~86°)
    _RADIUS_INC = 0.5               # |r-1| increase every radius stage (clamped)
    _ANGLE_INC = 0.3                # radian increase every angle stage (clamped)

    def __init__(self, GM=1.0, dt=0.01, max_steps=1000, num_envs=1, sim_device=None, angle_curriculum_rate=None):
        # NOTE: angle_curriculum_rate is kept for backward-compat but no longer
        # used – the new alternating curriculum supersedes it.
        self.angle_curriculum_rate = angle_curriculum_rate  # retained for backward compatibility
        self.num_envs, self.device = num_envs, sim_device or torch.device("cpu")
        self.GM, self.dt, self.max_steps = GM, dt, max_steps
        self.r_target = 1.0
        
        self.x, self.y, self.vx, self.vy = [torch.zeros(num_envs, device=self.device) for _ in range(4)]
        self.current_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        
        self.current_episode_in_loop = 0
        # NEW: curriculum stage trackers
        self.radius_stage = 0  # starts easy (BASE_RADIUS_ERR)
        self.angle_stage  = 0  # starts with 0 rad allowable angle
        
        self.prev_r, self.prev_v_radial, self.prev_eccentricity, self.prev_apoapsis = [torch.zeros(num_envs, device=self.device) for _ in range(4)]
        self.integral_r_err, self.integral_ecc_error = [torch.zeros(num_envs, device=self.device) for _ in range(2)]
        self.integral_clamp = 5.0
        self.previous_r_error = torch.zeros(num_envs, device=self.device)
        
        self.episode_rewards, self.episode_lengths = [torch.zeros(num_envs, device=self.device) for _ in range(2)]
        self.initial_radii = torch.zeros(num_envs, device=self.device)
        
        # initialise angle limit based on the new curriculum rules
        self.max_angle_range = self._compute_max_angle_range()
        
        self.reset()

    def reset(self, env_indices=None):
        indices = slice(None) if env_indices is None else torch.tensor(env_indices, device=self.device, dtype=torch.long)
        num_to_reset = self.num_envs if env_indices is None else len(env_indices)
        
        self.episode_rewards[indices], self.episode_lengths[indices] = 0.0, 0
        self.integral_r_err[indices], self.integral_ecc_error[indices] = 0.0, 0.0

        # ========================== CURRICULUM: INITIAL RADIUS ==========================
        # Radius error window grows linearly from BASE to MAX over episodes 101-400.
        epi = self.current_episode_in_loop
        radius_frac = 0.0
        if epi > 100:
            radius_frac = min((epi - 100) / 300.0, 1.0)
        max_err = self._BASE_RADIUS_ERR + radius_frac * (self._MAX_RADIUS_ERR - self._BASE_RADIUS_ERR)

        if num_to_reset == self.num_envs and self.num_envs > 1:
            # For a full-batch reset in vectorised training, spread radii evenly.
            lin = torch.linspace(-max_err, max_err, num_to_reset, device=self.device)
            init_r = 1.0 + lin
        else:
            # For partial resets fall back to random sampling.
            init_r = 1.0 + (torch.rand(num_to_reset, device=self.device) * max_err) \
                             * torch.sign(torch.randn(num_to_reset, device=self.device))
        init_r = torch.clamp(init_r, 0.2, 4.0)

        self.x[indices], self.y[indices], self.vx[indices] = init_r, 0.0, 0.0
        self.vy[indices] = torch.sqrt(self.GM / torch.clamp(init_r, min=1e-6))
        self.current_step[indices] = 0

        r, vr, vt, apo, ecc, _ = self._get_raw_state()
        self.prev_r[indices], self.prev_v_radial[indices], self.prev_eccentricity[indices], self.prev_apoapsis[indices] = r[indices], vr[indices], ecc[indices], apo[indices]
        self.previous_r_error[indices] = torch.abs(r[indices] - self.r_target)
        self.initial_radii[indices] = init_r
        
        return self._get_observation()

    def _acceleration(self, x, y):
        dist_sq = x**2 + y**2
        dist_cubed = torch.clamp(dist_sq, min=1e-9)**1.5
        inv_dist_cubed = 1.0 / dist_cubed
        ax = -self.GM * x * inv_dist_cubed
        ay = -self.GM * y * inv_dist_cubed
        return torch.stack([ax, ay], dim=-1)

    def _get_raw_state(self):
        r = torch.sqrt(self.x**2 + self.y**2)
        r_safe = torch.clamp(r, min=1e-6)
        v2 = self.vx**2 + self.vy**2
        vr = (self.x * self.vx + self.y * self.vy) / r_safe
        vt = (self.x * self.vy - self.y * self.vx) / r_safe
        energy = 0.5 * v2 - self.GM / r_safe
        h = self.x * self.vy - self.y * self.vx
        pe = -self.GM / r_safe
        a = torch.full_like(energy, float('inf'))
        is_ellip = energy < 0
        a[is_ellip] = -self.GM / (2 * energy[is_ellip])
        ecc = torch.sqrt(torch.clamp(1 + 2 * energy * h**2 / (self.GM**2), min=0))
        apo = a * (1 + ecc)
        apo[~is_ellip] = float('inf')
        return r, vr, vt, torch.clamp(apo, 0.0, 10.0), torch.clamp(ecc, 0.0, 2.0), pe

    def _get_observation(self):
        r, vr, vt, apo, ecc, _ = self._get_raw_state()
        r_err = r - self.r_target
        r_err_deriv = vr
        apo_err = apo - 1.0
        vr_deriv = (vr - self.prev_v_radial) / self.dt
        ecc_deriv = (ecc - self.prev_eccentricity) / self.dt
        self.integral_r_err += r_err * self.dt
        self.integral_ecc_error += ecc * self.dt
        self.integral_r_err.clamp_(-self.integral_clamp, self.integral_clamp)
        self.integral_ecc_error.clamp_(-self.integral_clamp, self.integral_clamp)
        self.prev_r, self.prev_v_radial, self.prev_eccentricity = r.clone(), vr.clone(), ecc.clone()
        return torch.stack([
            r_err, 
            r_err_deriv, 
            vr, 
            vr_deriv, 
            ecc, 
            ecc_deriv,
            apo, 
            apo_err, 
            self.integral_r_err, 
            self.integral_ecc_error, 
            vt
        ], dim=-1)

    def _compute_reward(self, r, vr, vt, apo, ecc, terminated, truncated):
        r_err_abs = torch.abs(r - self.r_target)
        r_prog = self.previous_r_error - r_err_abs
        self.previous_r_error = r_err_abs
        apo_rew = -0.5 * torch.abs(apo - 1.0)
        # Ideal tangential velocity for circular orbit at r=1 is sqrt(GM)=1.0 (since GM=1.0).
        vt_rew = -0.1 * torch.abs(vt - 1.0)
        ecc_rew = -0.5 * ecc
        vr_pen = -0.1 * torch.abs(vr)

        # --- Hohmann-transfer inspired radial envelope penalty ---
        init_err = torch.abs(self.initial_radii - self.r_target)
        progress = torch.clamp(self.current_step.to(r.dtype) / self.max_steps, 0.0, 1.0)
        allowed_err = init_err * (1.0 - progress)
        margin = 0.1  # tolerate small deviations above the ideal envelope
        hohmann_excess = torch.clamp(r_err_abs - (allowed_err + margin), min=0.0)
        hohmann_pen = -0.2 * hohmann_excess
        act_pen = -0.01
        total_rew = (150.0 * r_prog) + vr_pen + act_pen + 0.01 + apo_rew + ecc_rew + vt_rew + hohmann_pen
        good_state = (r_err_abs < 0.05) & \
                     (torch.abs(vr) < 0.05) & \
                     (torch.abs(apo - 1.0) < 0.05) & \
                     (ecc < 0.05) & \
                     (torch.abs(vt - 1.0) < 0.05)
        total_rew[truncated & good_state] += 5.0
        total_rew[terminated] = -2.0
        reward_components = {'r_prog': r_prog.mean().item(), 'apo_rew': apo_rew.mean().item(), 'ecc_rew': ecc_rew.mean().item(), 'vt_rew': vt_rew.mean().item(), 'hohmann_pen': hohmann_pen.mean().item()}
        return total_rew, reward_components
    
    def step(self, action):
        pos, vel = torch.stack([self.x, self.y], dim=-1), torch.stack([self.vx, self.vy], dim=-1)
        k1_v = self._acceleration(pos[:, 0], pos[:, 1]); k1_p = vel
        k2_v = self._acceleration(pos[:, 0] + 0.5*self.dt*k1_p[:, 0], pos[:, 1] + 0.5*self.dt*k1_p[:, 1]); k2_p = vel + 0.5*self.dt*k1_v
        k3_v = self._acceleration(pos[:, 0] + 0.5*self.dt*k2_p[:, 0], pos[:, 1] + 0.5*self.dt*k2_p[:, 1]); k3_p = vel + 0.5*self.dt*k2_v
        k4_v = self._acceleration(pos[:, 0] + self.dt*k3_p[:, 0], pos[:, 1] + self.dt*k3_p[:, 1]); k4_p = vel + self.dt*k3_v
        vel += (self.dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        pos += (self.dt/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        self.x, self.y, self.vx, self.vy = pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1]

        thrust_magnitude, angle_control = action[:, 0], action[:, 1].clone()
        if self.max_angle_range > 0:
            angle_control = torch.clamp(angle_control, -self.max_angle_range, self.max_angle_range)
        else:
            angle_control.zero_()

        dist = torch.sqrt(self.x**2 + self.y**2)
        safe_dist = torch.clamp(dist, min=1e-6)
        radial_vx, radial_vy = self.x / safe_dist, self.y / safe_dist
        tangent_vx, tangent_vy = -self.y / safe_dist, self.x / safe_dist
        alpha = angle_control * (np.pi / 2.0)
        cos_alpha, sin_alpha = torch.cos(alpha), torch.sin(alpha)
        thrust_vx = thrust_magnitude * (cos_alpha * tangent_vx + sin_alpha * radial_vx)
        thrust_vy = thrust_magnitude * (cos_alpha * tangent_vy + sin_alpha * radial_vy)
        self.vx += thrust_vx * self.dt
        self.vy += thrust_vy * self.dt
        
        self.current_step += 1
        obs = self._get_observation()
        r, vr, vt, apo, ecc, _ = self._get_raw_state()
        
        terminated = (r > 10.0) | (r < 0.1)
        truncated = self.current_step >= self.max_steps
        reward, reward_components = self._compute_reward(r, vr, vt, apo, ecc, terminated, truncated)
        dones = terminated | truncated
        self.episode_rewards += reward
        self.episode_lengths += 1

        info = {}
        if torch.any(dones):
            done_indices = torch.where(dones)[0]
            info = {
                'final_rewards': self.episode_rewards[done_indices].cpu().numpy(),
                'final_lengths': self.episode_lengths[done_indices].cpu().numpy(),
                'final_radius': r[done_indices].cpu().numpy(),
                'final_vr': vr[done_indices].cpu().numpy(),
                'final_vt': vt[done_indices].cpu().numpy(),
                'final_apoapsis': apo[done_indices].cpu().numpy(),
                'final_eccentricity': ecc[done_indices].cpu().numpy()
            }
            self.reset(done_indices)
        return obs, reward, dones, info, reward_components
    
    # =========================== CURRICULUM HELPERS ===================================
    def _compute_max_angle_range(self) -> float:
        """Angle grows linearly to the maximum over the first 100 update calls."""
        epi = self.current_episode_in_loop
        if epi >= 100:
            return self._MAX_ANGLE_RANGE
        return (epi / 100.0) * self._MAX_ANGLE_RANGE

    def set_curriculum_episode(self, episode_idx: int) -> None:
        """Set the internal episode counter and update angle limit accordingly."""
        self.current_episode_in_loop = int(episode_idx)
        self.max_angle_range = self._compute_max_angle_range()