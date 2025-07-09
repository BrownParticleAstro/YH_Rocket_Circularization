import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from datetime import datetime
from environment import OrbitalEnvironment
from model import PolicyNetwork, ValueNetwork
from environment import ObservationNormalizer

class TrainingLogger:
    def __init__(self, log_dir="logs", save_plots=True):
        self.log_dir, self.save_plots = log_dir, save_plots
        os.makedirs(log_dir, exist_ok=True)
        self.episode_rewards, self.episode_lengths, self.losses, self.policy_entropy = [], [], [], []
        self.gradient_norms, self.success_rates, self.value_function_error = [], [], []
        self.final_radius_stats, self.final_vr_stats, self.final_vt_stats, self.reward_components_history = [], [], [], defaultdict(list)
        self.initial_radius_dist, self.initial_apoapsis_dist, self.initial_eccentricity_dist = [], [], []
        self.recent_rewards, self.recent_success = deque(maxlen=100), deque(maxlen=100)
        self.episode_times, self.start_time, self.num_envs, self.n_steps = [], time.time(), 1, 1
        self.angle_range_history = []  # NEW: Track angle range over time

    def log_episode(self, episode, episode_data): # No changes needed
        self.episode_rewards.append(episode_data['total_reward']); self.episode_lengths.append(episode_data['episode_length']); self.losses.append(episode_data.get('loss', 0)); self.value_function_error.append(episode_data.get('value_loss', 0)); self.recent_rewards.append(episode_data['total_reward']); self.policy_entropy.append(episode_data.get('entropy', 0)); self.gradient_norms.append(episode_data.get('gradient_norm', 0)); success_rate = episode_data.get('success_rate', 0); self.success_rates.append(success_rate); self.recent_success.append(success_rate); self.final_radius_stats.append(episode_data.get('final_radius_stats', {})); self.final_vr_stats.append(episode_data.get('final_vr_stats', {})); self.final_vt_stats.append(episode_data.get('final_vt_stats', {})); self.initial_radius_dist.append(episode_data.get('initial_radius_dist', {})); self.initial_apoapsis_dist.append(episode_data.get('initial_apoapsis_dist', {})); self.initial_eccentricity_dist.append(episode_data.get('initial_eccentricity_dist', {}));
        for k, v in episode_data.get('reward_components', {}).items(): self.reward_components_history[k].append(v)
        self.episode_times.append(time.time() - self.start_time)
        self.angle_range_history.append(episode_data.get('angle_range', 0.0))  # NEW: Log angle range

    def print_detailed_stats(self, episode, episode_data): # No changes needed
        print(f"\n{'='*80}\nUPDATE BATCH {episode} - TRAINING DIAGNOSTICS\n{'='*80}")
        if len(self.recent_rewards) > 0: print(f"📊 RECENT PERFORMANCE (last {len(self.recent_rewards)} batches):\n  Reward: {np.mean(self.recent_rewards):.3f} ± {np.std(self.recent_rewards):.3f}\n  Success Rate: {np.mean(self.recent_success) * 100:.1f}%")
        print(f"🎯 TRAINING DYNAMICS:\n  Policy Loss: {episode_data.get('loss', 0):.4f} | Value Loss: {episode_data.get('value_loss', 0):.4f}\n  Gradient Norm: {episode_data.get('gradient_norm', 0):.4f} | Policy Entropy: {episode_data.get('entropy', 0):.4f}")
        print(f"🌍 ENVIRONMENT OUTCOMES:"); episode_length = episode_data.get('episode_length', 0);
        if episode_data.get('did_episodes_finish'): print(f"  Avg. Completion Length: {episode_length:.1f} steps")
        else: print(f"  Avg. Ongoing Length: {episode_length:.1f} steps (no terminations in batch)")
        print(f"  Success Rate (this batch): {episode_data.get('success_rate', 0)*100:.1f}%\n🌱 CURRICULUM STATE:\n  Initial Radius: {episode_data['initial_radius_dist'].get('mean', 0):.3f} ± {episode_data['initial_radius_dist'].get('std', 0):.3f}")
        if episode_data.get('angle_range') is not None:
            print(f"  Angle Range: ±{episode_data['angle_range']:.3f} rad (±{episode_data['angle_range'] * 180 / np.pi:.1f}°)")
        if 'final_radius_stats' in episode_data and episode_data['final_radius_stats']:
            print(f"📍 FINAL STATE (Truncated Envs):\n  Final Radius: {episode_data['final_radius_stats'].get('mean', 0):.3f} ± {episode_data['final_radius_stats'].get('std', 0):.3f}\n  Final Radial Vel: {episode_data['final_vr_stats'].get('mean', 0):.3f} ± {episode_data['final_vr_stats'].get('std', 0):.3f}\n  Final Tangential Vel: {episode_data['final_vt_stats'].get('mean', 0):.3f} ± {episode_data['final_vt_stats'].get('std', 0):.3f}\n  Final Apoapsis: {episode_data['initial_apoapsis_dist'].get('mean', 0):.3f} ± {episode_data['initial_apoapsis_dist'].get('std', 0):.3f}\n  Final Eccentricity: {episode_data['initial_eccentricity_dist'].get('mean', 0):.3f} ± {episode_data['initial_eccentricity_dist'].get('std', 0):.3f}")
        print(f"{'='*80}\n")

    def save_logs_and_plots(self, filename_prefix="training_log"): # No changes needed
        log_data = {'episode_rewards': self.episode_rewards, 'episode_lengths': self.episode_lengths, 'losses': self.losses, 'value_losses': self.value_function_error, 'success_rates': self.success_rates, 'policy_entropy': self.policy_entropy, 'gradient_norms': self.gradient_norms, 'reward_components': dict(self.reward_components_history), 'episode_times': self.episode_times, 'initial_radius_dist': self.initial_radius_dist, 'initial_apoapsis_dist': self.initial_apoapsis_dist, 'initial_eccentricity_dist': self.initial_eccentricity_dist, 'angle_range_history': self.angle_range_history}
        with open(f"{self.log_dir}/{filename_prefix}.json", 'w') as f: json.dump(log_data, f, indent=2)
        if self.save_plots and len(self.episode_rewards) > 1: self._create_diagnostic_plots(filename_prefix)

    def _create_diagnostic_plots(self, filename_prefix): # No changes needed
        fig, axes = plt.subplots(5, 2, figsize=(15, 25)); fig.suptitle(f'Training Diagnostics - {filename_prefix}', fontsize=16)
        training_steps = [i * self.num_envs * self.n_steps for i in range(len(self.episode_rewards))]; window = max(1, len(self.episode_rewards) // 20)
        axes[0, 0].plot(training_steps, self.episode_rewards, alpha=0.3, label='Avg Reward'); smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid'); axes[0, 0].plot(training_steps[window-1:], smoothed, 'r-', label=f'MA({window})'); axes[0, 0].set_title('Average Reward per Episode Batch'); axes[0, 0].set_xlabel('Training Steps'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(training_steps, [r*100 for r in self.success_rates], 'g-'); axes[0, 1].set_title('Success Rate'); axes[0, 1].set_ylabel('Success Rate (%)'); axes[0, 1].set_xlabel('Training Steps'); axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(training_steps, self.losses, 'purple', label='Policy Loss'); axes[1, 0].plot(training_steps, self.value_function_error, 'orange', label='Value Loss'); axes[1, 0].set_title('Training Losses'); axes[1, 0].set_yscale('log'); axes[1, 0].set_xlabel('Training Steps'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].plot(training_steps, self.policy_entropy, 'c-'); axes[1, 1].set_title('Policy Entropy'); axes[1, 1].set_xlabel('Training Steps'); axes[1, 1].grid(True, alpha=0.3)
        axes[2, 0].plot(training_steps, self.episode_lengths, 'm-'); axes[2, 0].set_ylabel("Avg Steps"); axes[2, 0].set_title('Average Episode Length'); axes[2, 0].set_xlabel('Training Steps'); axes[2, 0].grid(True, alpha=0.3)
        means = [d.get('mean', 1.0) for d in self.initial_radius_dist]; stds = [d.get('std', 0) for d in self.initial_radius_dist]; axes[2, 1].plot(training_steps, means, 'b-', label='Mean Initial Radius'); axes[2, 1].fill_between(training_steps, [m-s for m,s in zip(means,stds)], [m+s for m,s in zip(means,stds)], color='blue', alpha=0.2); axes[2, 1].set_title('Initial Radius Distribution (Curriculum)'); axes[2, 1].set_xlabel('Training Steps'); axes[2, 1].legend(); axes[2, 1].grid(True, alpha=0.3)
        apo_means = [d.get('mean', 1.0) for d in self.initial_apoapsis_dist]; ecc_means = [d.get('mean', 0.0) for d in self.initial_eccentricity_dist]; axes[3, 0].plot(training_steps, apo_means, 'tab:red'); axes[3, 0].set_title('Final Apoapsis'); axes[3, 0].set_xlabel('Training Steps'); axes[3, 0].grid(True, alpha=0.3); axes[3, 1].plot(training_steps, ecc_means, 'tab:green'); axes[3, 1].set_title('Final Eccentricity'); axes[3, 1].set_xlabel('Training Steps'); axes[3, 1].grid(True, alpha=0.3)
        # NEW: Add angle range progression plot
        if self.angle_range_history:
            angle_degrees = [angle * 180 / np.pi for angle in self.angle_range_history]
            axes[4, 0].plot(training_steps, angle_degrees, 'tab:orange', linewidth=2)
            axes[4, 0].set_title('Angle Range Progression (Curriculum)')
            axes[4, 0].set_xlabel('Training Steps')
            axes[4, 0].set_ylabel('Angle Range (±degrees)')
            axes[4, 0].grid(True, alpha=0.3)
            axes[4, 0].set_ylim(bottom=0)
        # Hide the unused subplot
        axes[4, 1].set_visible(False)
        print(f"Saving diagnostic plots to {self.log_dir}/{filename_prefix}_diagnostics.png"); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(f"{self.log_dir}/{filename_prefix}_diagnostics.png", dpi=150); plt.close()


def train_model(save_dir, total_updates=1, dt=0.01, step_scale_factor=1.0, angle_curriculum_rate=None, pretrained_policy=None, pretrained_normalizer=None, env=None):
    """
    Trains a PPO agent for the orbital environment.
    
    MODIFIED: Added dt, step_scale_factor, angle_curriculum_rate, pretrained_policy, and pretrained_normalizer to signature.
    Key hyperparameters (n_steps, gamma, gae_lambda) are now scaled based on these factors.
    If pretrained_policy and pretrained_normalizer are provided, training continues from those weights.
    """
    # --- HYPERPARAMETERS ---
    # Baseline parameters (for dt=0.01)
    base_n_steps = 1000
    base_gamma = 0.99
    base_gae_lambda = 0.95
    base_dt = 0.01

    # Scale parameters based on the configured DT
    # Fewer steps are needed for a larger time step to cover the same sim time.
    n_steps = int(base_n_steps * step_scale_factor)
    
    # The discount factor needs to be adjusted for the new time step duration.
    # new_gamma ^ (1/new_dt) should equal old_gamma ^ (1/old_dt)
    # This leads to new_gamma = old_gamma ^ (new_dt / old_dt)
    time_multiplier = dt / base_dt
    gamma = base_gamma ** time_multiplier
    gae_lambda = base_gae_lambda ** time_multiplier

    num_envs = 512
    num_ppo_epochs = 10
    minibatch_size = 2048
    lr = 3e-4
    clip_eps = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    max_grad_norm = 0.5
    log_interval = 1
    save_interval = 5

    # --- SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = f"model_{datetime.now().strftime('%H-%M-%S_%d-%m-%Y')}"
    model_save_path = os.path.join(save_dir, model_name)
    
    logger = TrainingLogger(log_dir=model_save_path); logger.num_envs, logger.n_steps = num_envs, n_steps
    
    state_dim, action_dim = 11, 2
    # MODIFIED: Re-use a provided environment **or** create a new one if none supplied.
    if env is None:
        env = OrbitalEnvironment(num_envs=num_envs, max_steps=n_steps, sim_device=device, dt=dt, angle_curriculum_rate=angle_curriculum_rate)
    else:
        # Ensure env has correct max_steps & dt settings for this training run.
        env.max_steps = n_steps
        env.dt = dt
        # If the caller passed an env with a different num_envs we keep it – PPO
        # will simply use whatever size is present (may be slower with 1 env).
        num_envs = env.num_envs
        logger.num_envs = num_envs
    
    # MODIFIED: Use pre-trained normalizer if provided, otherwise create new one
    if pretrained_normalizer is not None:
        obs_normalizer = pretrained_normalizer
        print(f"✅ Using pre-trained observation normalizer")
    else:
        obs_normalizer = ObservationNormalizer((state_dim,), device=device)
        print(f"✅ Created fresh observation normalizer")
    
    policy_net = PolicyNetwork(state_dim, action_dim).to(device)
    value_net = ValueNetwork(state_dim).to(device)
    
    # MODIFIED: Load pre-trained policy weights if provided
    if pretrained_policy is not None:
        policy_net.load_state_dict(pretrained_policy.state_dict())
        print(f"✅ Loaded pre-trained policy weights")
    else:
        print(f"✅ Created fresh policy network")
    
    optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    s, a, lp, r, d, v = [torch.zeros((n_steps,num_envs,*shape), device=device) for shape in [(state_dim,), (action_dim,), (), (), (), ()]]

    print(f"\n🚀 Starting PPO training on {device} with DT={dt}. Saving to {model_save_path}\n{'='*80}")
    obs = env.reset()
    
    # --- TRAINING LOOP --- (No changes needed in the core PPO logic)
    for update in range(total_updates):
        env.set_curriculum_episode(update)
        finished_rewards, finished_lengths, finished_final_radii, finished_final_vrs, finished_final_vts, finished_final_apoapsis, finished_final_eccentricity = [[] for _ in range(7)]
        
        step = 0
        all_episodes_completed = False
        while not all_episodes_completed and step < n_steps:
            norm_obs = obs_normalizer(obs)
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                dist = policy_net(norm_obs)
                action = dist.sample()
                value = value_net(norm_obs)
                log_prob = dist.log_prob(action)
            next_obs, reward, done, info, reward_components = env.step(torch.clamp(action, -0.1, 0.1))
            if step < n_steps:
                s[step], a[step], lp[step], r[step], d[step], v[step] = norm_obs, action, log_prob.sum(dim=-1), reward, done, value.squeeze()
            obs, step = next_obs, step + 1
            if info:
                finished_rewards.extend(info['final_rewards']); finished_lengths.extend(info['final_lengths']); finished_final_radii.extend(info['final_radius']); finished_final_vrs.extend(info['final_vr']); finished_final_vts.extend(info['final_vt']); finished_final_apoapsis.extend(info['final_apoapsis']); finished_final_eccentricity.extend(info['final_eccentricity'])
            all_episodes_completed = torch.all(done).item()
        
        if step < n_steps:
            for i in range(step, n_steps): s[i], a[i], lp[i], r[i], d[i], v[i] = s[step-1], a[step-1], lp[step-1], r[step-1], d[step-1], v[step-1]

        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            next_val = value_net(obs_normalizer(obs, update=False)).squeeze()
            adv = torch.zeros_like(r)
            last_gae = 0
            for t in reversed(range(n_steps)):
                next_non_term = 1.0 - d[t]
                next_vals = next_val if t == n_steps-1 else v[t+1]
                delta = r[t] + gamma * next_vals * next_non_term - v[t]
                adv[t] = last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
            ret = adv + v

        b_s, b_a, b_lp, b_adv, b_ret = [t.reshape(-1, *t.shape[2:]) for t in (s,a)] + [t.reshape(-1) for t in (lp, adv, ret)]
        b_inds = np.arange(b_s.shape[0])
        
        for _ in range(num_ppo_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, b_s.shape[0], minibatch_size):
                mb_inds = b_inds[start:start+minibatch_size]
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    mb_adv = (b_adv[mb_inds] - b_adv[mb_inds].mean()) / (b_adv[mb_inds].std() + 1e-8)
                    new_dist, new_val = policy_net(b_s[mb_inds]), value_net(b_s[mb_inds])
                    ent, nlp = new_dist.entropy().mean(), new_dist.log_prob(b_a[mb_inds])
                    ratio = torch.exp(nlp.sum(dim=-1) - b_lp[mb_inds])
                    pg_loss = -torch.min(mb_adv * ratio, mb_adv * torch.clamp(ratio, 1-clip_eps, 1+clip_eps)).mean()
                    v_loss = 0.5 * ((new_val.squeeze() - b_ret[mb_inds])**2).mean()
                    loss = pg_loss - ent_coef * ent + vf_coef * v_loss
                optimizer.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(list(policy_net.parameters())+list(value_net.parameters()), max_grad_norm); scaler.step(optimizer); scaler.update()

        if update % log_interval == 0:
            # Logging logic... (remains the same)
            grad_norm = sum(p.grad.norm().item()**2 for p in policy_net.parameters() if p.grad is not None)**0.5
            success_rate = 0.0
            if finished_final_radii:
                final_r, final_vr, final_vt_arr, final_apo, final_ecc = np.array(finished_final_radii), np.array(finished_final_vrs), np.array(finished_final_vts), np.array(finished_final_apoapsis), np.array(finished_final_eccentricity)
                success_mask = (np.abs(final_r - 1.0) < 0.05) & (np.abs(final_vr) < 0.05) & (np.abs(final_apo - 1.0) < 0.05) & (final_ecc < 0.05)
                success_rate = float(np.mean(success_mask))
            did_episodes_finish = bool(finished_lengths)
            avg_length_this_batch = float(np.mean(finished_lengths)) if did_episodes_finish else float(env.current_step.float().mean().item())
            episode_data = {'total_reward': float(np.mean(finished_rewards)) if finished_rewards else 0.0, 'episode_length': avg_length_this_batch, 'did_episodes_finish': did_episodes_finish, 'loss': pg_loss.item(), 'value_loss': v_loss.item(), 'entropy': ent.item(), 'gradient_norm': grad_norm, 'success_rate': success_rate, 'final_radius_stats': {'mean': float(np.mean(final_r)), 'std': float(np.std(final_r))} if finished_final_radii else {}, 'final_vr_stats': {'mean': float(np.mean(final_vr)), 'std': float(np.std(final_vr))} if finished_final_vrs else {}, 'final_vt_stats': {'mean': float(np.mean(final_vt_arr)), 'std': float(np.std(final_vt_arr))} if finished_final_vts else {}, 'initial_radius_dist': {'mean': float(np.mean(env.initial_radii.cpu().numpy())), 'std': float(np.std(env.initial_radii.cpu().numpy()))}, 'initial_apoapsis_dist': {'mean': float(np.mean(final_apo)), 'std': float(np.std(final_apo))} if finished_final_apoapsis else {}, 'initial_eccentricity_dist': {'mean': float(np.mean(final_ecc)), 'std': float(np.std(final_ecc))} if finished_final_eccentricity else {}, 'reward_components': reward_components, 'angle_range': float(env.max_angle_range)}
            logger.log_episode(update, episode_data)
            logger.print_detailed_stats(update, episode_data)
            
        if update > 0 and update % save_interval == 0:
            logger.save_logs_and_plots(f"update_{update}")
            torch.save(policy_net.state_dict(), os.path.join(model_save_path, f"policy_update_{update}.pt"))
            print(f"💾 Saved logs, model checkpoint, and plots at update {update}")

    print("\n🏁 Training completed.")
    logger.save_logs_and_plots("final")
    torch.save(policy_net.state_dict(), os.path.join(model_save_path, "policy_final.pt"))
    torch.save(obs_normalizer, os.path.join(model_save_path, "obs_normalizer_final.pt"))
    return policy_net, model_save_path, obs_normalizer, env