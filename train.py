import os
import datetime
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import json
import time
import matplotlib.pyplot as plt
from collections import defaultdict, deque

from environment import OrbitalEnvironment, ObservationNormalizer
from model import PolicyNetwork, ValueNetwork

# ===================================================================
# Comprehensive Training Logger
# ===================================================================
class TrainingLogger:
    """
    A comprehensive logging system for RL training diagnostics.
    It tracks various metrics, prints summary statistics to the console,
    saves logs to a JSON file, and generates diagnostic plots.
    """
    def __init__(self, log_dir="logs", save_plots=True):
        self.log_dir = log_dir
        self.save_plots = save_plots
        os.makedirs(log_dir, exist_ok=True)
        # Initialize lists to store metrics over training
        self.episode_rewards, self.episode_lengths, self.losses, self.policy_entropy = [], [], [], []
        self.gradient_norms, self.success_rates, self.value_function_error = [], [], []
        self.final_radius_stats, self.final_vr_stats = [], []
        self.reward_components_history = defaultdict(list)
        self.initial_radius_dist, self.initial_apoapsis_dist, self.initial_eccentricity_dist = [], [], []
        # Use deques for efficient calculation of rolling averages
        self.window_size = 100
        self.recent_rewards = deque(maxlen=self.window_size)
        self.recent_success = deque(maxlen=self.window_size)
        self.episode_times = []
        self.start_time = time.time()
        # Parameters for scaling the x-axis of plots correctly
        self.num_envs, self.n_steps = 1, 1

    def log_episode(self, episode, episode_data):
        """Logs the metrics from a completed batch of episodes."""
        self.episode_rewards.append(episode_data['total_reward'])
        self.episode_lengths.append(episode_data['episode_length'])
        self.losses.append(episode_data.get('loss', 0))
        self.value_function_error.append(episode_data.get('value_loss', 0))
        self.recent_rewards.append(episode_data['total_reward'])
        self.policy_entropy.append(episode_data.get('entropy', 0))
        self.gradient_norms.append(episode_data.get('gradient_norm', 0))
        success_rate = episode_data.get('success_rate', 0)
        self.success_rates.append(success_rate)
        self.recent_success.append(success_rate)
        self.final_radius_stats.append(episode_data.get('final_radius_stats', {}))
        self.final_vr_stats.append(episode_data.get('final_vr_stats', {}))
        self.initial_radius_dist.append(episode_data.get('initial_radius_dist', {}))
        self.initial_apoapsis_dist.append(episode_data.get('initial_apoapsis_dist', {}))
        self.initial_eccentricity_dist.append(episode_data.get('initial_eccentricity_dist', {}))
        for component, value in episode_data.get('reward_components', {}).items():
            self.reward_components_history[component].append(value)
        self.episode_times.append(time.time() - self.start_time)

    def print_detailed_stats(self, episode, episode_data):
        """Prints a detailed summary of the latest training batch to the console."""
        print(f"\n{'='*80}\nUPDATE BATCH {episode} - TRAINING DIAGNOSTICS\n{'='*80}")
        if len(self.recent_rewards) > 0:
            print(f"📊 RECENT PERFORMANCE (last {len(self.recent_rewards)} batches):")
            print(f"  Reward: {np.mean(self.recent_rewards):.3f} ± {np.std(self.recent_rewards):.3f}")
            print(f"  Success Rate: {np.mean(self.recent_success) * 100:.1f}%")
        print(f"🎯 TRAINING DYNAMICS:")
        print(f"  Policy Loss: {episode_data.get('loss', 0):.4f} | Value Loss: {episode_data.get('value_loss', 0):.4f}")
        print(f"  Gradient Norm: {episode_data.get('gradient_norm', 0):.4f} | Policy Entropy: {episode_data.get('entropy', 0):.4f}")
        print(f"🌍 ENVIRONMENT OUTCOMES:")
        print(f"  Avg Episode Length: {episode_data.get('episode_length', 0):.1f}")
        print(f"  Success Rate (this batch): {episode_data.get('success_rate', 0)*100:.1f}%")
        print(f"🌱 CURRICULUM STATE:")
        print(f"  Initial Radius: {episode_data['initial_radius_dist'].get('mean', 0):.3f} ± {episode_data['initial_radius_dist'].get('std', 0):.3f}")
        if 'final_radius_stats' in episode_data and episode_data['final_radius_stats']:
            print(f"📍 FINAL STATE (Truncated Envs):")
            print(f"  Final Radius: {episode_data['final_radius_stats'].get('mean', 0):.3f} ± {episode_data['final_radius_stats'].get('std', 0):.3f}")
            print(f"  Final Radial Vel: {episode_data['final_vr_stats'].get('mean', 0):.3f} ± {episode_data['final_vr_stats'].get('std', 0):.3f}")
        print(f"{'='*80}\n")

    def save_logs_and_plots(self, filename_prefix="training_log"):
        """Saves all logged data to a JSON file and generates diagnostic plots."""
        log_data = {
            'episode_rewards': self.episode_rewards, 'episode_lengths': self.episode_lengths,
            'losses': self.losses, 'value_losses': self.value_function_error,
            'success_rates': self.success_rates, 'policy_entropy': self.policy_entropy,
            'gradient_norms': self.gradient_norms, 'reward_components': dict(self.reward_components_history),
            'episode_times': self.episode_times, 'initial_radius_dist': self.initial_radius_dist,
            'initial_apoapsis_dist': self.initial_apoapsis_dist, 'initial_eccentricity_dist': self.initial_eccentricity_dist,
        }
        with open(f"{self.log_dir}/{filename_prefix}.json", 'w') as f:
            json.dump(log_data, f, indent=2)
        if self.save_plots and len(self.episode_rewards) > 1:
            self._create_diagnostic_plots(filename_prefix)

    def _create_diagnostic_plots(self, filename_prefix):
        """Generates and saves a figure with multiple diagnostic subplots."""
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        fig.suptitle(f'Training Diagnostics - {filename_prefix}', fontsize=16)
        
        # Calculate total training steps for the x-axis
        training_steps = [i * self.num_envs * self.n_steps for i in range(len(self.episode_rewards))]
        window = max(1, len(self.episode_rewards) // 20) # For moving average

        # Reward Plot
        axes[0, 0].plot(training_steps, self.episode_rewards, alpha=0.3, label='Avg Reward')
        smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(training_steps[window-1:], smoothed, 'r-', label=f'MA({window})')
        axes[0, 0].set_title('Average Reward per Episode Batch'); axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        # Success Rate Plot
        axes[0, 1].plot(training_steps, [r*100 for r in self.success_rates], 'g-')
        axes[0, 1].set_title('Success Rate'); axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].set_xlabel('Training Steps'); axes[0, 1].grid(True, alpha=0.3)

        # Loss Plot
        axes[1, 0].plot(training_steps, self.losses, 'purple', label='Policy Loss')
        axes[1, 0].plot(training_steps, self.value_function_error, 'orange', label='Value Loss')
        axes[1, 0].set_title('Training Losses'); axes[1, 0].set_yscale('log')
        axes[1, 0].set_xlabel('Training Steps'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

        # Entropy Plot (shows how much the policy is exploring)
        axes[1, 1].plot(training_steps, self.policy_entropy, 'c-')
        axes[1, 1].set_title('Policy Entropy'); axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].grid(True, alpha=0.3)

        # Episode Length Plot
        axes[2, 0].plot(training_steps, self.episode_lengths, 'm-')
        axes[2, 0].set_ylabel("Avg Steps Until Termination"); axes[2, 0].set_title('Average Episode Length')
        axes[2, 0].set_xlabel('Training Steps'); axes[2, 0].grid(True, alpha=0.3)

        # Initial Radius Distribution Plot (visualizes curriculum)
        means = [d.get('mean', 1.0) for d in self.initial_radius_dist]
        stds = [d.get('std', 0) for d in self.initial_radius_dist]
        axes[2, 1].plot(training_steps, means, 'b-', label='Mean Initial Radius')
        axes[2, 1].fill_between(training_steps, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], color='blue', alpha=0.2)
        axes[2, 1].set_title('Initial Radius Distribution (Curriculum)'); axes[2, 1].set_xlabel('Training Steps')
        axes[2, 1].legend(); axes[2, 1].grid(True, alpha=0.3)

        # Apoapsis and Eccentricity Plots (final state analysis)
        apo_means = [d.get('mean', 1.0) for d in self.initial_apoapsis_dist]
        ecc_means = [d.get('mean', 0.0) for d in self.initial_eccentricity_dist]
        axes[3, 0].plot(training_steps, apo_means, 'tab:red'); axes[3, 0].set_title('Final Apoapsis')
        axes[3, 0].set_xlabel('Training Steps'); axes[3, 0].grid(True, alpha=0.3)
        axes[3, 1].plot(training_steps, ecc_means, 'tab:green'); axes[3, 1].set_title('Final Eccentricity')
        axes[3, 1].set_xlabel('Training Steps'); axes[3, 1].grid(True, alpha=0.3)
        
        print(f"Saving diagnostic plots to {self.log_dir}/{filename_prefix}_diagnostics.png")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{self.log_dir}/{filename_prefix}_diagnostics.png", dpi=150)
        plt.close()


def train_model(save_dir, total_updates=1000):
    """
    Trains a PPO agent for the orbital environment using a custom PyTorch loop.

    Args:
        save_dir (str): The root directory to save models and logs.
        total_updates (int): The total number of policy updates to perform (number of training batches).
    
    Returns:
        tuple: A tuple containing the (trained policy network, path to the model directory, trained observation normalizer).
    """
    # --- HYPERPARAMETERS ---
    num_envs = 512          # Number of parallel environments
    n_steps = 1024          # Number of steps each environment runs before a policy update
    num_ppo_epochs = 10     # Number of times to iterate over the collected data in each update
    minibatch_size = 2048   # Size of minibatches for SGD
    lr = 3e-4               # Learning rate for the Adam optimizer
    gamma = 0.999           # Discount factor for future rewards
    gae_lambda = 0.95       # Lambda for Generalized Advantage Estimation
    clip_eps = 0.2          # Clipping parameter for the PPO policy loss
    vf_coef = 0.5           # Weight for the value function loss in the total loss
    ent_coef = 0.01         # Weight for the entropy bonus to encourage exploration
    max_grad_norm = 0.5     # Maximum norm for gradient clipping to prevent exploding gradients
    log_interval = 1        # How often (in updates) to log detailed stats
    save_interval = 50      # How often (in updates) to save model checkpoints and plots

    # --- SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = f"model_{datetime.datetime.now().strftime('%H-%M-%S_%d-%m-%Y')}"
    model_save_path = os.path.join(save_dir, model_name)
    
    logger = TrainingLogger(log_dir=model_save_path); logger.num_envs, logger.n_steps = num_envs, n_steps
    
    state_dim, action_dim = 10, 1
    env = OrbitalEnvironment(num_envs=num_envs, max_steps=1000, sim_device=device)
    obs_normalizer = ObservationNormalizer((state_dim,), device=device)
    policy_net = PolicyNetwork(state_dim, action_dim).to(device)
    value_net = ValueNetwork(state_dim).to(device)
    optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=lr)
    # GradScaler for mixed-precision training on GPU to speed things up
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # PPO Rollout Buffer Storage
    s, a, lp, r, d, v = [torch.zeros(shape, device=device) for shape in [(n_steps,num_envs,state_dim), (n_steps,num_envs,action_dim), (n_steps,num_envs), (n_steps,num_envs), (n_steps,num_envs), (n_steps,num_envs)]]

    print(f"\n🚀 Starting PPO training on {device}. Saving to {model_save_path}\n{'='*80}")
    obs = env.reset()
    
    # --- TRAINING LOOP ---
    for update in range(total_updates):
        env.current_episode_in_loop = update
        # Lists to aggregate data from completed episodes within this update batch
        finished_rewards, finished_lengths, finished_final_radii, finished_final_vrs, finished_final_apoapsis, finished_final_eccentricity = [[] for _ in range(6)]

        # --- 1. Data Collection (Rollout Phase) ---
        for step in range(n_steps):
            norm_obs = obs_normalizer(obs) # Normalize observations
            # Use autocast for potential performance boost on compatible GPUs
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                dist = policy_net(norm_obs)
                action = dist.sample() # Sample action from policy
                value = value_net(norm_obs) # Get value estimate from critic
                log_prob = dist.log_prob(action)
            
            # Execute action in the environment
            next_obs, reward, done, info, reward_components = env.step(torch.clamp(action, -0.1, 0.1))
            
            # Store transition data in the buffer
            s[step], a[step], lp[step], r[step], d[step], v[step] = norm_obs, action, log_prob.squeeze(), reward, done, value.squeeze()
            obs = next_obs

            # If an episode finished, log its data
            if info:
                finished_rewards.extend(info['final_rewards']); finished_lengths.extend(info['final_lengths'])
                finished_final_radii.extend(info['final_radius']); finished_final_vrs.extend(info['final_vr'])
                finished_final_apoapsis.extend(info['final_apoapsis']); finished_final_eccentricity.extend(info['final_eccentricity'])

        # --- 2. PPO Update Phase ---
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            # Calculate advantages using Generalized Advantage Estimation (GAE)
            norm_next_obs = obs_normalizer(obs, update=False)
            next_val = value_net(norm_next_obs).squeeze()
            adv = torch.zeros_like(r)
            last_gae = 0
            for t in reversed(range(n_steps)):
                next_non_term = 1.0 - d[t]
                next_vals = next_val if t == n_steps-1 else v[t+1]
                delta = r[t] + gamma * next_vals * next_non_term - v[t]
                adv[t] = last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
            # Returns are advantages + value estimates
            ret = adv + v

        # Flatten the batch for training
        b_s, b_a, b_lp, b_adv, b_ret = [t.reshape(-1, *t.shape[2:]) for t in (s,a)] + [t.reshape(-1) for t in (lp, adv, ret)]
        b_inds = np.arange(b_s.shape[0])
        
        # --- 3. SGD Update Loop ---
        for _ in range(num_ppo_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, b_s.shape[0], minibatch_size):
                mb_inds = b_inds[start:start+minibatch_size]
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    # Normalize advantages for the minibatch
                    mb_adv = (b_adv[mb_inds] - b_adv[mb_inds].mean()) / (b_adv[mb_inds].std() + 1e-8)
                    
                    # Recalculate policy distribution and value for the minibatch
                    new_dist = policy_net(b_s[mb_inds]); new_val = value_net(b_s[mb_inds])
                    
                    # Calculate PPO loss components
                    ent = new_dist.entropy().mean()
                    nlp = new_dist.log_prob(b_a[mb_inds])
                    ratio = torch.exp(nlp - b_lp[mb_inds].unsqueeze(-1))
                    
                    # Policy Loss (Clipped Surrogate Objective)
                    pg_loss1 = mb_adv.unsqueeze(-1) * ratio
                    pg_loss2 = mb_adv.unsqueeze(-1) * torch.clamp(ratio, 1-clip_eps, 1+clip_eps)
                    pg_loss = -torch.min(pg_loss1, pg_loss2).mean()
                    
                    # Value Loss (MSE)
                    v_loss = 0.5 * ((new_val.squeeze() - b_ret[mb_inds])**2).mean()
                    
                    # Total Loss
                    loss = pg_loss - ent_coef * ent + vf_coef * v_loss
                
                # Gradient descent step
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(list(policy_net.parameters())+list(value_net.parameters()), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

        # --- 4. Logging and Saving ---
        if update % log_interval == 0:
            grad_norm = sum(p.grad.norm().item()**2 for p in policy_net.parameters() if p.grad is not None)**0.5
            success_rate = 0.0
            if finished_final_radii:
                final_r, final_vr = np.array(finished_final_radii), np.array(finished_final_vrs)
                final_apo, final_ecc = np.array(finished_final_apoapsis), np.array(finished_final_eccentricity)
                success_mask = (np.abs(final_r - 1.0) < 0.05) & (np.abs(final_vr) < 0.05) & (np.abs(final_apo - 1.0) < 0.05) & (final_ecc < 0.05)
                success_rate = float(np.mean(success_mask))
            
            episode_data = {
                'total_reward': float(np.mean(finished_rewards)) if finished_rewards else 0.0,
                'episode_length': float(np.mean(finished_lengths)) if finished_lengths else 0.0,
                'loss': pg_loss.item(), 'value_loss': v_loss.item(), 'entropy': ent.item(), 'gradient_norm': grad_norm,
                'success_rate': success_rate,
                'final_radius_stats': {'mean': float(np.mean(final_r)), 'std': float(np.std(final_r))} if finished_final_radii else {},
                'final_vr_stats': {'mean': float(np.mean(final_vr)), 'std': float(np.std(final_vr))} if finished_final_vrs else {},
                'initial_radius_dist': {'mean': float(np.mean(env.initial_radii.cpu().numpy())), 'std': float(np.std(env.initial_radii.cpu().numpy()))},
                'initial_apoapsis_dist': {'mean': float(np.mean(final_apo)), 'std': float(np.std(final_apo))} if finished_final_apoapsis else {},
                'initial_eccentricity_dist': {'mean': float(np.mean(final_ecc)), 'std': float(np.std(final_ecc))} if finished_final_eccentricity else {},
                'reward_components': reward_components
            }
            logger.log_episode(update, episode_data)
            logger.print_detailed_stats(update, episode_data)

        if update > 0 and update % save_interval == 0:
            logger.save_logs_and_plots(f"update_{update}")
            torch.save(policy_net.state_dict(), os.path.join(model_save_path, f"policy_update_{update}.pt"))
            print(f"💾 Saved logs, model checkpoint, and plots at update {update}")

    print("\n🏁 Training completed.")
    logger.save_logs_and_plots("final")
    final_model_path = os.path.join(model_save_path, "policy_final.pt")
    torch.save(policy_net.state_dict(), final_model_path)
    
    return policy_net, model_save_path, obs_normalizer