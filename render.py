# render.py

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio
from environment import OrbitalEnvironment
from model import PolicyNetwork

plt.style.use('./rose-pine-dawn.mplstyle')

class Renderer:
    """
    Handles the rendering of post-training evaluation plots and animations.
    It takes a trained model and generates insightful visualizations of its performance.
    """
    def __init__(self, model_save_path):
        """
        Initializes the Renderer.

        Args:
            model_save_path (str): Path where evaluation outputs (plots, GIFs) will be stored.
        """
        self.model_save_path = model_save_path
        os.makedirs(self.model_save_path, exist_ok=True)

    def render_episode_to_gif(self, policy_net, obs_normalizer, env_prototype, filename, device='cpu', fps=30):
        """
        Renders multiple episodes from uniformly spaced initial radii and saves them as a single GIF.
        This provides a good qualitative sense of the policy's general behavior across a wide range of starting conditions.

        Args:
            policy_net (PolicyNetwork): The trained policy network.
            obs_normalizer (ObservationNormalizer): The trained observation normalizer.
            env_prototype (OrbitalEnvironment): A sample environment to get parameters like max_steps.
            filename (str): The full path to save the output GIF file.
            device (str): The device ('cpu' or 'cuda') to run the simulation on.
            fps (int): Frames per second for the output GIF.
        """
        print(f"🎬 Generating GIF from diverse initial radii: {filename}...")
        num_render_envs = 10 # Number of trajectories to show in the GIF
        colors = plt.cm.viridis(np.linspace(0, 1, num_render_envs))
        max_steps = 5_000
        gif_env = OrbitalEnvironment(num_envs=num_render_envs, max_steps=max_steps, sim_device=device)

        gif_env.reset()
        initial_radii = torch.linspace(0.2, 4.0, num_render_envs, device=device)
        gif_env.x = initial_radii
        gif_env.y.zero_() # Start on the x-axis
        gif_env.vx.zero_()
        gif_env.vy = torch.sqrt(gif_env.GM / torch.clamp(initial_radii, min=1e-6))

        # Re-initialize state after manually setting positions
        r, vr, _, apo, ecc, pe = gif_env._get_raw_state()
        gif_env.prev_r, gif_env.prev_v_radial, gif_env.prev_eccentricity = r.clone(), vr.clone(), ecc.clone()
        gif_env.previous_pe_error = torch.abs(pe - gif_env.pe_target)
        obs = gif_env._get_observation()

        trajectories = [[(gif_env.x[i].item(), gif_env.y[i].item())] for i in range(num_render_envs)]
        completed_episodes = torch.zeros(num_render_envs, dtype=torch.bool, device=device)

        with imageio.get_writer(filename, mode='I', fps=fps) as writer:
            last_image = None
            for step in range(gif_env.max_steps):
                plt.figure(figsize=(8, 8))
                plt.scatter(0, 0, color='yellow', s=1000, label='Central Body', zorder=5)
                angles = np.linspace(0, 2 * np.pi, 200)
                plt.plot(np.cos(angles), np.sin(angles), 'g--', label='Target Orbit (r=1.0)', zorder=1)

                for i, traj in enumerate(trajectories):
                    if not traj: continue
                    traj_np = np.array(traj)
                    plt.plot(traj_np[:, 0], traj_np[:, 1], linestyle=':', color=colors[i], zorder=2)
                    plt.plot(traj_np[-1, 0], traj_np[-1, 1], 'o', color=colors[i], markersize=8, markeredgecolor='black', zorder=3)

                plt.title("Multi-Episode Trajectory Visualization (Diverse Initial Radii)")
                plt.xlim([-4.5, 4.5]); plt.ylim([-4.5, 4.5])
                plt.gca().set_aspect('equal', adjustable='box'); plt.grid(True, alpha=0.3)

                plt.text(-4.5, 5.0, f"Step: {step+1}", fontsize=14, ha='left', va='bottom', fontweight='bold')
                
                # Get current radii for each env
                current_r, _, _, _, _, _ = gif_env._get_raw_state()
                radius_text = "Radius: " + ", ".join([f"{rad:.2f}" for rad in current_r.cpu().numpy()])
                plt.text(-4.5, 5.7, radius_text, fontsize=12, ha='left', va='top', color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

                fig = plt.gcf()
                fig.canvas.draw()
                image = np.array(fig.canvas.renderer.buffer_rgba())
                plt.close(fig)
                writer.append_data(image)
                last_image = image

                with torch.no_grad():
                    norm_obs = obs_normalizer(obs, update=False)
                    action = policy_net(norm_obs).mean
                    next_obs, _, done_tensor, _, _ = gif_env.step(torch.clamp(action, -0.1, 0.1))
                    obs = next_obs

                completed_episodes |= done_tensor
                for i in range(num_render_envs):
                    if not completed_episodes[i]:
                        trajectories[i].append((gif_env.x[i].item(), gif_env.y[i].item()))

                if completed_episodes.all(): break
                if step % 100 == 0: print(f"Step: {step}")

            if last_image is not None:
                for _ in range(fps): writer.append_data(last_image)
        print(f"✅ GIF saved to {filename}")

    def evaluate_and_plot_policy(self, policy_net, obs_normalizer, env_prototype, filename, device, init_r=None):
        """
        Runs a single deterministic episode and plots radius and action over time.
        """
        print(f"📈 Evaluating policy and plotting results to {filename}...")
        eval_env = OrbitalEnvironment(num_envs=1, max_steps=5_000, sim_device=device)
        
        if init_r is None:
            eval_env.x[0] = 0.2 + torch.rand(1, device=device) * 3.8
        else:
            eval_env.x[0] = init_r
        obs = eval_env.reset(env_indices=[0])

        radius_history, actions_history = [], []
        done = False
        
        r, _, _, _, _, pe = eval_env._get_raw_state()
        radius_history.append(r.item())
        pe_history = [pe.item()]
        
        for _ in range(eval_env.max_steps):
            if done: break
            with torch.no_grad():
                norm_obs = obs_normalizer(obs, update=False)
                action = policy_net(norm_obs).mean
            
            actions_history.append(action.item())
            obs, _, done_tensor, _, _ = eval_env.step(action)
            done = done_tensor.item()
            
            r, _, _, _, _, pe = eval_env._get_raw_state()
            radius_history.append(r.item())
            pe_history.append(pe.item())
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        fig.suptitle('Policy Evaluation: Radius, Potential Energy & Action vs. Time', fontsize=16)
        
        axes[0].plot(radius_history[:-1], label='Radius (r)', color='dodgerblue')
        axes[0].axhline(y=1.0, color='r', linestyle='--', label='Target Radius (r=1.0)')
        axes[0].set_title('Satellite Radius vs. Time'); axes[0].set_ylabel('Radius')
        axes[0].legend(); axes[0].grid(True, alpha=0.5)
        
        axes[1].plot(pe_history[:-1], label='Potential Energy (U)', color='purple')
        axes[1].axhline(y=eval_env.pe_target, color='r', linestyle='--', label=f'Target Potential Energy (U={eval_env.pe_target:.2f})')
        axes[1].set_title('Potential Energy vs. Time'); axes[1].set_ylabel('Potential Energy')
        axes[1].legend(); axes[1].grid(True, alpha=0.5)
        
        axes[2].plot(actions_history, label='Action', color='seagreen', marker='.', linestyle='-')
        axes[2].set_title('Action vs. Time'); axes[2].set_xlabel('Time Step'); axes[2].set_ylabel('Action Magnitude')
        axes[2].legend(); axes[2].grid(True, alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename)
        plt.close(fig)
        print(f"✅ Evaluation plot saved to {filename}")

    def evaluate_across_initial_radii(self, policy_net, obs_normalizer, filename, device, num_data_points=30):
        """
        Evaluates the policy's performance across a range of initial radii and plots
        final metrics, including radius, against the starting radius.
        """
        print(f"\U0001F4CA Evaluating policy across initial radii and plotting to {filename}...")
        eval_env = OrbitalEnvironment(num_envs=1, max_steps=5_000, sim_device=device)
        
        initial_radii_to_test = torch.linspace(0.2, 4.0, num_data_points, device=device)
        initial_actions, episode_lengths, final_eccentricities, final_radii, final_potential_energies = [], [], [], [], []

        for init_r_val in initial_radii_to_test:
            eval_env.x[0], eval_env.y[0], eval_env.vx[0] = init_r_val, 0.0, 0.0
            eval_env.vy[0] = torch.sqrt(eval_env.GM / torch.clamp(init_r_val, min=1e-6))
            obs = eval_env.reset(env_indices=[0])
            
            with torch.no_grad():
                norm_obs = obs_normalizer(obs, update=False)
                initial_actions.append(policy_net(norm_obs).mean.item())

            done = False
            final_r, final_ecc, final_pe = None, None, None
            
            for step in range(eval_env.max_steps):
                if done: break
                with torch.no_grad():
                    norm_obs = obs_normalizer(obs, update=False)
                    action = policy_net(norm_obs).mean
                obs, _, done_tensor, info, _ = eval_env.step(torch.clamp(action, -0.1, 0.1))
                done = done_tensor.item()
                
                if done and info and 'final_radius' in info:
                    final_r = info['final_radius'][0]
                    final_ecc = info['final_eccentricity'][0]
                    final_pe = info['final_potential_energy'][0]
            
            if final_r is None:
                r, _, _, _, ecc, pe = eval_env._get_raw_state()
                final_r, final_ecc, final_pe = r.item(), ecc.item(), pe.item()
            
            episode_lengths.append(step + 1)
            final_eccentricities.append(final_ecc)
            final_radii.append(final_r)
            final_potential_energies.append(final_pe)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12), tight_layout=True)
        fig.suptitle('Policy Performance Across Initial Radii', fontsize=16)
        
        axes[0, 0].plot(initial_radii_to_test.cpu().numpy(), initial_actions, 'o-'); axes[0, 0].set_title('Initial Action')
        axes[0, 1].plot(initial_radii_to_test.cpu().numpy(), episode_lengths, 'o-', color='tab:orange'); axes[0, 1].set_title('Episode Length')
        axes[0, 2].plot(initial_radii_to_test.cpu().numpy(), final_eccentricities, 'o-', color='tab:green'); axes[0, 2].set_title('Final Eccentricity')
        axes[1, 0].plot(initial_radii_to_test.cpu().numpy(), final_radii, 'o-', color='tab:red'); axes[1, 0].set_title('Final Radius')
        axes[1, 0].axhline(y=1.0, color='black', linestyle='--', label='Target r (1.0)')
        axes[1, 0].legend()
        axes[1, 1].plot(initial_radii_to_test.cpu().numpy(), final_potential_energies, 'o-', color='tab:purple'); axes[1, 1].set_title('Final Potential Energy')
        axes[1, 1].axhline(y=eval_env.pe_target, color='black', linestyle='--', label=f'Target PE ({eval_env.pe_target:.2f})')
        axes[1, 1].legend()
        axes[1, 2].axis('off')  # Empty subplot for better layout

        axes[0, 2].set_ylim(0, 1)
        axes[1, 0].set_ylim(0, 5) # Adjust ylim for radius

        for ax in axes.flatten()[:5]: ax.set_xlabel('Initial Radius'); ax.grid(True, alpha=0.5)
        plt.savefig(filename)
        plt.close(fig)
        print(f"✅ Cross-radii evaluation plot saved to {filename}")

    def plot_action_heatmap(self, policy_net, obs_normalizer, filename, device, 
                           x_range=(-3, 3), y_range=(-3, 3), resolution=50, 
                           fixed_vx=0.0, fixed_vy=1.0):
        """
        Creates a heatmap showing the mean action output by the model across the x,y space
        with fixed velocity components. This helps visualize the policy's behavior patterns.
        """
        print(f"🔥 Generating action heatmap across x,y space: {filename}...")
        
        x_vals = torch.linspace(x_range[0], x_range[1], resolution, device=device)
        y_vals = torch.linspace(y_range[0], y_range[1], resolution, device=device)
        X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
        
        x_flat, y_flat = X.flatten(), Y.flatten()
        num_positions = len(x_flat)
        
        temp_env = OrbitalEnvironment(num_envs=1, sim_device=device)
        actions = torch.zeros(num_positions, device=device)
        
        batch_size = 1000
        for i in range(0, num_positions, batch_size):
            end_idx = min(i + batch_size, num_positions)
            batch_size_actual = end_idx - i
            
            batch_obs = torch.zeros(batch_size_actual, 10, device=device)
            
            for j, pos_idx in enumerate(range(i, end_idx)):
                x, y = x_flat[pos_idx], y_flat[pos_idx]
                
                temp_env.x[0], temp_env.y[0] = x, y
                temp_env.vx[0], temp_env.vy[0] = fixed_vx, fixed_vy
                
                temp_env.integral_pe_error[0] = 0.0
                temp_env.integral_ecc_error[0] = 0.0
                temp_env.previous_pe_error[0] = 0.0
                
                r, vr, _, apo, ecc, pe = temp_env._get_raw_state()
                temp_env.prev_r[0] = r[0]
                temp_env.prev_v_radial[0] = vr[0]
                temp_env.prev_eccentricity[0] = ecc[0]
                
                obs = temp_env._get_observation()
                batch_obs[j] = obs[0]
            
            with torch.no_grad():
                norm_obs = obs_normalizer(batch_obs, update=False)
                batch_actions = policy_net(norm_obs).mean.squeeze()
                actions[i:end_idx] = batch_actions
        
        action_grid = actions.reshape(resolution, resolution).cpu().numpy()
        X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.contourf(X_np, Y_np, action_grid, levels=50, cmap='RdBu_r', extend='both')
        cbar = plt.colorbar(im, ax=ax); cbar.set_label('Action Magnitude', rotation=270, labelpad=15)
        ax.scatter(0, 0, color='yellow', s=200, label='Central Body', zorder=5, edgecolors='black')
        angles = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(angles), np.sin(angles), 'g--', label='Target Orbit (r=1.0)', zorder=3, linewidth=2)
        
        vel_mag = np.sqrt(fixed_vx**2 + fixed_vy**2)
        if vel_mag > 0:
            for x, y in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
                    ax.arrow(x, y, fixed_vx/vel_mag*0.2, fixed_vy/vel_mag*0.2, 
                            head_width=0.05, head_length=0.1, fc='black', ec='black', zorder=4)
        
        ax.set_xlabel('X Position'); ax.set_ylabel('Y Position')
        ax.set_title(f'Policy Action Heatmap\nFixed Velocity: vx={fixed_vx:.2f}, vy={fixed_vy:.2f}')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend()
        vel_text = f'Fixed Velocity: ({fixed_vx:.2f}, {fixed_vy:.2f})'
        ax.text(0.02, 0.98, vel_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Action heatmap saved to {filename}")