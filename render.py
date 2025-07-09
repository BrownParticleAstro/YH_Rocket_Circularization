import os
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from environment import OrbitalEnvironment

plt.style.use('./rose-pine-dawn.mplstyle')

class Renderer:
    def __init__(self, model_save_path):
        self.model_save_path = model_save_path
        os.makedirs(self.model_save_path, exist_ok=True)

    def render_episode_to_gif(self, policy_net, obs_normalizer, env_prototype, filename, device='cpu', fps=30, dt=0.01, step_scale_factor=1.0):
        print(f"🎬 Generating GIF from diverse initial radii: {filename}...")
        num_render_envs = 10
        colors = plt.cm.Dark2(np.linspace(0, 1, num_render_envs))
        
        base_max_steps = 6_000
        max_steps = int(base_max_steps * step_scale_factor) # Scale max_steps
        gif_env = OrbitalEnvironment(num_envs=num_render_envs, max_steps=max_steps, sim_device=device, dt=dt)
        
        # FIX: Advance curriculum state if angle curriculum is enabled
        if env_prototype.angle_curriculum_rate is not None:
            gif_env.angle_curriculum_rate = env_prototype.angle_curriculum_rate
            gif_env.set_curriculum_episode(env_prototype.current_episode_in_loop)
            print(f"   Using angle curriculum: ±{gif_env.max_angle_range:.3f} rad (±{gif_env.max_angle_range * 180 / np.pi:.1f}°)")

        gif_env.reset()
        initial_radii = torch.linspace(0.2, 4.0, num_render_envs, device=device)
        gif_env.x, gif_env.y, gif_env.vx = initial_radii, torch.zeros_like(initial_radii), torch.zeros_like(initial_radii)
        gif_env.vy = torch.sqrt(gif_env.GM / torch.clamp(initial_radii, min=1e-6))
        r, vr, _, _, ecc, _ = gif_env._get_raw_state()
        gif_env.prev_r, gif_env.prev_v_radial, gif_env.prev_eccentricity = r.clone(), vr.clone(), ecc.clone()
        gif_env.previous_r_error = torch.abs(r - gif_env.r_target)
        obs = gif_env._get_observation()
        trajectories = [[(gif_env.x[i].item(), gif_env.y[i].item())] for i in range(num_render_envs)]
        completed_episodes = torch.zeros(num_render_envs, dtype=torch.bool, device=device)

        with imageio.get_writer(filename, mode='I', fps=fps) as writer:
            last_image = None
            for step in range(gif_env.max_steps):
                # -------------------------------------------------
                # 1) Query policy for the next action (thrust & angle)
                # -------------------------------------------------
                with torch.no_grad():
                    norm_obs = obs_normalizer(obs, update=False)
                    action = policy_net(norm_obs).mean
                # Clamp thrust and angle to the same bounds used during training
                clamped_action = torch.stack([
                    torch.clamp(action[:, 0], -0.1, 0.1),
                    torch.clamp(action[:, 1], -1.0, 1.0)
                ], dim=-1)

                # -------------------------------------------------
                # 2) Render the current state *before* applying action
                # -------------------------------------------------
                plt.figure(figsize=(8, 8))
                plt.scatter(0, 0, color='yellow', s=1000, label='Central Body', zorder=5)
                angles = np.linspace(0, 2 * np.pi, 200)
                plt.plot(np.cos(angles), np.sin(angles), 'g--', label='Target Orbit (r=1.0)', zorder=1)
                for i, traj in enumerate(trajectories):
                    if not traj: continue
                    traj_np = np.array(traj)
                    plt.plot(traj_np[:, 0], traj_np[:, 1], linestyle=':', color=colors[i], zorder=2)
                    # Plot satellite position
                    sat_x, sat_y = traj_np[-1, 0], traj_np[-1, 1]
                    plt.plot(sat_x, sat_y, 'o', color=colors[i], markersize=8, markeredgecolor='black', zorder=3)

                    # ----------------------------
                    # Draw thrust direction arrow
                    # ----------------------------
                    thrust_mag   = clamped_action[i, 0].item()
                    angle_ctrl   = clamped_action[i, 1].item()
                    # Skip arrow if thrust magnitude is (almost) zero to avoid clutter
                    if abs(thrust_mag) > 1e-4:
                        # Compute radial & tangential unit vectors
                        r_vec = np.array([sat_x, sat_y])
                        r_norm = np.linalg.norm(r_vec) + 1e-9
                        radial_unit = r_vec / r_norm
                        tang_unit   = np.array([-radial_unit[1], radial_unit[0]])
                        alpha = angle_ctrl * (np.pi / 2.0)  # same formula as env
                        thrust_dir = np.cos(alpha) * tang_unit + np.sin(alpha) * radial_unit
                        # Scale arrow length for visibility – empirical factor
                        arrow_scale = 3.0  # tweak if needed
                        dx, dy = thrust_dir * thrust_mag * arrow_scale
                        plt.arrow(sat_x, sat_y, dx, dy,
                                  width=0.01, head_width=0.08, head_length=0.12,
                                  color=colors[i], length_includes_head=True, zorder=4, alpha=0.8)
                plt.title("Multi-Episode Trajectory Visualization (Diverse Initial Radii)")
                plt.xlim([-4.5, 4.5]); plt.ylim([-4.5, 4.5]); plt.gca().set_aspect('equal', adjustable='box'); plt.grid(True, alpha=0.3)
                plt.text(-4.5, 5.0, f"Step: {step+1}", fontsize=14, ha='left', va='bottom', fontweight='bold')
                fig = plt.gcf(); fig.canvas.draw()
                image = np.array(fig.canvas.renderer.buffer_rgba()); plt.close(fig)
                writer.append_data(image)
                last_image = image

                # -------------------------------------------------
                # 3) Apply the action to advance the simulation
                # -------------------------------------------------
                next_obs, _, done_tensor, _, _ = gif_env.step(clamped_action)
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

    def evaluate_and_plot_policy(self, policy_net, obs_normalizer, env_prototype, filename, device, init_r=None, dt=0.01, step_scale_factor=1.0):
        print(f"📈 Evaluating policy and plotting results to {filename}...")
        base_max_steps = 5_000
        scaled_max_steps = int(base_max_steps * step_scale_factor)
        eval_env = OrbitalEnvironment(num_envs=1, max_steps=scaled_max_steps, sim_device=device, 
                                      angle_curriculum_rate=env_prototype.angle_curriculum_rate, dt=dt)
        
        # FIX: Advance curriculum state if angle curriculum is enabled
        if env_prototype.angle_curriculum_rate is not None:
            eval_env.set_curriculum_episode(env_prototype.current_episode_in_loop)
            print(f"   Using angle curriculum: ±{eval_env.max_angle_range:.3f} rad (±{eval_env.max_angle_range * 180 / np.pi:.1f}°)")
        
        obs = eval_env.reset(env_indices=[0])
        if init_r is not None:
            # Override logic...
            init_r_tensor = torch.tensor([init_r], device=device, dtype=torch.float32)
            eval_env.x[0], eval_env.y[0], eval_env.vx[0] = init_r_tensor, 0.0, 0.0
            eval_env.vy[0] = torch.sqrt(eval_env.GM / torch.clamp(init_r_tensor, min=1e-6))
            r, vr, _, _, ecc, _ = eval_env._get_raw_state()
            eval_env.prev_r[0], eval_env.prev_v_radial[0], eval_env.prev_eccentricity[0] = r[0], vr[0], ecc[0]
            eval_env.previous_r_error[0] = torch.abs(r[0] - eval_env.r_target)
            obs = eval_env._get_observation()
        
        radius_history, actions_history, eccentricity_history, done = [], [], [], False
        r, _, _, _, ecc, _ = eval_env._get_raw_state()
        radius_history.append(r.item()); eccentricity_history.append(ecc.item())
        
        for _ in range(eval_env.max_steps):
            if done: break
            with torch.no_grad():
                norm_obs = obs_normalizer(obs, update=False)
                action = policy_net(norm_obs).mean
            actions_history.append(action.squeeze().cpu().numpy())
            clamped_action = torch.stack([torch.clamp(action[:, 0], -0.1, 0.1), torch.clamp(action[:, 1], -1.0, 1.0)], dim=-1)
            obs, _, done_tensor, _, _ = eval_env.step(clamped_action)
            done = done_tensor.item()
            r, _, _, _, ecc, _ = eval_env._get_raw_state()
            radius_history.append(r.item()); eccentricity_history.append(ecc.item())
        
        # Plotting logic... (remains the same)
        actions_history_np = np.array(actions_history)
        # Clamp angle-control values to the prototype environment's allowable range for cleaner plots
        angle_limit = getattr(env_prototype, 'max_angle_range', None)
        if angle_limit is not None and angle_limit > 0 and actions_history_np.shape[1] > 1:
            actions_history_np[:, 1] = np.clip(actions_history_np[:, 1], -angle_limit, angle_limit)
        is_angle_enabled = env_prototype.max_angle_range > 0
        num_plots = 4 if is_angle_enabled else 3
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=True) 
        fig.suptitle('Policy Evaluation: State & Action vs. Time', fontsize=16)
        print(f"Length of actions_history_np: {actions_history_np.shape[0]}, min thrust: {np.min(actions_history_np[:,0])}, max thrust: {np.max(actions_history_np[:,0])}")
        axes = np.array(axes).flatten()
        axes[0].plot(radius_history[:-1], label='Radius (r)', color='dodgerblue'); axes[0].axhline(y=1.0, color='r', linestyle='--', label='Target Radius (r=1.0)'); axes[0].set_title('Satellite Radius vs. Time'); axes[0].set_ylabel('Radius'); axes[0].legend(); axes[0].grid(True, alpha=0.5)
        axes[1].plot(eccentricity_history[:-1], label='Eccentricity (e)', color='purple'); axes[1].axhline(y=0.0, color='r', linestyle='--', label='Target Eccentricity (e=0.0)'); axes[1].set_title('Orbital Eccentricity vs. Time'); axes[1].set_ylabel('Eccentricity'); axes[1].legend(); axes[1].grid(True, alpha=0.5)
        axes[2].plot(actions_history_np[:, 0], label='Thrust Action', color='seagreen', marker='.', linestyle='none'); axes[2].set_title('Action (Thrust Magnitude) vs. Time'); axes[2].set_ylabel('Thrust Magnitude'); axes[2].legend(); axes[2].grid(True, alpha=0.5)
        if is_angle_enabled:
            print(f"Length of actions_history_np: {actions_history_np.shape[0]}, min angle: {np.min(actions_history_np[:,1])}, max angle: {np.max(actions_history_np[:,1])}")
            axes[3].plot(actions_history_np[:, 1], label='Angle Action', color='darkorange', marker='.', linestyle='none'); axes[3].set_title('Action (Angle Control) vs. Time'); axes[3].set_xlabel('Time Step'); axes[3].set_ylabel('Angle Control'); axes[3].set_ylim([-angle_limit, angle_limit]); axes[3].legend(); axes[3].grid(True, alpha=0.5)
        else:
            axes[2].set_xlabel('Time Step')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(filename); plt.close(fig)
        print(f"✅ Evaluation plot saved to {filename}")

    def evaluate_across_initial_radii(self, policy_net, obs_normalizer, env_prototype, filename, device, num_data_points=30, dt=0.01, step_scale_factor=1.0):
        print(f"📊 Evaluating policy across initial radii and plotting to {filename}...")
        base_max_steps = 5_000
        scaled_max_steps = int(base_max_steps * step_scale_factor)
        eval_env = OrbitalEnvironment(num_envs=1, max_steps=scaled_max_steps, sim_device=device,
                                      angle_curriculum_rate=env_prototype.angle_curriculum_rate, dt=dt)
        
        # FIX: Advance curriculum state if angle curriculum is enabled
        if env_prototype.angle_curriculum_rate is not None:
            eval_env.set_curriculum_episode(env_prototype.current_episode_in_loop)
            print(f"   Using angle curriculum: ±{eval_env.max_angle_range:.3f} rad (±{eval_env.max_angle_range * 180 / np.pi:.1f}°)")
        
        initial_radii_to_test = torch.linspace(0.2, 4.0, num_data_points, device=device)
        initial_actions, episode_lengths, final_eccentricities, final_radii = [], [], [], []

        for init_r_val in initial_radii_to_test:
            # Stepping and data collection logic... (remains the same)
            eval_env.x[0], eval_env.y[0], eval_env.vx[0] = init_r_val, 0.0, 0.0
            eval_env.vy[0] = torch.sqrt(eval_env.GM / torch.clamp(init_r_val, min=1e-6))
            obs = eval_env.reset(env_indices=[0])
            with torch.no_grad():
                initial_actions.append(policy_net(obs_normalizer(obs, update=False)).mean[0, 0].item())
            done, final_r, final_ecc = False, None, None
            for step in range(eval_env.max_steps):
                if done: break
                with torch.no_grad():
                    action = policy_net(obs_normalizer(obs, update=False)).mean
                    clamped_action = torch.stack([torch.clamp(action[:, 0], -0.1, 0.1), torch.clamp(action[:, 1], -1.0, 1.0)], dim=-1)
                obs, _, done_tensor, info, _ = eval_env.step(clamped_action)
                done = done_tensor.item()
                if done and info and 'final_radius' in info:
                    final_r, final_ecc = info['final_radius'][0], info['final_eccentricity'][0]
            if final_r is None:
                r, _, _, _, ecc, _ = eval_env._get_raw_state()
                final_r, final_ecc = r.item(), ecc.item()
            episode_lengths.append(step + 1); final_eccentricities.append(final_ecc); final_radii.append(final_r)

        # Plotting logic... (remains the same)
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), tight_layout=True)
        fig.suptitle('Policy Performance Across Initial Radii', fontsize=16)
        axes[0, 0].plot(initial_radii_to_test.cpu().numpy(), initial_actions, 'o-'); axes[0, 0].set_title('Initial Thrust Action')
        axes[0, 1].plot(initial_radii_to_test.cpu().numpy(), episode_lengths, 'o-', color='tab:orange'); axes[0, 1].set_title('Episode Length')
        axes[1, 0].plot(initial_radii_to_test.cpu().numpy(), final_eccentricities, 'o-', color='tab:green'); axes[1, 0].set_title('Final Eccentricity')
        axes[1, 1].plot(initial_radii_to_test.cpu().numpy(), final_radii, 'o-', color='tab:red'); axes[1, 1].set_title('Final Radius')
        axes[1, 1].axhline(y=1.0, color='black', linestyle='--', label='Target r (1.0)'); axes[1, 1].legend()
        axes[1, 0].set_ylim(0, 1); axes[1, 1].set_ylim(0, 5)
        for ax in axes.flatten(): ax.set_xlabel('Initial Radius'); ax.grid(True, alpha=0.5)
        plt.savefig(filename); plt.close(fig)
        print(f"✅ Cross-radii evaluation plot saved to {filename}")

    def plot_action_heatmap(self, policy_net, obs_normalizer, env_prototype, filename, device, 
                           x_range=(-3, 3), y_range=(-3, 3), resolution=50, 
                           fixed_vx=0.0, fixed_vy=1.0, dt=0.01):
        print(f"🔥 Generating action heatmap across x,y space: {filename}...")
        x_vals, y_vals = torch.linspace(*x_range, resolution, device=device), torch.linspace(*y_range, resolution, device=device)
        X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
        x_flat, y_flat = X.flatten(), Y.flatten()
        
        temp_env = OrbitalEnvironment(num_envs=1, sim_device=device,
                                      angle_curriculum_rate=env_prototype.angle_curriculum_rate, dt=dt)
        
        # FIX: Advance curriculum state if angle curriculum is enabled
        if env_prototype.angle_curriculum_rate is not None:
            temp_env.set_curriculum_episode(env_prototype.current_episode_in_loop)
            print(f"   Using angle curriculum: ±{temp_env.max_angle_range:.3f} rad (±{temp_env.max_angle_range * 180 / np.pi:.1f}°)")
        
        actions = torch.zeros(len(x_flat), 2, device=device)
        
        # Batch processing logic... (remains the same)
        batch_size = 1000
        for i in range(0, len(x_flat), batch_size):
            end_idx = min(i + batch_size, len(x_flat))
            batch_obs_list = []
            for pos_idx in range(i, end_idx):
                temp_env.x[0], temp_env.y[0], temp_env.vx[0], temp_env.vy[0] = x_flat[pos_idx], y_flat[pos_idx], fixed_vx, fixed_vy
                temp_env.integral_r_err[0], temp_env.integral_ecc_error[0], temp_env.previous_r_error[0] = 0.0, 0.0, 0.0
                r = torch.sqrt(x_flat[pos_idx]**2 + y_flat[pos_idx]**2)
                vy_circ = torch.sqrt(temp_env.GM / torch.clamp(r, min=1e-6))
                temp_env.vx[0] = fixed_vx           # usually 0
                temp_env.vy[0] = vy_circ            # instead of hard-coded 1.0
                r, vr, _, _, ecc, _ = temp_env._get_raw_state()
                temp_env.prev_r[0], temp_env.prev_v_radial[0], temp_env.prev_eccentricity[0] = r[0], vr[0], ecc[0]
                batch_obs_list.append(temp_env._get_observation())
            with torch.no_grad():
                norm_obs = obs_normalizer(torch.cat(batch_obs_list, dim=0), update=False)
                raw_actions = policy_net(norm_obs).mean

                # 1.  Clamp exactly the same way the env does
                clamped = torch.stack(
                    [torch.clamp(raw_actions[:, 0], -0.1, 0.1),
                     torch.clamp(raw_actions[:, 1], -1.0, 1.0)], dim=-1)

                actions[i:end_idx] = clamped
        
        # Plotting logic... (remains the same)
        action_grid_thrust = actions[:, 0].reshape(resolution, resolution).cpu().numpy()
        X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
        is_angle_enabled = env_prototype.max_angle_range > 0
        if is_angle_enabled:
            fig, axes = plt.subplots(1, 2, figsize=(20, 9))
            fig.suptitle(f'Policy Action Heatmap\nFixed Velocity: vx={fixed_vx:.2f}, vy={fixed_vy:.2f}', fontsize=16)
            ax1, ax2 = axes
            im_thrust = ax1.contourf(X_np, Y_np, action_grid_thrust, levels=50, cmap='RdBu_r', vmin=-0.1, vmax=0.1, extend='both'); plt.colorbar(im_thrust, ax=ax1, orientation='vertical', shrink=0.8).set_label('Thrust Magnitude', rotation=270, labelpad=15)
            ax1.set_title('Action: Thrust Magnitude')
            action_grid_angle = actions[:, 1].reshape(resolution, resolution).cpu().numpy()
            im_angle = ax2.contourf(X_np, Y_np, action_grid_angle, levels=50, cmap='twilight_shifted', vmin=-1, vmax=1, extend='both'); plt.colorbar(im_angle, ax=ax2, orientation='vertical', shrink=0.8).set_label('Angle Control (-1: In, 1: Out)', rotation=270, labelpad=15)
            ax2.set_title('Action: Angle Control')
            for ax in axes:
                ax.scatter(0, 0, color='yellow', s=200, label='Central Body', zorder=5, edgecolors='black'); ax.plot(np.cos(np.linspace(0, 2*np.pi, 200)), np.sin(np.linspace(0, 2*np.pi, 200)), 'g--', label='Target Orbit', zorder=3); ax.set_xlabel('X Position'); ax.set_ylabel('Y Position'); ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend()
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.suptitle(f'Policy Action Heatmap\nFixed Velocity: vx={fixed_vx:.2f}, vy={fixed_vy:.2f}', fontsize=16)
            im_thrust = ax.contourf(X_np, Y_np, action_grid_thrust, levels=50, cmap='RdBu_r', vmin=-0.1, vmax=0.1, extend='both'); plt.colorbar(im_thrust, ax=ax).set_label('Thrust Magnitude', rotation=270, labelpad=15)
            ax.set_title('Action: Thrust Magnitude'); ax.scatter(0, 0, color='yellow', s=200, label='Central Body', zorder=5, edgecolors='black'); ax.plot(np.cos(np.linspace(0, 2*np.pi, 200)), np.sin(np.linspace(0, 2*np.pi, 200)), 'g--', label='Target Orbit (r=1.0)', zorder=3); ax.set_xlabel('X Position'); ax.set_ylabel('Y Position'); ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(filename, dpi=300, bbox_inches='tight'); plt.close(fig)
        print(f"✅ Action heatmap(s) saved to {filename}")