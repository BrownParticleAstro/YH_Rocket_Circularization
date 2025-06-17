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
        gif_env = OrbitalEnvironment(num_envs=num_render_envs, max_steps=env_prototype.max_steps, sim_device=device)
        gif_env.max_steps = 2_000

        # --- FIX START: Manually set initial states for diverse starting radii ---

        # 1. Reset the environment to initialize all internal states (like episode counters, integrals, etc.)
        gif_env.reset()

        # 2. Create a tensor of uniformly spaced initial radii across the desired range.
        initial_radii = torch.linspace(0.2, 4.0, num_render_envs, device=device)

        # 3. Manually overwrite the position and velocity to create circular orbits at these new radii.
        gif_env.x = initial_radii
        gif_env.y.zero_() # Start on the x-axis
        gif_env.vx.zero_()
        # v = sqrt(GM/r) for a circular orbit
        gif_env.vy = torch.sqrt(gif_env.GM / torch.clamp(initial_radii, min=1e-6))

        # 4. Re-calculate the initial observation based on this new manually-set state.
        #    This is crucial because the policy network needs the correct starting observation.
        #    We also reset the PID-related 'previous' state trackers.
        r, vr, _, apo, ecc = gif_env._get_raw_state()
        gif_env.prev_r, gif_env.prev_v_radial, gif_env.prev_eccentricity = r.clone(), vr.clone(), ecc.clone()
        gif_env.previous_r_error = torch.abs(r - 1.0)
        obs = gif_env._get_observation()

        # --- FIX END ---

        frames = []
        # Initialize trajectory logging from the new starting positions
        trajectories = [[(gif_env.x[i].item(), gif_env.y[i].item())] for i in range(num_render_envs)]
        dones = torch.zeros(num_render_envs, dtype=torch.bool, device=device)

        for step in range(gif_env.max_steps):
            # --- Create a single frame for the GIF ---
            plt.figure(figsize=(8, 8))
            # Central body and target orbit
            plt.scatter(0, 0, color='yellow', s=1000, label='Central Body', zorder=5)
            angles = np.linspace(0, 2 * np.pi, 200)
            plt.plot(np.cos(angles), np.sin(angles), 'g--', label='Target Orbit (r=1.0)', zorder=1)

            # Plot each trajectory
            for i, traj in enumerate(trajectories):
                if not traj: continue
                traj_np = np.array(traj)
                plt.plot(traj_np[:, 0], traj_np[:, 1], linestyle=':', color=colors[i], zorder=2)
                plt.plot(traj_np[-1, 0], traj_np[-1, 1], 'o', color=colors[i], markersize=8, markeredgecolor='black', zorder=3)

            plt.title("Multi-Episode Trajectory Visualization (Diverse Initial Radii)")
            plt.xlim([-4.5, 4.5]); plt.ylim([-4.5, 4.5]) # Increased limits to see all trajectories
            plt.gca().set_aspect('equal', adjustable='box'); plt.grid(True, alpha=0.3)

            # Convert plot to an image array
            fig = plt.gcf()
            fig.canvas.draw()
            image = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            frames.append(image)

            # --- Step the environment ---
            with torch.no_grad():
                norm_obs = obs_normalizer(obs, update=False)
                action = policy_net(norm_obs).mean
                # Use a smaller, more stable action for evaluation rendering
                next_obs, _, done_tensor, _, _ = gif_env.step(torch.clamp(action, -0.1, 0.1))
                obs = next_obs

            for i in range(num_render_envs):
                if not dones[i]: # Only append to non-finished trajectories
                    trajectories[i].append((gif_env.x[i].item(), gif_env.y[i].item()))

            dones |= done_tensor
            if dones.all(): break # Stop if all episodes are done

        if frames:
            # Hold on the last frame for a moment
            for _ in range(fps): frames.append(frames[-1])
            imageio.mimsave(filename, frames, fps=fps)
        print(f"✅ GIF saved to {filename}")

    def evaluate_and_plot_policy(self, policy_net, obs_normalizer, env_prototype, filename, device):
        """
        Runs a single deterministic episode from a random start and plots key metrics
        (radius and action) over time. This helps analyze the policy's control strategy.

        Args:
            policy_net (PolicyNetwork): The trained policy network.
            obs_normalizer (ObservationNormalizer): The trained observation normalizer.
            env_prototype (OrbitalEnvironment): A sample environment to get parameters like max_steps.
            filename (str): The full path to save the output plot file.
            device (str): The device ('cpu' or 'cuda') to run the simulation on.
        """
        print(f"📈 Evaluating policy and plotting results to {filename}...")
        eval_env = OrbitalEnvironment(num_envs=1, max_steps=env_prototype.max_steps, sim_device=device)
        # Give a random start within the curriculum range for a robust test
        eval_env.x[0] = 0.2 + torch.rand(1, device=device) * 3.8
        obs = eval_env.reset(env_indices=[0])

        radii_history, actions_history = [], []
        done = False
        
        initial_r, _, _, _, _ = eval_env._get_raw_state()
        radii_history.append(initial_r.item())
        
        for _ in range(eval_env.max_steps):
            if done: break
            with torch.no_grad():
                norm_obs = obs_normalizer(obs, update=False)
                action = policy_net(norm_obs).mean
            
            actions_history.append(action.item())
            obs, _, done_tensor, _, _ = eval_env.step(action)
            done = done_tensor.item()
            
            r, _, _, _, _ = eval_env._get_raw_state()
            radii_history.append(r.item())
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle('Policy Evaluation: Radius & Action vs. Time', fontsize=16)
        
        axes[0].plot(radii_history[:-1], label='Radius (r)', color='dodgerblue')
        axes[0].axhline(y=1.0, color='r', linestyle='--', label='Target Radius (r=1.0)')
        axes[0].set_title('Satellite Radius vs. Time'); axes[0].set_ylabel('Radius')
        axes[0].legend(); axes[0].grid(True, alpha=0.5)
        
        axes[1].plot(actions_history, label='Action', color='seagreen', marker='.', linestyle='-')
        axes[1].set_title('Action vs. Time'); axes[1].set_xlabel('Time Step'); axes[1].set_ylabel('Action Magnitude')
        axes[1].legend(); axes[1].grid(True, alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename)
        plt.close(fig)
        print(f"✅ Evaluation plot saved to {filename}")

    def evaluate_across_initial_radii(self, policy_net, obs_normalizer, env_prototype, filename, device, num_data_points=30):
        """
        Evaluates the policy's performance across a range of initial radii to test its
        robustness and generalization. Plots final metrics against the starting radius.

        Args:
            policy_net (PolicyNetwork): The trained policy network.
            obs_normalizer (ObservationNormalizer): The trained observation normalizer.
            env_prototype (OrbitalEnvironment): A sample environment to get parameters like max_steps.
            filename (str): The full path to save the output plot file.
            device (str): The device ('cpu' or 'cuda') to run the simulation on.
            num_data_points (int): How many different initial radii to test.
        """
        print(f"📊 Evaluating policy across initial radii and plotting to {filename}...")
        eval_env = OrbitalEnvironment(num_envs=1, max_steps=env_prototype.max_steps, sim_device=device)
        
        initial_radii_to_test = torch.linspace(0.2, 4.0, num_data_points, device=device)
        initial_actions, episode_lengths, final_eccentricities, final_radii = [], [], [], []

        for init_r_val in initial_radii_to_test:
            # Manually set the environment state for this specific test case
            eval_env.x[0], eval_env.y[0], eval_env.vx[0] = init_r_val, 0.0, 0.0
            eval_env.vy[0] = torch.sqrt(eval_env.GM / torch.clamp(init_r_val, min=1e-6))
            obs = eval_env.reset(env_indices=[0])
            
            # Record the very first action the policy takes
            with torch.no_grad():
                norm_obs = obs_normalizer(obs, update=False)
                initial_actions.append(policy_net(norm_obs).mean.item())

            # Run the full episode
            done = False
            for step in range(eval_env.max_steps):
                if done: break
                with torch.no_grad():
                    norm_obs = obs_normalizer(obs, update=False)
                    action = policy_net(norm_obs).mean
                obs, _, done_tensor, _, _ = eval_env.step(torch.clamp(action, -0.1, 0.1))
                done = done_tensor.item()
            
            # Record the final state metrics
            r, _, _, _, ecc = eval_env._get_raw_state()
            episode_lengths.append(step + 1)
            final_eccentricities.append(ecc.item())
            final_radii.append(r.item())

        # Create a 2x2 grid of plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), tight_layout=True)
        fig.suptitle('Policy Performance Across Initial Radii', fontsize=16)
        
        axes[0, 0].plot(initial_radii_to_test.cpu().numpy(), initial_actions, 'o-'); axes[0, 0].set_title('Initial Action')
        axes[0, 1].plot(initial_radii_to_test.cpu().numpy(), episode_lengths, 'o-', color='tab:orange'); axes[0, 1].set_title('Episode Length')
        axes[1, 0].plot(initial_radii_to_test.cpu().numpy(), final_eccentricities, 'o-', color='tab:green'); axes[1, 0].set_title('Final Eccentricity')
        axes[1, 1].plot(initial_radii_to_test.cpu().numpy(), final_radii, 'o-', color='tab:red'); axes[1, 1].set_title('Final Radius')

        axes[1, 0].set_ylim(0, 1)
        axes[1, 1].set_ylim(0, 4)

        for ax in axes.flatten(): ax.set_xlabel('Initial Radius'); ax.grid(True, alpha=0.5)
        plt.savefig(filename)
        plt.close(fig)
        print(f"✅ Cross-radii evaluation plot saved to {filename}")