# main.py

import os
import torch

from environment import OrbitalEnvironment
from train import train_model
from test import test_model
from render import Renderer
from model import PolicyNetwork

# ===================================================================
# Main Execution Block
# ===================================================================
if __name__ == '__main__':
    """
    This script serves as the main entry point for the project.
    It orchestrates the following sequence:
    1. Training: Trains the PPO agent using the `train_model` function.
    2. Evaluation: Loads the final trained model.
    3. Testing: Runs a standard test episode and saves its trajectory data.
    4. Rendering: Generates a suite of visualizations to analyze the agent's performance.
    """

    # --- 1. Train the model ---
    # The `save_dir` is where all models and their corresponding logs/plots will be stored.
    save_dir = './models'
    # The number of updates determines how long the agent trains.
    # A quick run might use 200, while a full training run might use 1000+.
    # Set to 1 for a quick test of the pipeline.
    # scp -r ~/Desktop/CS_Classwork/UTRA/new Gaitskell/Grav-Nav-RL ccv-vscode-node:~/scratch
    _, model_save_path, obs_normalizer = train_model(save_dir, total_updates=1000) 
    
    print("\n" + "="*50)
    print("      TRAINING COMPLETE - STARTING EVALUATION")
    print("="*50 + "\n")

    # --- 2. Setup for Testing and Rendering ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # A prototype environment instance is needed by the renderer to get parameters like max_steps.
    env_proto = OrbitalEnvironment(num_envs=1, sim_device=device)
    
    # Load the final trained model weights into a new network instance.
    final_model_path = os.path.join(model_save_path, "policy_final.pt")
    state_dim, action_dim = 10, 1
    eval_policy_net = PolicyNetwork(state_dim, action_dim).to(device)
    eval_policy_net.load_state_dict(torch.load(final_model_path))
    # Set the network to evaluation mode (this disables things like dropout if it were used).
    eval_policy_net.eval()

    # --- 3. Run a Standard Test Episode ---
    # This saves a single trajectory .npz file, which could be used for other types of analysis.
    test_env = OrbitalEnvironment(num_envs=1, sim_device=device, max_steps=1000)
    test_data_dir = os.path.join(model_save_path, "testing_data")
    test_model(test_env, eval_policy_net, obs_normalizer, test_data_dir, episode_num=1)
    
    # --- 4. Generate All Final Visualizations ---
    # Create a renderer instance pointing to the specific model's output directory.
    renderer = Renderer(model_save_path=model_save_path)

    print("\n--- Generating final visualizations ---")
    
    # Generate the single-episode evaluation plot (Radius & Action vs. Time)
    eval_plot_filename = os.path.join(model_save_path, "final_evaluation_plot.png")
    eval_3_8_plot_filename = os.path.join(model_save_path, "final_3_8_evaluation_plot.png")
    renderer.evaluate_and_plot_policy(eval_policy_net, obs_normalizer, env_proto, eval_plot_filename, device)
    renderer.evaluate_and_plot_policy(eval_policy_net, obs_normalizer, env_proto, eval_3_8_plot_filename, device, init_r=3.8)

    # Generate the plot showing performance across a range of starting conditions
    eval_radii_plot_filename = os.path.join(model_save_path, "final_eval_across_radii.png")
    renderer.evaluate_across_initial_radii(eval_policy_net, obs_normalizer, filename=eval_radii_plot_filename, device=device)

    # Generate action heatmap across x,y space with fixed velocity
    heatmap_filename = os.path.join(model_save_path, "action_heatmap.png")
    renderer.plot_action_heatmap(eval_policy_net, obs_normalizer, heatmap_filename, device=device, 
                                x_range=(-3, 3), y_range=(-3, 3), resolution=50, 
                                fixed_vx=0.0, fixed_vy=1.0)

    # Generate the final GIF animation showing multiple trajectories
    gif_filename = os.path.join(model_save_path, "final_orbit_animation.gif")
    renderer.render_episode_to_gif(eval_policy_net, obs_normalizer, env_proto, gif_filename, device=device)

    print(f"\n✅ All evaluation and rendering finished. Results are in: {model_save_path}")