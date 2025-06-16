import torch
import numpy as np
import os

from environment import OrbitalEnvironment
from model import PolicyNetwork

def test_model(env, policy_net, obs_normalizer, save_path, episode_num):
    """
    Runs a single test episode of the trained model on the environment and saves the trajectory data.
    This is useful for creating specific, repeatable visualizations.

    Args:
        env (OrbitalEnvironment): A single instance of the environment, already initialized.
        policy_net (PolicyNetwork): The trained PolicyNetwork model.
        obs_normalizer (ObservationNormalizer): The trained ObservationNormalizer.
        save_path (str): Directory to save the test episode data as an .npz file.
        episode_num (int): A number to identify the saved episode file.
    """
    print(f"--- Running test episode {episode_num} ---")
    os.makedirs(save_path, exist_ok=True)
    
    # Reset the single environment instance for the test run
    obs = env.reset(env_indices=[0])
    done = False
    episode_data = []
    timestep = 0

    while not done:
        # We don't need to track gradients during testing
        with torch.no_grad():
            # Normalize observation without updating the running stats
            norm_obs = obs_normalizer(obs, update=False)
            # Use the mean of the policy distribution for deterministic action
            action = policy_net(norm_obs).mean
        
        # Step the environment with the deterministic action
        obs, reward, done_tensor, info, _ = env.step(torch.clamp(action, -0.1, 0.1))
        done = done_tensor.any().item()

        # Extract state from the single environment instance for logging
        x, y, vx, vy = env.x[0].item(), env.y[0].item(), env.vx[0].item(), env.vy[0].item()
        
        # Store the state, action, and reward for this timestep
        episode_data.append([x, y, vx, vy, timestep, action[0].item(), reward[0].item()])
        timestep += 1
        if timestep >= env.max_steps:
            done = True

    # Save the collected episode data to a compressed .npz file
    np.savez(os.path.join(save_path, f'episode_{episode_num}.npz'),
             x=np.array([step[0] for step in episode_data]),
             y=np.array([step[1] for step in episode_data]),
             vx=np.array([step[2] for step in episode_data]),
             vy=np.array([step[3] for step in episode_data]),
             episode_step=np.array([step[4] for step in episode_data]),
             action=np.array([step[5] for step in episode_data]),
             reward=np.array([step[6] for step in episode_data]))

    print(f"Test episode {episode_num} completed and data saved in {save_path}")