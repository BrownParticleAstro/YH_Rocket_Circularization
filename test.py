import torch
import numpy as np
import os

def test_model(env, policy_net, obs_normalizer, save_path, episode_num):
    """
    Runs a single test episode. No changes needed here as it uses the pre-configured
    environment instance passed to it, which will have the correct dt and max_steps.
    """
    print(f"--- Running test episode {episode_num} ---")
    os.makedirs(save_path, exist_ok=True)
    obs, done, episode_data, timestep = env.reset(env_indices=[0]), False, [], 0

    while not done:
        with torch.no_grad():
            norm_obs = obs_normalizer(obs, update=False)
            action = policy_net(norm_obs).mean
        clamped_action = torch.stack([torch.clamp(action[:, 0], -0.1, 0.1), torch.clamp(action[:, 1], -1.0, 1.0)], dim=-1)
        obs, reward, done_tensor, _, _ = env.step(clamped_action)
        done = done_tensor.any().item()
        x, y, vx, vy = env.x[0].item(), env.y[0].item(), env.vx[0].item(), env.vy[0].item()
        episode_data.append([x, y, vx, vy, timestep, action[0].cpu().numpy(), reward[0].item()])
        timestep += 1
        if timestep >= env.max_steps:
            done = True

    np.savez(os.path.join(save_path, f'episode_{episode_num}.npz'),
             x=np.array([s[0] for s in episode_data]), y=np.array([s[1] for s in episode_data]),
             vx=np.array([s[2] for s in episode_data]), vy=np.array([s[3] for s in episode_data]),
             episode_step=np.array([s[4] for s in episode_data]), action=np.array([s[5] for s in episode_data]),
             reward=np.array([s[6] for s in episode_data]))
    print(f"Test episode {episode_num} completed and data saved in {save_path}")