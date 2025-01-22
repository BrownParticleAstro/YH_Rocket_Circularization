from stable_baselines3 import PPO
from PPO import PPO_v1
import numpy as np
import os

""" Runs a test episode of the trained model on the environment and saves the results. """
def test_model(env, model_path, model_save_dir, episode_num):
    """
    Args:
        env: Environment to test on (gym.Env).
        model_path: Path to the trained PPO model (str).
        model_save_path: Directory to save the test episode data (str).
        episode_num: Number to identify the episode (int).

    Returns:
        None. Saves the test episode data to a file.
    """
    # Ensure the directory for saving testing data exists
    test_data_dir = os.path.join(model_save_dir, "testing")
    os.makedirs(test_data_dir, exist_ok=True)

    model = PPO.load(model_path)
    print("model loaded")
    obs = env.reset()
    done = False
    episode_data = []  # To store the test episode data
    timestep = 0  # Initialize a manual timestep tracker

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        x, y, vx, vy = env.state
        r_err_norm = info['r_err_norm']
        d_r_err_norm = info['d_r_err_norm']
        int_r_err_norm = info['int_r_err_norm']

        # Append data with new quantities
        episode_data.append([
            x, y, vx, vy, timestep, action, reward,
            r_err_norm, d_r_err_norm, int_r_err_norm
        ])
        timestep += 1

    # Save the episode data to the test data directory
    np.savez(os.path.join(test_data_dir, f'episode_{episode_num}.npz'),
             x=np.array([step[0] for step in episode_data]),
             y=np.array([step[1] for step in episode_data]),
             vx=np.array([step[2] for step in episode_data]),
             vy=np.array([step[3] for step in episode_data]),
             episode_step=np.array([step[4] for step in episode_data]),
             action=np.array([step[5] for step in episode_data]),
             reward=np.array([step[6] for step in episode_data]),
             r_err_norm=np.array([step[7] for step in episode_data]),
             d_r_err_norm=np.array([step[8] for step in episode_data]),
             int_r_err_norm=np.array([step[9] for step in episode_data]))

    print(f"Test episode {episode_num} completed and saved in {test_data_dir}")

def test_PPO(env, model_path, model_save_path, total_num_ep):
    test_data_dir = os.path.join(model_save_path, "testing")
    os.makedirs(test_data_dir, exist_ok=True)

    ########## HYPER PARAMETERS ##############################
    state_dim = 9
    action_dim = 1 
    lr_actor = 2e-4
    lr_critic = 5e-3 
    gamma = 0.99
    K_epochs = 80
    eps_clip = 0.2
    action_std_init = 0.6 

    ep_num = 0 
    episode_data = []
    ppo_agent = PPO_v1(state_dim, action_dim,lr_actor, lr_critic,gamma,K_epochs,eps_clip, action_std_init )
    ppo_agent.load(model_path)
    print(f'the PPO has been loaded from{model_path}')

    for ep in range(1,total_num_ep +1) : 
        ep_reward = 0 
        state = env.reset()
        done = False
        time_step = 0 
        while not done :
            action = ppo_agent.select_action(state)
            obs, reward, done, info = env.step(action)
            x, y, vx, vy = env.state
            r_err_norm = info['r_err_norm']
            d_r_err_norm = info['d_r_err_norm']
            int_r_err_norm = info['int_r_err_norm']

            episode_data.append([
            x, y, vx, vy, time_step, action, reward,
            r_err_norm, d_r_err_norm, int_r_err_norm, ep_num])
            time_step += 1

        np.savez(os.path.join(test_data_dir, f'episode_{ep_num}.npz'),
            x=np.array([step[0] for step in episode_data]),
            y=np.array([step[1] for step in episode_data]),
            vx=np.array([step[2] for step in episode_data]),
            vy=np.array([step[3] for step in episode_data]),
            episode_step=np.array([step[4] for step in episode_data]),
            action=np.array([step[5] for step in episode_data]),
            reward=np.array([step[6] for step in episode_data]),
            r_err_norm=np.array([step[7] for step in episode_data]),
            d_r_err_norm=np.array([step[8] for step in episode_data]),
            int_r_err_norm=np.array([step[9] for step in episode_data]))
        
        ep_num += 1 
        print(f"Test episode {ep_num} completed and saved in {test_data_dir}")
