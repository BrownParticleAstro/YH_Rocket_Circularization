from stable_baselines3 import PPO
import numpy as np
import os

""" Runs a test episode of the trained model on the environment and saves the results. """
### WOULD BE IMPLEMENTED BY STUDENT
def test_model(env, model_path, model_save_path, episode_num):
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
    test_data_dir = os.path.join(model_save_path, "testing")
    os.makedirs(test_data_dir, exist_ok=True)

    model = PPO.load(model_path)
    obs = env.reset()
    done = False
    episode_data = []  # To store the test episode data
    timestep = 0  # Initialize a manual timestep tracker

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        
        # Collect the raw state variables from the environment
        x = env.state[0]
        y = env.state[1]
        vx = env.state[2]
        vy = env.state[3]
        
        # Append the state and action data for this timestep
        episode_data.append([x, y, vx, vy, timestep, action])
        
        timestep += 1  # Increment the manual timestep counter

    # Save the episode data to the test data directory
    np.savez(os.path.join(test_data_dir, f'episode_{episode_num}.npz'), 
             x=np.array([step[0] for step in episode_data]), 
             y=np.array([step[1] for step in episode_data]),
             vx=np.array([step[2] for step in episode_data]),
             vy=np.array([step[3] for step in episode_data]),
             timestep=np.array([step[4] for step in episode_data]),
             action=np.array([step[5] for step in episode_data]))

    print(f"Test episode {episode_num} completed and saved in {test_data_dir}")

