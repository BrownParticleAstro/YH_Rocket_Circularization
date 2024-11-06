import os
import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from model import create_model

# SaveBest is a custom callback that saves the best model during training based on average episode reward.
class SaveBest(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(SaveBest, self).__init__(verbose)
        self.save_path = os.path.join(save_path, 'training')  # Create a "training" folder
        os.makedirs(self.save_path, exist_ok=True)  # Ensure the folder exists
        self.episode_data = []
        self.best_mean_reward = -float('inf')
        self.episode_num = 0  # Track episode number for saving
        self.episode_step = 0  # Track number of steps in the current episode

    def _on_step(self) -> bool:
        """
        This method is called at each step in the environment.
        """
        # Collect state and action data at each step
        info = self.locals["infos"][0]  # Assuming single environment
        state = info["state"]  # Assuming 'state' contains (x, y, vx, vy)
        action = self.locals["actions"]  # Actions taken at this step
        reward = self.locals["rewards"][0]  # Capture the reward at this step

        # Ensure the action is stored as 1D by squeezing out any extra dimensions
        action = np.squeeze(action)

        # Append relevant data with the current step in the episode
        self.episode_data.append((*state, self.episode_step, action, reward))

        # Increment the step counter for the current episode
        self.episode_step += 1

        # Check if the episode has finished
        done = self.locals["dones"][0]
        if done:
            # Save the current episode data
            self.save_episode_data(self.episode_data, self.episode_num)
            self.episode_num += 1
            self.episode_data = []  # Reset episode data for the next episode
            self.episode_step = 0  # Reset step counter for the next episode

        return True

    def save_episode_data(self, episode_data, episode_num):
        """
        Saves the current episode data to a file.
        """
        # Save the current episode data to a .npz file, ensuring each array is properly dimensioned
        np.savez(os.path.join(self.save_path, f'episode_{episode_num}.npz'), 
                 x=np.array([step[0] for step in episode_data]), 
                 y=np.array([step[1] for step in episode_data]),
                 vx=np.array([step[2] for step in episode_data]),
                 vy=np.array([step[3] for step in episode_data]),
                 episode_step=np.array([step[4] for step in episode_data]),  # Save the step within the episode
                 action=np.array([step[5] for step in episode_data]),  # Ensure actions are saved without extra dimensions
                 reward=np.array([step[6] for step in episode_data]))  # Save rewards

"""
    Train the model in "env" environment for X number of timesteps with Y reward threshold to stop training
    Save into particular directory
"""
def train_model(env, save_dir, total_timesteps=10_000):
    """
    Args:
        env: Training environment (gym.Env).
        save_dir: Directory to save the model and training data (str).
        total_timesteps: Number of timesteps to train (int).

    Returns:
        Tuple of (trained model, model save path).
    """
    # Create a unique folder for this specific model using timestamp
    model_name = f"model_{datetime.datetime.now().strftime('%H-%M-%S-_%d-%m-%Y')}"
    model_save_path = os.path.join(save_dir, model_name)
    os.makedirs(model_save_path, exist_ok=True)

    callback = SaveBest(save_path=model_save_path)
    model = create_model(env)
    
    # Train the model and save weights
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(os.path.join(model_save_path, "ppo_orbital_model"))

    print(f"Training completed. Model and data saved in {model_save_path}")
    return model, model_save_path
