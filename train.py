import os
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from model import create_model

# SaveBest is a custom callback that saves the best model during training based on average episode reward.
class SaveBest(BaseCallback):
    def __init__(self, reward_threshold, save_path, verbose=0):
        """
        Args:
            reward_threshold: Threshold for stopping training when the average reward is achieved (float).
            save_path: Path to save the best model and training data (str).
            verbose: Verbosity level (int).
        
        Returns:
            None. Initializes callback variables.
        """
        super(SaveBest, self).__init__(verbose)
        self.reward_threshold = reward_threshold
        self.save_path = save_path
        self.all_episodes_data = []
        self.best_mean_reward = -float('inf')

    def _on_step(self) -> bool:
        """
        Collects the episode data and checks if the current model's reward exceeds the threshold.
        Returns: Boolean to indicate whether to continue training.
        """
        done = self.locals['dones'][0]
        if done:
            episode_data = self.training_env.get_attr("episode_data")[0] # Collect episode data from env
            
            if len(episode_data) > 0:  # Ensure episode_data nonempty
                self.all_episodes_data.append(episode_data)  # Store episode data
                mean_reward = sum([step[4] for step in episode_data]) / len(episode_data)
                
                # Save best model if current mean reward is better
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(os.path.join(self.save_path, "best_model"))

                # Stop training if mean reward exceeds threshold
                if mean_reward >= self.reward_threshold:
                    return False
        
        return True

    def save_episode_data(self):
        """
        Saves the collected episode data after training.
        Returns: None. Saves data to a file.
        """
        with open(os.path.join(self.save_path, 'all_episodes_data.txt'), 'w') as f:
            for episode in self.all_episodes_data:
                f.write(f"{episode}\n")


"""
    Train the model in "env" environment for X number of timesteps with Y reward threshold to stop training
    Save into particular directory
"""
def train_model(env, save_dir, total_timesteps=10_000, reward_threshold=100):
    """
    Args:
        env: Training environment (gym.Env).
        save_dir: Directory to save the model and training data (str).
        total_timesteps: Number of timesteps to train (int).
        reward_threshold: Reward value to stop training early (float).

    Returns:
        Tuple of (trained model, model save path).
    """
    # Create a unique folder for this specific model using timestamp
    model_name = f"model_{datetime.datetime.now().strftime('%S-%M-%H_%d-%m-%Y')}"
    model_save_path = os.path.join(save_dir, model_name)
    os.makedirs(model_save_path, exist_ok=True)

    callback = SaveBest(reward_threshold=reward_threshold, save_path=model_save_path)
    model = create_model(env)
    
    # Train the model and save weights
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(os.path.join(model_save_path, "ppo_orbital_model"))

    # Save training episode data
    callback.save_episode_data()

    print(f"Training completed. Model and data saved in {model_save_path}")
    return model, model_save_path
