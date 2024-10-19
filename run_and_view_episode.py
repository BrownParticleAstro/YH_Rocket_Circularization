import os
from environment import OrbitalEnvWrapper
from stable_baselines3 import PPO
from render import Renderer
from model import create_model
from train import train_model
from test import test_model

# save_dir = './models'

# # Create the training environment
# env_train = OrbitalEnvWrapper(r0=1.0)

# # Train the model
# model, model_save_path = train_model(env_train, save_dir, total_timesteps=1_000_000, reward_threshold=4_000)

# # Load the trained model for inference and testing
# env_test = OrbitalEnvWrapper(r0=1.0)
# test_model(env_test, os.path.join(model_save_path, "ppo_orbital_model"), model_save_path, episode_num=1)

model_save_path = "./models/model_10-08-21_18-10-2024"
# Create a renderer instance using the dynamic model_save_path
renderer = Renderer(model_save_path=model_save_path)
renderer.render(episode_num=1, data_type='testing')
