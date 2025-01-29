from environment import OrbitalEnvWrapper
from train import train_PPO_model
from test import test_PPO
import os
from render import Renderer

env_train = OrbitalEnvWrapper()
save_dir = './models'
model_save_dir, model_file_path = train_PPO_model(env_train,save_dir, max_training_timesteps = 10000)
print(model_save_dir,model_file_path)

env_test = OrbitalEnvWrapper()
env_test.reset()

test_PPO(env_test,model_file_path,model_save_dir, total_num_ep = 7)
renderer = Renderer(model_save_dir)
renderer.render(episode_num=0,interval=50,data_type="testing")

