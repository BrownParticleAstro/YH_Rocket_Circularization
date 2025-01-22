from environment import OrbitalEnvWrapper
from train import train_PPO_model
from test import test_PPO
import os
from render import Renderer

env_train = OrbitalEnvWrapper()
save_dir = './models'
model_save_dir, model_file_path = train_PPO_model(env_train,save_dir)
print(model_save_dir,model_file_path)

env_test = OrbitalEnvWrapper()
env_test.reset()

test_PPO(env_test,model_file_path,model_save_dir, total_num_ep = 3)
#renderer = Renderer(model_save_path)
#renderer.render(episode_num=1,interval=50,data_type="testing")
