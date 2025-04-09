from dqn_agent import DQAgent
import gym
from transform import Transforms
import numpy as np
from GoT import GoT
import tensorflow as tf

# Initializes an openai gym environment
def init_gym_env():

    env = GoT()
    state_space = env.reset()
    print(state_space)
    state_space = [[int(x) for x in i] for i in state_space]
    state_space = tf.Tensor(state_space)
    state_raw = np.zeros(state_space.shape, dtype=int)
    state_raw = tf.Tensor(state_raw)
    processed_state = Transforms.to_gray(state_raw)
    state_space = processed_state.shape
    action_space = env.action_space.n

    return env, state_space, action_space

# Initialize Gym Environment
env, state_space, action_space = init_gym_env()
    
# Create an agent
agent = DQAgent(replace_target_cnt=5000, env=env, state_space=state_space, action_space=action_space, model_name='breakout_model', gamma=.99,
                eps_strt=.1, eps_end=.001, eps_dec=5e-6, batch_size=32, lr=.001)

# Train num_eps amount of times and save onnx model
agent.train(num_eps=75000)