import gym
import rocket_gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Sequence, Tuple
import torch as th
import stable_baselines3
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
import tensorflow as tf

# Initialize and wrap the environment
env_name = 'RocketCircularization-v1'
env = rocket_gym.make(env_name)
env.max_step = 10_000  # maximum episode length
env = rocket_gym.PolarizeAction(env)
env = rocket_gym.RadialThrust(env)
env = rocket_gym.PolarizeObservation(env)

# Examine the observation space after wrapping
env.observation_space = gym.spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)
print("Observation space:", env.observation_space)
print("Sample observation:", env.reset())

# Set up vectorized environment
env = make_vec_env(lambda: env, n_envs=4)

# Initialize PPO model
policy_kwargs = dict(activation_fn=th.nn.SiLU, net_arch=dict(pi=[48, 48, 48, 48, 48, 48], vf=[48, 48, 48, 48, 48, 48]))
model = PPO("MlpPolicy", env, batch_size=1024, gamma=0.995, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_rocketCircularization_tensorboard/", verbose=1)

# Check if the model starts learning without error
try:
    model.learn(total_timesteps=3_141_592, progress_bar=True)
    print("Model training started successfully!")
except Exception as e:
    print("Error during training:", e)

# Save the model
model.save("ppo_rocket_circularization")

# Optionally, evaluate the model here using evaluation environments or other metrics

# Load the model
model = PPO.load("ppo_rocket_circularization")

# Function to evaluate model across a grid of states
def evaluate_action_grid(model, grid_size=50, state_dims=(0, 2)):
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Select the third index to set as zero based on unused dimension
    zero_idx = 3 - sum(state_dims)  # sum of indices 0, 1, 2 is 3, so this gives the remaining one

    for i in range(grid_size):
        for j in range(grid_size):
            obs = np.zeros(3)
            obs[state_dims[0]] = X[i, j]
            obs[state_dims[1]] = Y[i, j]
            action, _states = model.predict(obs.reshape(1, -1), deterministic=True)
            Z[i, j] = action[0]

    return X, Y, Z

# Generate action grid
X02, Y02, action_grid02 = evaluate_action_grid(model, state_dims=(0, 2))
X01, Y01, action_grid01 = evaluate_action_grid(model, state_dims=(0, 1))
X12, Y12, action_grid12 = evaluate_action_grid(model, state_dims=(1, 2))

# Plotting
plt.figure(figsize=(15, 10))

# State[0] vs State[2] (original)
plt.subplot(1, 3, 1)
contour = plt.contourf(X02, Y02, action_grid02.squeeze(), cmap='viridis')
plt.colorbar(contour)
plt.xlabel('Radius')
plt.ylabel('Tangential Velocity')
plt.title('Action vs. Radius & Tangential Velocity')

# State[0] vs State[1]
plt.subplot(1, 3, 2)
contour = plt.contourf(X01, Y01, action_grid01.squeeze(), cmap='viridis')
plt.colorbar(contour)
plt.xlabel('Radius')
plt.ylabel('Radial Velocity')
plt.title('Action vs. Radius & Radial Velocity')

# State[1] vs State[2]
plt.subplot(1, 3, 3)
contour = plt.contourf(X12, Y12, action_grid12.squeeze(), cmap='viridis')
plt.colorbar(contour)
plt.xlabel('Radial Velocity')
plt.ylabel('Tangential Velocity')
plt.title('Action vs. Radial & Tangential Velocity')

plt.tight_layout()
plt.show()

# @tf.numpy_function(Tout=[tf.float32, tf.int32, tf.int32])
# def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# #(array([-0.15788858, -1.04781004,  1.43974995,  0.19851264]), -0.1183308541603517, False, False, {})
#   state, reward, done,truncated, info = env.step(action)
#   return (state.astype(np.float32),
#           np.array(reward, np.int32),
#           np.array(done, np.int32))

# def run_episode_for_test(
#     initial_state: tf.Tensor,
#     model: tf.keras.Model,
#     max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
#   """Runs a single episode to collect training data."""

#   action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
#   values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
#   rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

#   initial_state_shape = initial_state.shape
#   state = initial_state

#   for t in tf.range(max_steps):
#     # Convert state into a batched tensor (batch size = 1)
#     state = tf.expand_dims(state, 0)

#     # Run the model and to get action probabilities and critic value
#     action_logits_t, value = model(state)

#     # Sample next action from the action probability distribution
#     action = tf.random.categorical(action_logits_t, 1)[0, 0]
#     action_probs_t = tf.nn.softmax(action_logits_t)

#     # Store critic values
#     values = values.write(t, tf.squeeze(value))

#     # Store log probability of the action chosen
#     action_probs = action_probs.write(t, action_probs_t[0, action])
#     env.render()
#     # Apply action to the environment to get next state and reward
#     state, reward, done = env_step(action)
#     state.set_shape(initial_state_shape)

#     # Store reward
#     rewards = rewards.write(t, reward)

#     if tf.cast(done, tf.bool):
#       break

#   action_probs = action_probs.stack()
#   values = values.stack()
#   rewards = rewards.stack()
#   env.show(path='test.mp4')