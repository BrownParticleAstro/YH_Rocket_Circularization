import gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# CustomFeatureExtractor extracts features from the observation space for use in the PPO policy network.
# It uses separate neural networks for positional, angular momentum, energy, and timestep inputs, concatenating their outputs.
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        """
        Args: observation_space: Gym observation space object (spaces.Box).
        Returns: None. Initializes the neural networks for different input components.
        """
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=128)
        self.pos_vel_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.angular_momentum_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.specific_energy_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

    def forward(self, observations):
        """
        Forward pass to extract features from the input observations.
        Args: observations: Observation input from the environment (tensor).
        Returns: Feature tensor used by the policy network (tensor).
        """
        pos_vel = observations[:, :4]
        specific_energy = observations[:, 6].unsqueeze(1)
        angular_momentum = observations[:, 7].unsqueeze(1)
        pos_vel_features = self.pos_vel_net(pos_vel)
        angular_features = self.angular_momentum_net(angular_momentum)
        energy_features = self.specific_energy_net(specific_energy)
        return torch.cat([pos_vel_features, angular_features, energy_features], dim=1)

""" Initializes untrained network of specified structure for PPO training. """
def create_model(env, policy_kwargs=None):
    if policy_kwargs is None:
        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(),
            net_arch=[dict(pi=[64, 32], vf=[64, 32])],
            activation_fn=nn.ReLU,
        )
    return PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, ent_coef=0.01, gamma=0.999)