import gym
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from itertools import chain

# Custom NewtonOptimizer class
class NewtonOptimizer(optim.Optimizer):
    """
    Implements a simplified version of Newton's Method for deep learning.
    Computes parameter updates using the inverse of the Hessian matrix.
    """
    def __init__(self, params, lr=1.0, damping=1e-4):
        """
        Args:
            params: Iterable of model parameters to optimize.
            lr: Learning rate for the parameter updates.
            damping: Damping factor for stabilizing the Hessian inversion.
        """
        defaults = {
            'lr': lr,
            'damping': damping
        }
        super(NewtonOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Args:
            closure: A closure that re-evaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('NewtonOptimizer does not support sparse gradients.')

                # Compute the Hessian-vector product using autograd's second-order derivatives
                hessian = self._compute_hessian(grad, p)
                hessian_inv = torch.linalg.pinv(hessian + group['damping'] * torch.eye(hessian.size(0)))

                # Update rule: p_new = p - lr * H_inv * grad
                update = hessian_inv @ grad.view(-1)
                p.data.add_(-group['lr'] * update.view(p.size()))

        return loss

    def _compute_hessian(self, grad, param):
        """
        Computes the Hessian matrix for the given gradient.
        Args:
            grad: The gradient tensor.
            param: The parameter tensor for which the Hessian is computed.
        Returns:
            Hessian matrix as a tensor.
        """
        grad2rd = torch.autograd.grad(grad.sum(), param, create_graph=True)[0]
        hessian = []
        for g2 in grad2rd.view(-1):
            h_row = torch.autograd.grad(g2, param, retain_graph=True)[0].view(-1)
            hessian.append(h_row)
        return torch.stack(hessian)

# CustomFeatureExtractor class for extracting features from the observation space
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
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
        pos_vel = observations[:, :4]
        specific_energy = observations[:, 5].unsqueeze(1)
        angular_momentum = observations[:, 6].unsqueeze(1)
        pos_vel_features = self.pos_vel_net(pos_vel)
        angular_features = self.angular_momentum_net(angular_momentum)
        energy_features = self.specific_energy_net(specific_energy)
        return torch.cat([pos_vel_features,  # Position/velocity features
                        angular_features,    # Angular momentum features
                        energy_features],    # KE + PE
                        dim=1)
    

# Custom policy class that uses NewtonOptimizer as the optimizer
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)

    def _make_optimizers(self):
        # Use NewtonOptimizer with specified learning rate and damping for stability
        self.optimizer = NewtonOptimizer(self.parameters(), lr=self.learning_rate, damping=1e-4)

# Function to create the PPO model using the CustomActorCriticPolicy
def create_model(env, policy_kwargs=None):
    if policy_kwargs is None:
        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(),
            net_arch=[dict(pi=[64, 32], vf=[64, 32])],  # Actor and critic network architectures
            activation_fn=nn.ReLU,
        )
    return PPO(CustomActorCriticPolicy,     # Policy type, 2 layers of 64 neurons, w Newton optimizer
               env,                         # Environment
               policy_kwargs=policy_kwargs,
               verbose=1,                   
               learning_rate=3e-4,          # Learning rate
               n_steps=2048,                # num of steps before policy update
               batch_size=64,               # num of samples per policy update calculation
               ent_coef=0.01,               # factor to encourage randomness of policy (aka exploration)
               gamma=0.9994,                # far-sighted consideration of long term reward
               )