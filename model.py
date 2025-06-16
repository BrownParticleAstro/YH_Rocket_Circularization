import torch
import torch.nn as nn
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    """
    The policy network (the "actor") that learns which action to take.
    It takes the current state as input and outputs the parameters of a
    probability distribution over the continuous action space.

    Args:
        state_dim (int): The dimensionality of the state space.
        action_dim (int): The dimensionality of the action space.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256), 
            nn.Tanh(), 
            nn.Linear(256, 256), 
            nn.Tanh()
        )
        # The policy head outputs the mean of the action distribution
        self.mean_head = nn.Linear(256, action_dim)
        # A separate head outputs the log standard deviation of the action distribution
        self.log_std_head = nn.Linear(256, action_dim)
        
        # Initialize the output layer for the standard deviation.
        # A small negative bias encourages smaller initial standard deviations,
        # leading to more stable exploration at the beginning of training.
        self.log_std_head.bias.data.fill_(-1.0)
        self.log_std_head.weight.data.fill_(0.0) # Start with uniform std
        
    def forward(self, state):
        """
        Performs a forward pass through the network.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            torch.distributions.Normal: A Normal distribution over the actions.
        """
        x = self.network(state)
        # We use a Normal distribution for the continuous action space.
        mean = self.mean_head(x)
        # Clamp log_std for numerical stability. exp(log_std) gives the actual std.
        log_std = torch.clamp(self.log_std_head(x), -5.0, -0.5)
        std = torch.exp(log_std)
        return Normal(mean, std)

class ValueNetwork(nn.Module):
    """
    The value network (the "critic") that learns to estimate the expected
    return (value) from a given state. This is used to compute the advantage
    function, which guides the policy updates.

    Args:
        state_dim (int): The dimensionality of the state space.
    """
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256), 
            nn.Tanh(), 
            nn.Linear(256, 256), 
            nn.Tanh(), 
            nn.Linear(256, 1) # Outputs a single scalar value for the state
        )
        
    def forward(self, state):
        """
        Performs a forward pass to estimate the value of the state.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            torch.Tensor: A tensor containing the estimated value of the state.
        """
        return self.network(state)