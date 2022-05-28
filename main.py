from rockect_circularization import RocketCircularization
from VPG import PolicyNetworkBaseline
import numpy as np
import wandb
project_name = 'Rocket Circularization'


config_network = {
    'actor_hidden_dims': [32, 32],
    'critic_hidden_dims': [32, 32],
    'lr': 0.001
}
config_bounds = {
    'rmin_func': 'exponential',
    'rmin_strategy': [
        {
            'name': 'constant',
            'parameters': {'const': 0.8}
        },
        {
            'name': 'constant',
            'parameters': {'const': 0.99}
        },
        {
            'name': 'constant',
            'parameters': {'const': np.exp(-4)}
        }
    ],
    'rmax_func': 'exponential',
    'rmax_strategy': [
        {
            'name': 'constant',
            'parameters': {'const': 1.2}
        },
        {
            'name': 'constant',
            'parameters': {'const': 1.01}
        },
        {
            'name': 'constant',
            'parameters': {'const': np.exp(-4)}
        }
    ]
}

config_env = {
    'max_iter': 500,
    'evaluation_steps': 0,
    'radius_range': [0.1, 2],
    'target_radius': 1,
    'dt': 0.01,
    'M': 1,
    'm': 0.01,
    'G': 1,
    'init_state': [1, 0, 0, 1.1],
    'thrust_vectors': [[.1, 0], [0, .1], [-.1, 0], [0, -.1]],
    'evaluation_penalty': 1,
    'inbounds_reward': 1,
    'thrust_penalty': .1,
    't_vec_len': 1,
    'polar': True
}
config_training = {
    'episodes': 100000,
    'gamma': 1,
    'vdo_rate': 1000, 
    'save_rate': 100,
}

with wandb.init(project=project_name, config={**config_network, **config_env, **config_training, **config_bounds}) as run:
    dims = 2
    num_thrusters = 4
    rocket_policy = PolicyNetworkBaseline(input_dims=dims * 2,
                                            output_dims=2 ** num_thrusters,
                                            **config_network)
    env = RocketCircularization(bound_config=config_bounds, **config_env)
    rocket_policy.train(env, **config_training)
