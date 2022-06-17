import os
from initial_condition import DEFAULT_INITIAL_CONDITION
from rocket_circularization import RocketCircularization
#from LQR import LQR
from VPG import PolicyNetworkBaseline
import numpy as np
import wandb
project_name = 'Rocket Circularization'


config_network = {
    'actor_hidden_dims': [32, 32],
    'critic_hidden_dims': [32, 32],
    'lr': 1e-3
}
config_init_cond = {
    'function': 'rotated_state',
    'parameters': {'st': [1, 0, 0, 1.25], 'random': True}
}
config_bounds = {
    'rmin_func': 'constant',
    'rmin_strategy': [
        {
            'name': 'constant',
            'parameters': {'const': 0.1}
        }
    ],
    'rmax_func': 'constant',
    'rmax_strategy': [
        {
            'name': 'constant',
            'parameters': {'const': 2}
        }
    ]
}

config_env = {
    'max_iter': 500,
    'evaluation_steps': 0,
    'iter_steps': 10,
    'radius_range': [0.1, 2],
    'ignore_bounds': True,
    'target_radius': 1,
    'dt': 0.01,
    'M': 1,
    'm': 0.01,
    'G': 1,
    'init_state': config_init_cond,
    'thrust_vectors': [[.1, 0], [0, .1], [-.1, 0], [0, -.1]],
    'max_thrust': .2,
    'circularization_penalty': 1,
    'evaluation_penalty': 1,
    'inbounds_reward': 0,
    'thrust_penalty': .01,
    'penalty_mode': 'Quadratic',
    't_vec_len': 100,
    'thrust_mode': 'Continuous',
    'state_output_mode': 'Offset LR',
    'clip': True
}
config_training = {
    'episodes': 100000,
    'gamma': 0.9,
    'vdo_rate': 1000,
    'save_rate': 100,
    'summary_rate': 1
}

'''with wandb.init(project=project_name, config={**config_network, **config_env, **config_training, **config_bounds}) as run:
    env = RocketCircularization(bound_config=config_bounds, **config_env)
    rocket_policy = PolicyNetworkBaseline(input_dims=env.get_state_dims(),
                                          output_dims=env.get_action_dims(),
                                          output_mode=env.get_thrust_mode(),
                                          **config_network)
    rocket_policy.train(env, **config_training)'''

with wandb.init(project=project_name, id='33penxaz', resume=True) as run:
    env = RocketCircularization(bound_config=config_bounds, **config_env)
    rocket_policy = PolicyNetworkBaseline(input_dims=env.get_state_dims(),
                                          output_dims=env.get_action_dims(),
                                          output_mode=env.get_thrust_mode(),
                                          **config_network)
    restored_index = wandb.restore('model/model.ckpt.index')
    wandb.restore('model/model.ckpt.data-00000-of-00001')
    rocket_policy.load_weights(os.path.splitext(restored_index.name)[0])
    rocket_policy.train(env, **config_training)

'''config_init_cond = {
    'function': 'rotated_state',
    'parameters': {'st': [1, 0, 0, 1.25], 'random': False}
}

config_env = {
    'max_iter': 100,
    'evaluation_steps': 0,
    'iter_steps': 10,
    'radius_range': [0.1, 2],
    'target_radius': 1,
    'dt': 0.01,
    'M': 1,
    'm': 0.01,
    'G': 1,
    'init_state': config_init_cond,
    'max_thrust': .2,
    'circularization_penalty': 1,
    'evaluation_penalty': 1,
    'inbounds_reward': 1,
    'thrust_penalty': 0,
    't_vec_len': 100,
    'state_output_mode': 'No Theta',
    'state_target_r': True,
    'thrust_mode': 'Continuous',
    'clip': True
}

env = RocketCircularization(**config_env)
actor = LQR(mu=1, l_penalty=True, thrust_penalty=.01)

obs = env.reset()
done = False
total_reward = 0
iters = 0

l0 = 1
r0 = 1

while not done:
    if iters % 10 == 0:
        print(obs)
    obs, rwd, done, _ = env.step(actor.act(obs) / config_env['max_thrust'])

    total_reward += rwd
    iters += 1

print(total_reward, iters)
env.save('LQR-transfer-to-given-orbit.mp4')'''
