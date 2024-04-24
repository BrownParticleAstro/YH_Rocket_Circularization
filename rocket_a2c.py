import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import rocket_gym
from A2C import ActorCriticNetwork

from tqdm import tqdm
from timeit import default_timer as timer
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def main():
    model = ActorCriticNetwork(state_dims=3, action_dims=3,
                                actor_dims=[64, 64], critic_dims=[64, 64],
                                gamma=0.99, memory=100000, batch_size=64,
                                actor_lr=1e-4, critic_lr=1e-3)

    with rocket_gym.make('RocketCircularization-v1') as env:
        env = rocket_gym.PolarizeObservation(
            rocket_gym.TangentialThrust(
                rocket_gym.PolarizeAction(env)))

        # model.train(env, episodes=1500, render_frequency=10)
        model.load('./actor_critic_model/')
        model.simulate(env, render=True, evaluation=True)


if __name__ == '__main__':
    main()
