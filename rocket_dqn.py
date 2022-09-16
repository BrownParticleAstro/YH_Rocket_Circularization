import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import rocket_gym
from DQN import DeepQNetwork

from tqdm import tqdm
from timeit import default_timer as timer
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def main():

    model = DeepQNetwork(dims=[3, 128, 128, 3],
                         epsilon=1.0, epsilon_decay=.1, gamma=.95,
                         memory=100000, start_updating=50000,
                         batch_size=32, learning_rate=1e-4, descent_frequency=800, update_every=3200,
                         use_target=True, target_frequency=8, truncate=True)

    with rocket_gym.make('RocketCircularization-v1') as env:
        env = rocket_gym.PolarizeObservation(
            rocket_gym.TangentialThrust(
                rocket_gym.PolarizeAction(env)))
        # env = rocket_gym.RadialObservation(
        #     rocket_gym.RadialThrust(
        #         rocket_gym.PolarizeAction(env)))
        # model.train(env, episodes=1500, render_frequency=10000, summary=True, vdo_frequency=100, vdo_path='./')
        # model.save('./dqn_test_12/')

        model.load('./bounded_3/')
        # model.value_and_policy()
        # model.state_histogram()
        model.simulate(env, render=True, evaluation=True)
        # start = timer()
        # for _ in tqdm(range(100)):
        #     model.simulate(env, render=False, evaluation=True)
        # end = timer()
        # print(f'Total Elapsed Time: {end - start}')

        # state = env.reset()
        # done = False

        # iters = 0
        # rewards = []
        # states = []

        # while not done:
        #     env.render()

        #     action = model.act(state, evaluation=True)
        #     new_state, reward, done, _ = env.step(action)

        #     state = new_state

        #     iters += 1
        #     rewards.append(reward)
        #     states.append(state)

        # env.show()

        # print(rewards)
        # print(states)


if __name__ == '__main__':
    main()
