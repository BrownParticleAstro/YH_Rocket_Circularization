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
                         use_target=True, target_frequency=8)

    with rocket_gym.make('RocketCircularization-v0') as env:
        env = rocket_gym.PolarizeObservation(
            rocket_gym.TangentialThrust(
                rocket_gym.PolarizeAction(env)))
        # env = rocket_gym.RadialObservation(
        #     rocket_gym.RadialThrust(
        #         rocket_gym.PolarizeAction(env)))
        # model.train(env, episodes=1500, render_frequency=10000, summary=True, vdo_frequency=100, vdo_path='./')
        # model.save('./dqn_test_12/')

        model.load('./bounded_2/')
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

        # rm, rdtm, tdtm = np.mgrid[.5:1.5:40j, -.5:.5:40j, .5:1.5:40j]

        # # ellipsoid
        # inputs = np.hstack(
        # (rm.reshape(-1, 1), rdtm.reshape(-1, 1), tdtm.reshape(-1, 1)))
        # # action = tf.reshape(tf.argmax(model.q_net(inputs), axis=-1), (100, 100, 100))
        # # value = tf.reshape(tf.reduce_max(model.q_net(inputs), axis=-1), (100, 100, 100))
        # action = tf.argmax(model.q_net(inputs), axis=-1)
        # value = tf.reduce_max(model.q_net(inputs), axis=-1)

        # fig = make_subplots(
        #     rows=1, cols=2,
        #     specs = [[{'type': 'isosurface'}, {'type': 'isosurface'}]],
        #     subplot_titles=('Max Q-Value', 'Action'))

        # fig.add_trace(
        #     go.Isosurface(
        #         x=rm.flatten(),
        #         y=rdtm.flatten(),
        #         z=tdtm.flatten(),
        #         # value=(action - 1).numpy().flatten(),
        #         value=value.numpy().flatten(),
        #         isomin=-20,
        #         isomax=1,
        #         surface_count=10, # number of isosurfaces, 2 by default: only min and max
        #         colorbar_nticks=5, # colorbar ticks correspond to isosurface values
        #         caps=dict(x_show=False, y_show=False),
        #         colorbar_x=.45),
        #     row=1, col=1)

        # fig.add_trace(
        #     go.Isosurface(
        #         x=rm.flatten(),
        #         y=rdtm.flatten(),
        #         z=tdtm.flatten(),
        #         value=(action - 1).numpy().flatten(),
        #         isomin=-1,
        #         isomax=1,
        #         surface_count=10, # number of isosurfaces, 2 by default: only min and max
        #         colorbar_nticks=5, # colorbar ticks correspond to isosurface values
        #         caps=dict(x_show=False, y_show=False)
        #     ),
        #     row=1, col=2
        # )

        # fig.update_layout(
        #     width=1200, height=600,
        #     scene=dict(
        #         xaxis_title='Radius',
        #         yaxis_title='Radial Velocity',
        #         zaxis_title='Tangential Velocity'),
        #     scene2=dict(
        #         xaxis_title='Radius',
        #         yaxis_title='Radial Velocity',
        #         zaxis_title='Tangential Velocity'))
        # fig.show()


if __name__ == '__main__':
    main()
