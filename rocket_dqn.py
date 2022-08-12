import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import rocket_gym
from DQN import DeepQNetwork


def main():
    model = DeepQNetwork(dims=[3, 128, 128, 3],
                         epsilon=1.0, epsilon_decay=.1, gamma=.95,
                         memory=100000, start_updating=10000,
                         batch_size=32, learning_rate=1e-4, descent_frequency=16, update_frequency=1,
                         use_target=True, target_frequency=8)
    # model.load('./dqn_test_5/')
    # rs = np.linspace(0.5, 1.5, 100)
    # rdts = np.linspace(-10, 10, 100)
    # rm, rdtm = np.meshgrid(rs, rdts)
    # inputs = np.hstack(
    #     (rm.reshape(-1, 1), rdtm.reshape(-1, 1)))
    # action = tf.reshape(tf.argmax(model.q_net(inputs), axis=-1), (100, 100))

    # cs = plt.contourf(rm, rdtm, action - 1, label='max Q value')
    # plt.colorbar(cs)
    # plt.xlabel('radius')
    # plt.ylabel('r dot')
    # plt.legend()
    # plt.show()

    with rocket_gym.make('RocketCircularization-v0') as env:
        env = rocket_gym.PolarizeObservation(
            rocket_gym.TangentialThrust(
                rocket_gym.PolarizeAction(env)))
        # env = rocket_gym.RadialObservation(
        #     rocket_gym.RadialThrust(
        #         rocket_gym.PolarizeAction(env)))
        # model.train(env, episodes=1000, render_frequency=1000)
        # model.save('./dqn_test_5/')

        model.load('./dqn_test_10/')
        model.simulate(env, render=True, evaluation=True)

    # with rocket_gym.make('RocketCircularization-v0') as env:
    #     state = env.reset()
    #     done = False

    #     iters = 0
    #     total_rwd = 0

    #     while not done:
    #         env.render()

    #         new_state, reward, done, _ = env.step([0, 0])

    #         state = new_state

    #         iters += 1
    #         total_rwd += reward

    #     env.show()

    # print(f'iters: {iters}, total_rwd: {total_rwd}')


if __name__ == '__main__':
    main()
