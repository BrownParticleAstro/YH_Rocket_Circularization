import rocket_gym

with rocket_gym.make('RocketCircularization-v0') as env:
    done = False
    iters = 0
    total_reward = 0
    state = env.reset()
    while not done:
        env.render()
        state, reward, done, info = env.step([0, 0])
        print(state, reward, done, info)
        iters += 1
        total_reward += reward

    print(f'Iters: {iters}, Total Reward: {total_reward}')

    env.show()
