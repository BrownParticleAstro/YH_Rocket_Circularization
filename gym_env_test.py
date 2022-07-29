import rocket_gym

with rocket_gym.make('RocketCircularization-v0') as env:
    env = rocket_gym.DiscretiseAction(rocket_gym.PolarizeAction(env))
    done = False
    iters = 0
    total_reward = 0
    state = env.reset()
    while not done:
        env.render()
        state, reward, done, info = env.step(2)
        print(state, reward, done, info)
        iters += 1
        total_reward += reward

    print(f'Iters: {iters}, Total Reward: {total_reward}')

    env.show()
