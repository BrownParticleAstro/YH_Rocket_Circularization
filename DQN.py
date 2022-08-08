import tensorflow as tf
import numpy as np

import collections
import pickle

from nn import create_mlp

Experience = collections.namedtuple('Experience',
                                    field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
            np.array(rewards, dtype=np.float32), \
            list(dones), np.array(next_states) \



class DeepQNetwork(tf.keras.Model):

    def __init__(self, dims, epsilon, epsilon_decay, gamma, memory, start_updating, batch_size, learning_rate, descent_frequency, update_frequency, use_target, target_frequency) -> None:
        super().__init__()

        self.q_net = create_mlp(dims, final_activation='linear')
        self.use_target = use_target

        if use_target:
            self.q_target = tf.keras.models.clone_model(self.q_net)
            self.target_frequency = target_frequency

        self.epsilon_init = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        self.memory = memory
        self.replay = ExperienceReplay(memory)
        self.start_updating = start_updating

        self.batch_size = batch_size
        self.descent_frequency = descent_frequency
        self.update_frequency = update_frequency
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    def act(self, state, evaluation=False):
        output = self.q_net(state.reshape((1, -1)))[0]
        if not evaluation and np.random.uniform() < self.epsilon:
            return np.random.randint(0, len(output))
        else:
            return tf.argmax(output).numpy()

    def simulate(self, env, render=False, evaluation=False, graph=False):
        state = env.reset()
        done = False

        iters = 0
        total_rwd = 0
        states = []

        while not done:
            if render:
                env.render()

            action = self.act(state, evaluation=evaluation)
            new_state, reward, done, _ = env.step(action)

            record = Experience(state, action, reward, done, new_state)
            self.replay.append(record)

            state = new_state

            iters += 1
            total_rwd += reward
            states.append(state)

        if render:
            env.show(summary=graph)

        return iters, total_rwd, states

    def _update_weights(self, gamma):
        for updates in range(self.descent_frequency):
            states, actions, rewards, dones, new_states = self.replay.sample(
                self.batch_size)
            if self.use_target and updates % self.target_frequency == 0:
                self.q_target.set_weights(self.q_net.get_weights())
            with tf.GradientTape() as tape:
                this_Q = tf.convert_to_tensor(
                    [q[act] for q, act in zip(self.q_net(states), actions)],
                    dtype=tf.float32)
                if self.use_target:
                    next_Q = tf.reduce_max(
                        self.q_target(new_states), axis=-1).numpy()
                else:
                    next_Q = tf.reduce_max(
                        self.q_net(new_states), axis=-1).numpy()

                y = np.where(dones, rewards, np.array(
                    rewards) + gamma * next_Q)

                loss = tf.reduce_mean(tf.math.square(y - this_Q))

                grad = tape.gradient(loss, self.q_net.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(grad, self.q_net.trainable_variables))

        return loss

    def _epsilon(self, step, total_steps):
        return self.epsilon_init * np.power(self.epsilon_decay, step / total_steps)

    def train(self, env, episodes, render_frequency, summary=False):
        for episode in range(episodes):
            print(f'Episode: {episode}')

            self.epsilon = self._epsilon(episode, episodes)
            iters, total_rwd, _ = self.simulate(
                env, render=(episode % render_frequency == 0), graph=summary)
            if len(self.replay) >= self.start_updating and episode % self.update_frequency == 0:
                loss = self._update_weights(self.gamma)
                print(
                    f'iters: {iters}, tot_rwd: {total_rwd:.3e}, lss: {loss:.3e}')
            else:
                print(f'iters: {iters}, tot_rwd: {total_rwd:.3e}')

    def save(self, path):
        self.q_net.save_weights(path)
        with open(path + 'experience.pk', 'wb+') as file:
            pickle.dump(self.replay, file)

    def load(self, path):
        self.q_net.load_weights(path)
        with open(path + 'experience.pk', 'rb') as file:
            self.replay = pickle.load(file)

    def clear_experience(self,):
        self.replay = ExperienceReplay(self.memory)
