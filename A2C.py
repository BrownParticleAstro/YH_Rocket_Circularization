import tensorflow as tf
import numpy as np
import gym
import pickle
from typing import Optional, Tuple, List

from nn import create_mlp

class ActorCriticNetwork(tf.keras.Model):
    def __init__(self,
                 state_dims: int,
                 action_dims: int,
                 actor_dims: List[int],
                 critic_dims: List[int],
                 gamma: float,
                 memory: int,
                 batch_size: int,
                 actor_lr: float,
                 critic_lr: float) -> None:
        super().__init__()

        self.state_dims = state_dims
        self.action_dims = action_dims
        self.gamma = gamma

        self.memory = memory
        self.replay = []
        self.batch_size = batch_size

        self.actor = create_mlp(actor_dims + [action_dims], final_activation='tanh')
        self.critic = create_mlp(critic_dims + [1])
        self.actor_opt = tf.optimizers.Adam(actor_lr)
        self.critic_opt = tf.optimizers.Adam(critic_lr)

    def act(self, state: np.ndarray) -> np.ndarray:
        action = self.actor(np.expand_dims(state, 0))[0]
        return action.numpy()

    def simulate(self, env: gym.Env, render: bool = False) -> Tuple[int, float, List[np.ndarray]]:
        state = env.reset()
        done = False

        iters = 0
        total_rwd = 0
        states = []

        while not done:
            if render:
                env.render()

            print(state)
            action = self.act(state)
            print(action)
            new_state, reward, done, truncated, _ = env.step(action)

            self.replay.append((state, action, reward, new_state, done))

            state = new_state

            iters += 1
            total_rwd += reward
            states.append(state)

        return iters, total_rwd, states

    def _update_weights(self) -> Tuple[float, float]:
        states, actions, rewards, new_states, dones = zip(*self.replay)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        new_states = np.array(new_states)
        dones = np.array(dones, dtype=np.float32)

        with tf.GradientTape(persistent=True) as tape:
            values = self.critic(states)
            new_values = self.critic(new_states)

            targets = rewards + self.gamma * new_values * (1 - dones)
            critic_loss = tf.reduce_mean(tf.square(targets - values))

            actions_pred = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic(states, actions_pred))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)

        self.critic_opt.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
        self.actor_opt.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        del tape

        return critic_loss, actor_loss

    def train(self, env: gym.Env, episodes: int, render_frequency: int) -> None:
        for episode in range(episodes):
            print(f'Episode: {episode}')

            iters, total_rwd, _ = self.simulate(env, render=(episode % render_frequency == 0))
            print(f'iters: {iters}, tot_rwd: {total_rwd:.3e}')

            if len(self.replay) >= self.batch_size:
                critic_loss, actor_loss = self._update_weights()
                print(f'critic_loss: {critic_loss:.3e}, actor_loss: {actor_loss:.3e}')
                self.replay = []

    def save(self, path: str) -> None:
        self.actor.save_weights(path + 'actor.h5')
        self.critic.save_weights(path + 'critic.h5')

        with open(path + 'experience.pk', 'wb+') as file:
            pickle.dump(self.replay, file)

    def load(self, path: str) -> None:
        self.actor.load_weights(path + 'actor.h5')
        self.critic.load_weights(path + 'critic.h5')

        with open(path + 'experience.pk', 'rb') as file:
            self.replay = pickle.load(file)

    def clear_experience(self) -> None:
        self.replay = []

    def state_histogram(self) -> None:
        raise NotImplementedError

    def value_and_policy(self) -> None:
        raise NotImplementedError