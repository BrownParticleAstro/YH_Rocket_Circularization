import tensorflow as tf
import numpy as np
import gym
import pickle
from tqdm import tqdm
import rocket_gym

from nn import create_mlp  # Assumed to be defined as a function to create an MLP network.

class ActorCritic(tf.keras.Model):
    '''
    An implementation of the Actor-Critic algorithm.
    '''

    def __init__(self, state_dims, action_dims, gamma, learning_rate):
        super().__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.gamma = gamma

        # Actor and Critic networks
        self.actor = create_mlp(state_dims + [action_dims], final_activation='softmax')
        self.critic = create_mlp(state_dims + [1], final_activation=None)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def act(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.actor(state)
        action = tf.random.categorical(tf.math.log(action_probs), num_samples=1)
        return action.numpy()[0, 0]

    def _update_weights(self, states, actions, rewards, next_states, dones):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            state_values = tf.squeeze(self.critic(states), 1)
            next_state_values = tf.squeeze(self.critic(next_states), 1)
            action_probs = self.actor(states)

            action_log_probs = tf.math.log(tf.gather(action_probs, actions, batch_dims=1))
            td_targets = rewards + self.gamma * next_state_values * (1 - dones)
            td_errors = td_targets - state_values
            critic_loss = tf.reduce_mean(tf.square(td_errors))
            actor_loss = -tf.reduce_mean(action_log_probs * tf.stop_gradient(td_errors))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
        self.optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        return actor_loss.numpy(), critic_loss.numpy()

    def train(self, env, episodes):
        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, info = env.step(action)
                actor_loss, critic_loss = self._update_weights([state], [action], [reward], [next_state], [done])
                state = next_state
                print(f'Episode: {episode}, Action Loss: {actor_loss}, Critic Loss: {critic_loss}')

    def save(self, path):
        self.save_weights(path + 'actor_critic_weights')

    def load(self, path):
        self.load_weights(path + 'actor_critic_weights')

# Example usage
with rocket_gym.make('RocketCircularization-v0') as env:
    env = rocket_gym.PolarizeObservation(
        rocket_gym.TangentialThrust(
            rocket_gym.PolarizeAction(env)))
    model = ActorCritic([4], 2, gamma=0.99, learning_rate=0.01)
    model.train(env, episodes=1000)
    model.save('path_to_save_model/')
