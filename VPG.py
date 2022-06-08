import tensorflow as tf
import wandb
import numpy as np
import time
import os


def create_mlp(dims, activation='relu', final_activation='log_softmax'):
    assert len(dims) >= 2
    if isinstance(activation, str):
        activation = [activation] * (len(dims) - 2)
    if isinstance(activation, list):
        assert len(activation) == len(dims) - 2

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(dims[0],)))
    for dim, active in zip(dims[1:-1], activation):
        model.add(tf.keras.layers.Dense(dim, activation=active))
    model.add(tf.keras.layers.Dense(dims[-1], activation=final_activation))

    return model


class PolicyNetwork(tf.keras.Model):
    def __init__(self, input_dims, hidden_dims, output_dims, output_mode='Discrete', lr=0.001) -> None:
        '''
        Initiate a policy network with the indicated dimensions
        '''
        super(PolicyNetwork, self).__init__()
        
        print(output_mode)

        assert output_mode in {'Discrete', 'Continuous'}

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        if output_mode == 'Continuous':
            output_dims *= 2
            final_activation = 'linear'
        if output_mode == 'Discrete':
            final_activation = 'log_softmax'

        self.net = create_mlp(
            [input_dims, *hidden_dims, output_dims], final_activation=final_activation)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.iters = 0
        self.output_mode = output_mode

    def _act_discrete(self, state):
        state = state.reshape((1, -1))
        log_probs = self.net(state)
        action = tf.random.categorical(log_probs, 1)[0][0]
        log_prob = log_probs[0][action]

        return action.numpy(), log_prob

    def _act_continuous(self, state):
        state = state.reshape((1, -1))
        output = self.net(state)[0]
        mus, log_sigmas = output[:len(output) // 2], output[len(output) // 2:]
        mus = tf.math.tanh(mus)
        sigmas = tf.exp(log_sigmas)
        action = tf.random.normal(mus.shape, mus, sigmas, dtype=tf.float32)
        log_probs = -log_sigmas - tf.pow(((action - mus) / sigmas), 2) / 2
        log_prob = tf.reduce_sum(log_probs)

        return action.numpy(), log_prob

    def act(self, state):
        '''
        Given the current state, decide on the action. It also returns the log-probability of this action
        state: a numpy array (input_dims, ) indicating the current game state
        return: a tuple of the which action taken [0, input_dims) and tf.Tensor indicating log probabilty of this action
        '''
        if self.output_mode == 'Discrete':
            return self._act_discrete(state)
        if self.output_mode == 'Continuous':
            return self._act_continuous(state)

    def _discount_rewards(self, rewards, gamma=0.9):
        '''
        Calculate the discounted rewards given the rewards obtained during training
        rewards: a python list (iterations) containing the rewards obtained in each step
        gamma: a scalar, discount factor

        return: a np.array (iterations,) indicating the discounted rewards at each steop
        '''
        # Calculate the discounted rewards
        rewards = np.array(rewards)
        discount_ratios = np.array(
            [gamma ** pwr for pwr in range(len(rewards))])
        discounted_rewards = []
        for t in range(len(rewards)):
            if t == 0:
                discounted_rewards.append(np.dot(discount_ratios, rewards))
            else:
                discounted_rewards.append(
                    np.dot(discount_ratios[:-t], rewards[t:]))
        discounted_rewards = np.array(discounted_rewards)
        # Normalize rewards
        # discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-6)

        return discounted_rewards

    def loss(self, log_probs, discounted_rewards):
        '''
        Calculate the policy loss of a training step

        log_probs: a python array of shape (iterations,) of tf.Tensor constants 
        discounted_rewards: the calculated discounted rewards of this training step, np.array (iterations,)
        '''
        policy_loss = 0
        for log_prob, d_rwd in zip(log_probs, discounted_rewards):
            policy_loss -= log_prob * d_rwd

        return policy_loss

    def _train_step(self, log_probs, rewards, gamma, tape):
        '''
        Calculate the loss and update the weights

        log_probs: a python list (iterations, ) of the log probabilities during the episode
        rewards: a python list (iterations, ) of the rewards obtained for each step
        gamma: the discount ratio
        tape: the gradient tape used during training
        '''
        policy_loss = self.loss(
            log_probs, self._discount_rewards(rewards, gamma=gamma))
        gradients = tape.gradient(policy_loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.net.trainable_variables))

        return policy_loss

    def _simulate_step(self, env):
        '''
        Run the game from start to finish with the current state of the policy network

        env: The environment in which to run the game

        return: a tuple of
          log_probs: a python list (iterations, ) of the log probabilities during the episode
          rewards: a python list (iterations, ) of the rewards obtained for each step
        '''
        log_probs = []
        rewards = []
        obs = env.reset()
        done = False
        self.iters = 0

        while not done:
            # env.render()
            action, log_prob = self.act(obs)
            obs, rwd, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(rwd)
            self.iters += 1

        return log_probs, rewards

    def train(self, env, episodes=2000, gamma=0.9, vdo_rate=100, save_rate=100):
        '''
        Train the policy network with policy gradients

        env: The ai gym environment the agent is interacting with
        optimizer: a tf.keras.optimizer object indicating the optimizer used in this training
        episodes: the number of games the training will go through
        gamma: discount factor

        return: None
        '''
        wandb.define_metric("loss", summary="min")
        wandb.define_metric("rewards", summary="max")
        wandb.define_metric("iterations", summary='min')

        # Train for some eposodes
        for episode in range(episodes):
            with tf.GradientTape() as tape:
                print(f'Episode: {episode}')

                start_time = time.time()
                episode_data = self._simulate_step(env)
                simulation_time = time.time() - start_time

                start_time = time.time()
                policy_loss = self._train_step(*episode_data, gamma, tape)
                train_time = time.time() - start_time

                total_rewards = sum(episode_data[-1])

                if episode == 0:
                    max_rwd = total_rewards - 1

                # Log the training states
                print(
                    f'Iterations: {self.iters}, Rewards: {total_rewards:.3f}, Loss: {policy_loss.numpy():.3f}, SimuTime: {simulation_time:.2f}, TrainTime: {train_time:.2f}')

                wandb.log({
                    'iterations': self.iters,
                    'rewards': total_rewards,
                    'loss': policy_loss.numpy()
                })

                if total_rewards > max_rwd:
                    max_rwd = total_rewards

                    if episode > save_rate:

                        print('Saving best model..')

                        model_path = os.path.join(wandb.run.dir, 'best_model')
                        if not os.path.exists(os.path.join(wandb.run.dir, 'best_model')):
                            os.makedirs(model_path)

                        save_path = os.path.join(model_path, 'best_model.ckpt')
                        self.save_weights(save_path)
                        wandb.save(save_path + '*', base_path=wandb.run.dir)

                    if episode > save_rate:

                        media_path = os.path.join(wandb.run.dir, 'media')
                        if not os.path.exists(os.path.join(wandb.run.dir, 'media')):
                            os.makedirs(media_path)

                        vdo_path = os.path.join(media_path, f'{episode}.mp4')
                        self.play(env, vdo_path)
                        wandb.log(
                            {'play_test': wandb.Video(vdo_path, format='mp4')})

                if episode % vdo_rate == 0:
                    media_path = os.path.join(wandb.run.dir, 'media')
                    if not os.path.exists(os.path.join(wandb.run.dir, 'media')):
                        os.makedirs(media_path)

                    vdo_path = os.path.join(media_path, f'{episode}.mp4')
                    self.play(env, vdo_path)
                    wandb.log(
                        {'play_test': wandb.Video(vdo_path, format='mp4')})

                if episode % save_rate == 0:
                    model_path = os.path.join(wandb.run.dir, 'model')
                    if not os.path.exists(os.path.join(wandb.run.dir, 'model')):
                        os.makedirs(model_path)

                    save_path = os.path.join(model_path, 'model.ckpt')
                    self.save_weights(save_path)
                    wandb.save(save_path + '*', base_path=wandb.run.dir)

    def play(self, env, vdo_path='play.mp4'):
        '''
        Runs the game from start to finish

        env: The environment in which to run the game
        '''
        total_reward = 0
        obs = env.reset()
        # env.render()
        done = False
        while not done:
            # env.render()
            action, _ = self.act(obs)
            obs, rwd, done, _ = env.step(action)

            total_reward += rwd

            if done:
                break

        env.save(vdo_path)
        print(f"Total Reward: {total_reward}")


class PolicyNetworkBaseline(PolicyNetwork):
    def __init__(self, input_dims, actor_hidden_dims, output_dims, critic_hidden_dims, output_mode='Discrete', lr=0.001):

        super(PolicyNetworkBaseline, self).__init__(
            input_dims, actor_hidden_dims, output_dims, output_mode, lr)

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
            
        self.value = create_mlp([input_dims, critic_hidden_dims, 1], final_activation='linear')

    def loss(self, log_probs, values, discounted_rewards):
        '''
        Calculate the policy loss of a training step

        log_probs: a python array of shape (iterations,) of tf.Tensor constants 
        values: a python array of shape (iterations, ) of tf. Tensor constants, indicating the predicted value of each state in the episode
        discounted_rewards: the calculated discounted rewards of this training step, np.array (iterations,)
        '''
        advantage = tf.cast(tf.math.subtract(
            discounted_rewards, tf.concat(values, 0)), tf.float32)
        actor_loss = tf.tensordot(
            log_probs, -advantage.numpy(), 1) / advantage.shape[0]
        critic_loss = tf.math.sqrt(tf.reduce_mean(
            tf.math.square(advantage), axis=0))

        print(f'Actor Loss:{actor_loss:.3f}, Critic Loss:{critic_loss:.3f}')

        return actor_loss + critic_loss

    def _train_step(self, log_probs, values, rewards, gamma, tape):
        '''
        Calculate the loss and update the weights

        log_probs: a python list (iterations, ) of the log probabilities during the episode
        values: a python array of shape (iterations, ) of tf. Tensor constants, indicating the predicted value of each state in the episode
        rewards: a python list (iterations, ) of the rewards obtained for each step
        gamma: the discount ratio
        tape: the gradient tape used during training
        '''
        losses = self.loss(log_probs, values,
                           self._discount_rewards(rewards, gamma=gamma))
        gradients = tape.gradient(
            losses, self.net.trainable_variables + self.value.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.net.trainable_variables + self.value.trainable_variables))

        return losses

    def _simulate_step(self, env):
        '''
        Run the game from start to finish with the current state of the policy network

        env: The environment in which to run the game

        return: a tuple of
          log_probs: a python list (iterations, ) of the log probabilities during the episode
          rewards: a python list (iterations, ) of the rewards obtained for each step
        '''
        log_probs = []
        rewards = []
        values = []
        obs = env.reset()
        done = False
        self.iters = 0

        while not done:
            # env.render()
            action, log_prob = self.act(obs)
            value = self.value(obs.reshape(1, -1))[0]
            obs, rwd, done, _ = env.step(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(rwd)
            self.iters += 1

        return log_probs, values, rewards
