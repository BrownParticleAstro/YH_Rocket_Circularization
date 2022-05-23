import tensorflow as tf
import wandb
import numpy as np
import time


class PolicyNetwork(tf.keras.Model):
  def __init__(self, input_dims, hidden_dims, output_dims, lr=0.001) -> None:
    '''
    Initiate a policy network with the indicated dimensions
    '''
    super(PolicyNetwork, self).__init__()
    self.net = tf.keras.Sequential([
      tf.keras.layers.Dense(hidden_dims, activation='relu', input_shape=(input_dims,)),
      tf.keras.layers.Dense(output_dims, activation='log_softmax')
    ])
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    self.iters = 0
    
  def act(self, state):
    '''
    Given the current state, decide on the action. It also returns the log-probability of this action
    state: a numpy array (input_dims, ) indicating the current game state
    return: a tuple of the which action taken [0, input_dims) and tf.Tensor indicating log probabilty of this action
    '''
    state = state.reshape((1, -1))
    log_probs = self.net(state)
    action = tf.random.categorical(log_probs, 1)[0][0]
    log_prob = log_probs[0][action]
    return action.numpy(), log_prob
    
  def _discount_rewards(self, rewards, gamma=0.9):
    '''
    Calculate the discounted rewards given the rewards obtained during training
    rewards: a python list (iterations) containing the rewards obtained in each step
    gamma: a scalar, discount factor
        
    return: a np.array (iterations,) indicating the discounted rewards at each steop
    '''
    # Calculate the discounted rewards
    rewards = np.array(rewards)
    discount_ratios = np.array([gamma ** pwr for pwr in range(len(rewards))])
    discounted_rewards = []
    for t in range(len(rewards)):
      if t == 0:
        discounted_rewards.append(np.dot(discount_ratios, rewards))
      else:
        discounted_rewards.append(np.dot(discount_ratios[:-t], rewards[t:]))
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
    policy_loss = self.loss(log_probs, self._discount_rewards(rewards, gamma=gamma))
    gradients = tape.gradient(policy_loss, self.net.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))

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
    
  def train(self, env, episodes=2000, gamma=0.9):
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
        start_time = time.time()
        episode_data = self._simulate_step(env)
        simulation_time = time.time() - start_time
        start_time = time.time()
        policy_loss = self._train_step(*episode_data, gamma, tape)
        train_time = time.time() - start_time

        total_rewards = sum(episode_data[-1])

        # Log the training states
        print(f'Episode: {episode}, Iterations: {self.iters}, Rewards: {total_rewards:.3f}, Loss: {policy_loss.numpy():.3f}, SimuTime: {simulation_time:.2f}, TrainTime: {train_time:.2f}')

        wandb.log({
            'iterations': self.iters, 
            'rewards': total_rewards,
            'loss': policy_loss.numpy()
        })
        self.save_weights('model.tf')
        wandb.save('model.tf')

        if episode % 500 == 0:
            vdo_path = f'{episode}.mp4'
            self.play(env, vdo_path)
            wandb.log({'play_test': wandb.Video(vdo_path, format='mp4')})

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
                
      if done: break
    
    env.save(vdo_path)
    print(f"Total Reward: {total_reward}")

class PolicyNetworkBaseline(PolicyNetwork):
  def __init__(self, input_dims, actor_hidden_dims, output_dims, critic_hidden_dims, lr=0.001):

    super(PolicyNetworkBaseline, self).__init__(input_dims, actor_hidden_dims, output_dims, lr)

    self.value = tf.keras.Sequential([
      tf.keras.layers.Dense(critic_hidden_dims, activation='relu', input_shape=(input_dims,)),
      tf.keras.layers.Dense(1)
    ])

  def loss(self, log_probs, values, discounted_rewards):
    '''
    Calculate the policy loss of a training step
        
    log_probs: a python array of shape (iterations,) of tf.Tensor constants 
    values: a python array of shape (iterations, ) of tf. Tensor constants, indicating the predicted value of each state in the episode
    discounted_rewards: the calculated discounted rewards of this training step, np.array (iterations,)
    '''
    advantage = tf.cast(tf.math.subtract(discounted_rewards, tf.concat(values, 0)), tf.float32)
    actor_loss = tf.tensordot(log_probs, -advantage.numpy(), 1) / advantage.shape[0]
    critic_loss = tf.math.sqrt(tf.reduce_mean(tf.math.square(advantage), axis=0))

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
    losses = self.loss(log_probs, values, self._discount_rewards(rewards, gamma=gamma))
    gradients = tape.gradient(losses, self.net.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))

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