import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import gym

import collections
import pickle

from tqdm import tqdm

from typing import (
    Any,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    Dict
)

from nn import create_mlp

Experience = collections.namedtuple('Experience',
                                    field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceReplay:
    '''
    Experience Relay, used by DQN.

    Records and samples experiences. Discards Old Experiences.

    Code adapted from
    https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c
    '''

    def __init__(self, capacity: int) -> None:
        '''
        Initializes a replay buffer

        buffer: Structure that stores the experiences. It discards old
                elements when the capacity is full
        updates_since_last_batch: number of data entries since the last
                time the buffer is sampled
        '''
        self.buffer: collections.deque = collections.deque(maxlen=capacity)
        self.num_updates: int = 0
        self.updates_since_last_batch: int = 0

    def __len__(self) -> int:
        '''
        Number of records in the buffer

        Return: 
            number of elements in the buffer
        '''
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        '''
        Add a new experience record to the buffer

        experience: an experience record
        '''
        self.buffer.append(experience)
        self.updates_since_last_batch += 1
        self.num_updates += 1

    def updates_since_sample(self,) -> int:
        '''
        Returns the number of new data points since the last time the data is sampled

        Return:
            the number of new data points since the last time the data is sampled
        '''
        return self.updates_since_last_batch

    def updates(self,) -> int:
        '''
        Returns the total number of buffer updates

        Return:
            The total number of buffer updates
        '''
        return self.num_updates

    def sample(self, batch_size: int) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, List, np.ndarray]:
        '''
        Uniformly sample a batch of experiences from the buffer

        batch_size: the size of the batch sampled

        Return:
            5 lists, representing the state, actions, reward, done, and new
            state respectively
        '''

        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])

        self.updates_since_last_batch = 0
        return np.array(states), np.array(actions), \
            np.array(rewards, dtype=np.float32), \
            list(dones), np.array(next_states) \



class DeepQNetwork(tf.keras.Model):
    '''
    An implementation of Deep Q Network (DQN) algorithm with target network
    '''

    def __init__(self,
                 dims: List[int],
                 epsilon: float, epsilon_decay: float, gamma: float,
                 memory: int, start_updating: int,
                 batch_size: int, learning_rate: float,
                 descent_frequency: int, update_every: int,
                 use_target: bool, target_frequency: int) -> None:
        '''
        Initializes a DQN agent


        dims: the configuration of the feed-forward MLP, elements in the list
                represents the number of neurons in each layer


        epsilon: Off-policy probability, the probability during training that
                the agent takes a random action.
        epsilon_decay: rate of decay of epsilon. Final value of epislon.
        gamma: Discount rate for discounted future reward


        memory: size of the experience buffer
        start_updating: the number of samples in the experience buffer
                before the network starts updating
        batch_size: sample size each time the network updates
        learning_rate: the learning rate during update


        update_every: minimum number of new experiences generated between 
                each network update
        descent_frequency: number of gradient descents during each update
        use_target: if DQN use target networks while updating. When false, 
                would have the same effect as target_frequency=1
        target_frequency: number of gradient descents between each update 
                to target network
        '''
        super().__init__()

        self.dims = dims
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
        self.num_updates = 0

        self.batch_size = batch_size
        self.descent_frequency = descent_frequency
        self.update_every = update_every
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    def act(self, state: np.ndarray, evaluation: bool = False) -> np.int32:
        '''
        Given the current state, pass it through the network for the best
        action with a probability of (1 - epsilon). 
        Otherwise, choose a random action.
        Randomness is disabled during evaluation.

        state: the observation from environment, 1D array
        evaluation: if the model is in evaluation mode

        Return: 
            The action being performed
        '''
        if not evaluation and np.random.uniform() < self.epsilon:
            return np.random.randint(0, self.dims[-1])
        else:
            output = self.q_net(state.reshape((1, -1)))[0]
            return tf.argmax(output).numpy()

    def simulate(self, env: gym.Env,
                 render: bool = False, evaluation: bool = False,
                 graph: bool = False, path: Optional[str] = None) \
            -> Tuple[int, np.float32, List[np.ndarray]]:
        '''
        Run one episode of the environment with network predictions. 
        Record experiences in the replay buffer.

        env: The gym environment to run the episode in

        evaluation: Whether to act with epsilon-randomness. When in 
                evaluation, outputs are deterministic, and experiences
                are not saved
        render: Whether to generate an animation or summary after the 
                simulation
        graph: Whether to generate a summary. Only valid when `render=True`
        path: If generate animation, the path to the save file location.
                If None, the animation is shown through a pop-up window. 
                Pop-up animation will not work with Jupyter Notebook.
                Only valid when `graph=False`.

        Return:
            A tuple of 
                Number of iterations in the episode, 
                Total reward obtained, and
                A record of states
        '''
        state = env.reset()
        done = False

        iters = 0
        total_rwd = 0
        states = []

        while not done:
            if render:
                env.render()

            action = self.act(state, evaluation=evaluation)
            new_state, reward, done, truncated, _ = env.step(action)

            if truncated:
                break

            if not evaluation:
                record = Experience(state, action, reward, done, new_state)
                self.replay.append(record)

            state = new_state

            iters += 1
            total_rwd += reward
            states.append(state)

        if render:
            env.show(summary=graph, path=path)

        return iters, total_rwd, states

    def _update_weights(self, gamma: float) -> tf.float32:
        '''
        Perform an update on the Q-Network weights with the Bellman loss.
        A number of gradient descent iterations are taken with the optimizer.
        Each time, a batch of experiences are sampled from the replay buffer.

        gamma: discount factor used in the update

        Return:
            mean batch loss on the last update
        '''

        # Descent multiple times
        pbar = tqdm(range(self.descent_frequency))
        for updates in pbar:
            # Sample from replay buffer
            states, actions, rewards, dones, new_states = self.replay.sample(
                self.batch_size)
            # Update target network if needed
            if self.use_target and self.num_updates % self.target_frequency == 0:
                self.q_target.set_weights(self.q_net.get_weights())
            self.num_updates += 1

            with tf.GradientTape() as tape:
                # Calculate Bellman Loss
                this_Q = tf.gather(self.q_net(states), actions, batch_dims=1)
                if self.use_target:
                    # Double DQN, prevents overestimation
                    next_actions = tf.argmax(self.q_net(new_states), axis=-1)
                    next_Q = tf.gather(self.q_target(
                        new_states), next_actions, batch_dims=1)

                    # OG Target Network
                    # next_Q = tf.reduce_max(
                    #     self.q_target(new_states), axis=-1).numpy()
                else:
                    next_Q = tf.reduce_max(
                        self.q_net(new_states), axis=-1).numpy()

                y = np.where(dones, rewards, np.array(
                    rewards) + gamma * next_Q)

                loss = tf.reduce_mean(tf.math.square(y - this_Q))

                # Descend to reduce Bellman loss
                grad = tape.gradient(loss, self.q_net.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(grad, self.q_net.trainable_variables))
            pbar.set_postfix({'loss': loss.numpy()})

        return loss

    def _epsilon(self, step: int, total_steps: int) -> float:
        '''
        Anneal the epsilon exponentially based on initial epsilon value

        epsilon = epsilon_initial * decay^(step / total_steps)

        step: Current step in the annealing
        total_steps: Total number of steps to be performed

        Return:
            The current epsilon value
        '''
        return self.epsilon_init * np.power(self.epsilon_decay, step / total_steps)

    def train(self, env: gym.Env,
              episodes: int, render_frequency: int,
              model_save_path: Optional[str] = None,
              summary: bool = False,
              vdo_frequency: Optional[int] = None,
              vdo_path: Optional[str] = None) -> None:
        '''
        Train the DQN agent for a certain number of episodes

        episodes: Number of episodes (simulations) to train the agent

        model_save_path: The folder in which the model is saved every gradient descent
                step. If None, it is not saved. default None
        render_frequency: Number of episodes before rendering an animation
                or summary, for visualization.
        summary: Whether to present the record as a summary graph. Note
                that animations are pop-up and does not work with Jupyter
                Notebook. After viewing the pop-up animation, it needs to be
                closed for the training to continue.
                (Just set summary=True whenever you can)

        vdo_frequency: Number of episodes before a video is generated and saved.
                If None, no video is generated. A vdo path is required if a vdo is generated.
                default None
        vdo_path: the path to which the video is saved, renamed with episode name
        '''
        # Record until starts updating
        playouts = 0
        while len(self.replay) < self.start_updating:
            print(f'Random Playout {playouts}')
            iters, total_rwd, _ = self.simulate(env, render=False)
            print(f'iters: {iters}, tot_rwd: {total_rwd:.3e}')
            playouts += 1

        for episode in range(episodes):
            print(f'Episode: {episode}')

            # No change to epsilon until first update
            self.epsilon = self._epsilon(episode, episodes)

            # Simulate Step
            iters, total_rwd, _ = self.simulate(
                env, render=(episode % render_frequency == 0), graph=summary)
            print(
                f'iters: {iters}, tot_rwd: {total_rwd:.3e}, tot_updates: {self.replay.updates()}')

            # Video with evaluation
            if vdo_frequency is not None and episode % vdo_frequency == 0:
                assert vdo_path is not None
                _, _, _ = self.simulate(
                    env, render=True, evaluation=True, path=vdo_path+f'{episode}.mp4')

            # Update when replay buffer is large enough to not cause over-fitting
            if self.replay.updates_since_sample() >= self.update_every:
                loss = self._update_weights(self.gamma)
                if model_save_path is not None:
                    self.save(model_save_path)

            if episode % render_frequency == 0:
                self.state_histogram()
                # self.value_and_policy()

    def save(self, path: str) -> None:
        '''
        Save the model in the specified path. Both weights are replay buffers
        are saved. 
        Note: the replay buffer can be a quite large file sometimes.

        path: The folder path to save the model
        '''
        # Save weights
        self.q_net.save_weights(path)

        # Save replay buffer
        with open(path + 'experience.pk', 'wb+') as file:
            pickle.dump(self.replay, file)

    def load(self, path: str) -> None:
        '''
        Load a model from a specified path. The network needs to have the 
        same dimensions as the saved model.

        path: The folder path from which the model is loaded
        '''
        # Load Weights
        self.q_net.load_weights(path)

        # Load Replay Buffer
        with open(path + 'experience.pk', 'rb') as file:
            self.replay = pickle.load(file)

    def clear_experience(self,) -> None:
        '''
        Clear the replay buffer.
        '''
        self.replay = ExperienceReplay(self.memory)

    def state_histogram(self,) -> None:
        '''
        Show a histogram of distribution of states
        '''
        states, _, _, _, _ = zip(*self.replay.buffer)
        states = np.array(states)
        rs = states[:, 0]
        plt.hist(rs, bins=20)
        plt.title('Distibution of radius in replay buffer')
        plt.xlabel('Radius $r$')
        plt.ylabel('Number of samples')
        plt.show()

    def value_and_policy(self,) -> None:
        '''
        Show the current value and policy
        '''
        rm, rdtm, tdtm = np.mgrid[.5:1.5:40j, -.5:.5:40j, .5:1.5:40j]

        inputs = np.hstack(
            (rm.reshape(-1, 1), rdtm.reshape(-1, 1), tdtm.reshape(-1, 1)))
        # action = tf.reshape(tf.argmax(model.q_net(inputs), axis=-1), (100, 100, 100))
        # value = tf.reshape(tf.reduce_max(model.q_net(inputs), axis=-1), (100, 100, 100))
        action = tf.argmax(self.q_net(inputs), axis=-1)
        value = tf.reduce_max(self.q_net(inputs), axis=-1)

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'isosurface'}, {'type': 'isosurface'}]],
            subplot_titles=('Max Q-Value', 'Action'))

        fig.add_trace(
            go.Isosurface(
                x=rm.flatten(),
                y=rdtm.flatten(),
                z=tdtm.flatten(),
                # value=(action - 1).numpy().flatten(),
                value=value.numpy().flatten(),
                opacity=.6,
                isomin=np.min(value),
                isomax=np.max(value),
                surface_count=10,  # number of isosurfaces, 2 by default: only min and max
                colorbar_nticks=5,  # colorbar ticks correspond to isosurface values
                caps=dict(x_show=False, y_show=False),
                colorbar_x=.45),
            row=1, col=1)

        fig.add_trace(
            go.Isosurface(
                x=rm.flatten(),
                y=rdtm.flatten(),
                z=tdtm.flatten(),
                value=(action - 1).numpy().flatten(),
                opacity=.6,
                isomin=-1,
                isomax=1,
                surface_count=10,  # number of isosurfaces, 2 by default: only min and max
                colorbar_nticks=5,  # colorbar ticks correspond to isosurface values
                caps=dict(x_show=False, y_show=False)
            ),
            row=1, col=2
        )

        fig.update_layout(
            width=1200, height=600,
            scene=dict(
                xaxis_title='Radius',
                yaxis_title='Radial Velocity',
                zaxis_title='Tangential Velocity'),
            scene2=dict(
                xaxis_title='Radius',
                yaxis_title='Radial Velocity',
                zaxis_title='Tangential Velocity'))
        fig.show()
