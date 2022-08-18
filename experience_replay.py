import collections

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

import numpy as np

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



class PrioritizedExperienceReplay:
    '''
    Prioritized Experience Replay, used by DQN

    Records and samples experiences based on priority

    Adapted from https://nn.labml.ai/rl/dqn/replay_buffer.html 
    '''

    def __init__(self, capacity: int, alpha: float, beta: float, state_size: int) -> None:
        """
        Initializes the buffer

        capacity: the capacity of the buffer
        alpha: the alpha value
        beta: the beta value
        state_size: the shape of the state vector
        """
        # We use a power of $2$ for capacity because it simplifies the code and debugging
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta

        # Maintain segment binary trees to take sum and find minimum over a range
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]

        # Current max priority, $p$, to be assigned to new transitions
        self.max_priority = 1.

        # Arrays for buffer
        self.data = {
            'obs': np.zeros(shape=(capacity, state_size), dtype=np.uint8),
            'action': np.zeros(shape=capacity, dtype=np.int32),
            'reward': np.zeros(shape=capacity, dtype=np.float32),
            'next_obs': np.zeros(shape=(capacity, state_size), dtype=np.uint8),
            'done': np.zeros(shape=capacity, dtype=np.bool)
        }

        # keeps the index of the next empty slot in the cyclic buffer
        self.next_idx = 0

        # Size of the buffer
        self.size = 0

        # Number of samples entered
        self.num_updates: int = 0

        # Number of samples collected since last time a batch is sampled
        self.updates_since_last_batch: int = 0

    def __len__(self,) -> int:
        '''
        Returns number of records in the buffer
        '''
        return self.size

    def append(self, experience: Experience) -> None:
        '''
        Add a new experience record to the buffer

        experience: an experience record
        '''

        obs, action, reward, done, next_obs = experience

        # Get next available slot
        idx = self.next_idx

        # store in the queue
        self.data['obs'][idx] = obs
        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['next_obs'][idx] = next_obs
        self.data['done'][idx] = done

        # Increment next available slot
        self.next_idx = (idx + 1) % self.capacity
        # Calculate the size
        self.size = min(self.capacity, self.size + 1)

        # $p_i^\alpha$, new samples get `max_priority`
        priority_alpha = self.max_priority ** self.alpha
        # Update the two segment trees for sum and minimum
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

        self.updates_since_last_batch += 1
        self.num_updates += 1

    def _set_priority_min(self, idx: int, priority_alpha: float) -> None:
        """
        Set priority in binary segment tree for minimum

        idx: the index of the priority being set
        priority_alpha: the power-adjusted priority value being set
        """

        # Leaf of the binary tree
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the minimum of it's two children
            self.priority_min[idx] = min(
                self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        """
        Set priority in binary segment tree for sum

        idx: the index of the priority being set
        priority_alpha: the power-adjusted priority value being set
        """

        # Leaf of the binary tree
        idx += self.capacity
        # Set the priority at the leaf
        self.priority_sum[idx] = priority

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the sum of it's two children
            self.priority_sum[idx] = self.priority_sum[2 * idx] + \
                self.priority_sum[2 * idx + 1]

    def _sum(self) -> float:
        """
        Returns the sum of all power-adjusted priority values
        """

        # The root node keeps the sum of all values
        return self.priority_sum[1]

    def _min(self):
        """
        Returns the minimum power-adjusted priority values
        """

        # The root node keeps the minimum of all values
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum: float) -> int:
        """
        Find largest $i$ such that $\sum_{k=1}^{i} p_k^\alpha  \le P$

        Return:
            The index of the last value whose prefix sum is smaller than the value
        """

        # Start from the root
        idx = 1
        while idx < self.capacity:
            # If the sum of the left branch is higher than required sum
            if self.priority_sum[idx * 2] > prefix_sum:
                # Go to left branch of the tree
                idx = 2 * idx
            else:
                # Otherwise go to right branch and reduce the sum of left
                #  branch from required sum
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        # We are at the leaf node. Subtract the capacity by the index in the tree
        # to get the index of actual value
        return idx - self.capacity

    def sample(self, batch_size: int) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                     np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample from buffer with the priorities specified

        Return:
            7 lists, representing the state, actions, reward, done, new
            states, weights, and indices respectively
        """

        # Initialize samples
        indices = np.zeros(shape=batch_size, dtype=np.int32)

        # Get sample indexes
        for i in range(batch_size):
            p = np.random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            indices[i] = idx

        # $\min_i P(i) = \frac{\min_i p_i^\alpha}{\sum_k p_k^\alpha}$
        prob_min = self._min() / self._sum()
        # $\max_i w_i = \bigg(\frac{1}{N} \frac{1}{\min_i P(i)}\bigg)^\beta$
        max_weight = (prob_min * self.size) ** (-self.beta)

        # $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$
        probs = self.priority_sum[indices + self.capacity] / self._sum()
        # $w_i = \bigg(\frac{1}{N} \frac{1}{P(i)}\bigg)^\beta$
        weights = np.power(probs * self.size, -self.beta)
        # Normalize by $\frac{1}{\max_i w_i}$,
        #  which also cancels off the $\frac{1}{N}$ term
        weights = weights / max_weight

        # Get samples data
        states = self.data['obs'][indices]
        actions = self.data['action'][indices]
        rewards = self.data['reward'][indices]
        new_states = self.data['next_obs'][indices]
        dones = self.data['done'][indices]

        self.updates_since_last_batch = 0

        return states, actions, rewards, dones, new_states, indices, weights

    def update_priorities(self, indexes, priorities):
        """
        Update priorities of specified indices

        indexes: the indices for which the priorities are updated
        priorities: the corresponding priority values
        """

        for idx, priority in zip(indexes, priorities):
            # Set current max priority
            self.max_priority = max(self.max_priority, priority)

            # Calculate $p_i^\alpha$
            priority_alpha = priority ** self.alpha
            # Update the trees
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def is_full(self) -> bool:
        """
        Returns Whether the buffer is full
        """
        return self.capacity == self.size

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
