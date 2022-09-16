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
import matplotlib.pyplot as plt

import gym
from gym.spaces import Box
from gym.utils import seeding


class RadialBalance(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.G = 1
        self.M = 1
        self.rmin, self.rmax = 0.5, 1.5
        self.rtarget = 1
        self.ltarget = np.sqrt(self.G * self.M / self.rtarget)
        self.simulation_steps = 3
        self.max_iters = 200
        self.dt = 0.05
        self.max_thrust = 0.1

        self.end_rmin, self.end_rmax = 0.98, 1.02
        self.end_vmin, self.end_vmax = -.03, .03
        self.end_steps = 20

        self.end_energy_tolerance = 1e-4
        self.end_energy = - 1 / self.rtarget + \
            self.ltarget ** 2 / (2 * self.rtarget ** 2) + \
            self.end_energy_tolerance
        print(f'End Energy: {self.end_energy:.3f}')

        self.action_space = Box(low=-1, high=1, shape=(1,))
        self.observation_space = Box(low=np.array(
            [self.rmin, -10]), high=np.array([self.rmax, 10]), shape=(2,))

    def reset(self) -> Any:

        r_init = np.random.uniform(0.5, 1.5)
        rdt_init = np.random.uniform(-.5, .5)
        self.state = np.array([r_init, rdt_init])
        self.done = False
        self.iters = 0

        self.record = []
        self.actions = []
        self.last_action = [0]

        self.end_counter = 0

        return self.state

    def _reward(self, state, action):
        return - (state[0] - self.rtarget) ** 2 - 0.1 * state[1] ** 2 # - 0.01 * action[0] ** 2

    def _total_energy(self, state):
        r, v = state
        pe = - 1 / r + self.ltarget ** 2 / (2 * r ** 2)
        ke = v**2 / 2

        return pe + ke

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        r, rdt = self.state
        thrust = action[0] * self.max_thrust

        for _ in range(self.simulation_steps):
            acc = - self.G * self.M / r ** 2 + self.ltarget ** 2 / r ** 3 + thrust
            rdt = rdt + acc * self.dt
            r = r + rdt * self.dt
            if r < self.rmin:
                r = self.rmin
                if rdt < 0:
                    rdt = 0
            elif r > self.rmax:
                r = self.rmax
                if rdt > 0:
                    rdt = 0

        self.state = np.array([r, rdt])
        self.last_action = action
        reward = self._reward(self.state, action)
        self.iters += 1
        if self.end_rmin < r < self.end_rmax and self.end_vmin < rdt < self.end_vmax:
        # if self._total_energy(self.state) < self.end_energy:
            self.end_counter += 1
            if self.end_counter >= self.end_steps:
                self.done = True
        else:
            self.end_counter = 0

        if self.iters > self.max_iters:
            self.done = True

        return self.state, reward, self.done, False, dict()

    def render(self):
        self.record.append(self.state)
        self.actions.append(self.last_action[0])

    def show(self, summary, _):
        record = np.array(self.record)
        time = np.arange(0, len(record), dtype=np.float64)
        time *= self.dt * self.simulation_steps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(time, record[:, 0], label='r')
        ax1.plot(time, record[:, 1], label='$\dot{r}$')
        ax1.axhline(y=self.end_rmax)
        ax1.axhline(y=self.end_rmin)
        ax1.axhline(y=self.end_vmax)
        ax1.axhline(y=self.end_vmin)

        ax2.plot(time, self.actions)
        ax1.grid(True)
        ax2.grid(True)
        plt.legend()
        plt.show()


class DiscretiseAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.action_space = gym.spaces.Discrete(3)

        # self.thrust_levels = [-1, -0.1, -.01, 0, 0.01, 0.1, 1]

    def action(self, action):
        return np.array([action - 1])
        # return np.array([self.thrust_levels[action]])
