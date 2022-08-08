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

from animation import RocketAnimation


def make(name):
    if name == 'RocketCircularization-v0':
        return RocketEnv(max_step=400, simulation_step=3, rmax=1.5, rmin=0.5, max_thrust=.1, oob_penalty=100, dt=0.03, velocity_penalty_rate=0.01, thrust_penalty_rate=0.01)
    else:
        raise ValueError(f'No environment {name}')


def uniform(r_min=0.99, r_max=1.01, rdot_min=-0.05, rdot_max=0.05, thetadot_min=0.99, thetadot_max=1.01):
    def func():
        nonlocal r_min, r_max, rdot_min, rdot_max, thetadot_min, thetadot_max

        r = np.random.uniform(r_min, r_max)
        theta = np.random.uniform(0, 2 * np.pi)
        rdot = np.random.uniform(rdot_min, rdot_max)
        thetadot = np.random.uniform(thetadot_min, thetadot_max)

        pos = [r, 0]
        vel = [rdot, r * thetadot]

        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

        return [*(rot_mat @ pos), *(rot_mat @ vel)]
    return func


def varied_l(r_min=0.5, r_max=1.5, rdot_min=-0.5, rdot_max=0.5, dl_min=-0.1, dl_max=0.1):
    def func():
        nonlocal r_min, r_max, rdot_min, rdot_max

        r = np.random.uniform(r_min, r_max)
        theta = np.random.uniform(0, 2 * np.pi)
        rdot = np.random.uniform(rdot_min, rdot_max)
        thetadot = (1 + np.random.uniform(dl_min, dl_max)) / r ** 2

        pos = [r, 0]
        vel = [rdot, r * thetadot]

        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

        return [*(rot_mat @ pos), *(rot_mat @ vel)]
    return func


def target_l(r_min=0.5, r_max=1.5, rdot_min=-0.5, rdot_max=0.5):
    def func():
        nonlocal r_min, r_max, rdot_min, rdot_max

        r = np.random.uniform(r_min, r_max)
        theta = np.random.uniform(0, 2 * np.pi)
        rdot = np.random.uniform(rdot_min, rdot_max)
        thetadot = 1 / r ** 2

        pos = [r, 0]
        vel = [rdot, r * thetadot]

        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

        return [*(rot_mat @ pos), *(rot_mat @ vel)]
    return func


def reward_function(state, action, rtarget, velocity_penalty_rate, thrust_penalty_rate, G=1, M=1):
    vtarget = np.sqrt(G * M / rtarget)
    r, v = state[:2], state[2:]
    dist = np.linalg.norm(r)
    rhat = r / dist
    rotation_matrix = np.array([[rhat[0], rhat[1]], [-rhat[1], rhat[0]]])
    vpolar = rotation_matrix @ v

    return -((dist - rtarget)**2) - 0.1 * vpolar[0] ** 2 - 0.1 * (vpolar[1] - vtarget)**2 - \
        thrust_penalty_rate * np.linalg.norm(action) ** 2


def score(state, rtarget, velocity_penalty_rate, G=1, M=1):
    vtarget = np.sqrt(G * M / rtarget)
    r, v = state[:2], state[2:]
    dist = np.linalg.norm(r)
    rhat = r / dist
    rotation_matrix = np.array([[rhat[0], -rhat[1]], [rhat[1], rhat[0]]])
    vtarget = rotation_matrix @ np.array([0, vtarget])

    return -np.abs(dist - rtarget) - velocity_penalty_rate * np.sum(np.abs(v - vtarget))


# def reward_function(state, action, prev_score, rtarget, velocity_penalty_rate, thrust_penalty_rate, G=1, M=1):
#     curr_score = score(state, rtarget, velocity_penalty_rate, G=G, M=M)

#     return curr_score - prev_score - thrust_penalty_rate * np.sum(np.abs(action)), curr_score


def clip_by_norm(t, mins, maxs):
    norm = np.linalg.norm(t)
    if np.count_nonzero(t) == 0:
        raise ValueError('Trying to clip norm of zero vector')
    if norm < mins:
        t = t * mins / norm
    elif norm > maxs:
        t = t * maxs / norm

    return t


def wall_clip_velocity(v, r, mins, maxs):
    direction = v @ r
    along = (v @ r) / (r @ r) * r
    ortho = v - along
    dist = np.linalg.norm(r)
    # If there is a component facing in at minimum radius
    if dist < mins and direction < 0:
        return ortho
    # If there is a component facing out at maximum radius
    elif dist > maxs and direction > 0:
        return ortho
    else:
        return v


class RocketEnv(gym.Env):
    def __init__(self,
                 G: float = 1, M: float = 1, m: float = .01, dt: float = .01,
                 rmin: float = .1, rmax: float = 2, rtarget: float = 1, vmax: float = 10, oob_penalty: float = 10,
                 max_thrust: float = .1, clip_thrust: str = 'Ball', velocity_penalty_rate: float = .001, thrust_penalty_rate: float = .0001,
                 max_step: int = 500, simulation_step: int = 10) -> None:
        super().__init__()

        self.observation_space = Box(low=np.array([-rmax, -rmax, -vmax, -vmax]),
                                     high=np.array([rmax, rmax, vmax, vmax]),
                                     shape=(4, ), dtype=np.float32)
        self.action_space = Box(low=np.array([-max_thrust, -max_thrust]),
                                high=np.array([max_thrust, max_thrust]),
                                shape=(2, ), dtype=np.float32)

        self.G, self.M, self.m, self.dt = G, M, m, dt
        self.rmin, self.rmax, self.rtarget = rmin, rmax, rtarget
        self.vmax = vmax
        self.oob_penalty = oob_penalty
        self.max_thrust = max_thrust
        self.clip_thrust = clip_thrust
        self.velocity_penalty_rate = velocity_penalty_rate
        self.thrust_penalty_rate = thrust_penalty_rate

        self.max_step, self.simulation_step = max_step, simulation_step
        self.iters = 0

        lim = rmax * 1.1
        self.animation = RocketAnimation(r_min=rmin, r_target=rtarget, r_max=rmax, xlim=(-lim, lim), ylim=(-lim, lim),
                                         markersize=10, circle_alpha=1, t_vec_len=.1)

    def reset(self, seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        # super().reset(seed=seed)

        if options is not None and 'init_func' in options:
            init_func = options['init_func']
        else:
            init_func = target_l()

        self.state = np.array(init_func())
        self.init_state = self.state
        self.iters = 0
        self.prev_score = - \
            np.abs(self.rmax - self.rtarget) - \
            self.velocity_penalty_rate * 2 * self.vmax
        self.done = False
        self.last_action = np.array([0, 0])

        lim = self.rmax * 1.1
        self.animation = RocketAnimation(r_min=self.rmin, r_target=self.rtarget, r_max=self.rmax,
                                         xlim=(-lim, lim), ylim=(-lim, lim),
                                         markersize=10, circle_alpha=1, t_vec_len=.1)

        if return_info:
            return self.state, dict()
        else:
            return self.state

    def step(self, action: np.ndarray) -> Union[Tuple[np.ndarray, float, bool, bool, dict],
                                                Tuple[np.ndarray, float, bool, dict]]:
        if self.done:
            print('Warning: Stepping after done is True')

        action = np.array(action)
        if self.clip_thrust == 'Box':
            action = np.clip(action, -1, 1)
        elif self.clip_thrust == 'Ball':
            magnitude = np.linalg.norm(action)
            if magnitude > 1:
                action = action / magnitude
        elif self.clip_thrust == 'None':
            pass
        else:
            raise ValueError(
                f'Thrust clipping mode {self.clip_thrust} does not exist')

        self.last_action = action

        r, v = self.state[:2], self.state[2:]
        reward = 0
        info = dict()

        for _ in range(self.simulation_step):
            # Calculate total force
            gravitational_force = - (self.G * self.M * self.m) / \
                (np.power(np.linalg.norm(r), 3)) * r  # F = - GMm/|r|^3 * r
            # Point the thrust radially
            thrust_force = action * self.m * self.max_thrust
            total_force = gravitational_force + thrust_force
            # Update position and location, this can somehow guarantee energy conservation
            v = v + total_force / self.m * self.dt
            v = clip_by_norm(v, 0, self.vmax)
            r = r + v * self.dt
            v = wall_clip_velocity(v, r, self.rmin, self.rmax)
            r = clip_by_norm(r, self.rmin, self.rmax)
            reward += reward_function(np.array([*r, *v]), action, self.rtarget, self.velocity_penalty_rate,
                                      self.thrust_penalty_rate, self.G, self.M)
            # step_reward, self.prev_score = reward_function(np.array([*r, *v]), action, self.prev_score, self.rtarget, self.velocity_penalty_rate,
            #                                                self.thrust_penalty_rate, self.G, self.M)
            # reward += step_reward * self.dt
            # If out-of-bounds, end the game
            # if np.linalg.norm(r) > self.rmax or np.linalg.norm(r) < self.rmin:
            #     print('Out-of-Bounds')
            #     reward -= self.oob_penalty
            #     self.done = True
            #     # self.state = self.init_state
            #     break
            # else:
            #     self.state = np.array([*r, *v])
        self.state = np.array([*r, *v])
        self.iters += 1

        if self.iters >= self.max_step:
            self.done = True

        return self.state, reward, self.done, info

    def render(self, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> None:
        self.animation.render(self.state, self.last_action, self.last_action,
                              self.rmin, self.rtarget, self.rmax)

    def show(self, path: Optional[str] = None, summary: bool = False) -> None:
        if path is None:
            if summary:
                self.animation.summary_plot()
                plt.show()
            else:
                self.animation.show_animation()
        else:
            self.animation.save_animation(path)


class DiscretiseAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.action_space = gym.spaces.Discrete(9)
        self.thrust_vectors = [
            [0, 0]] + [[np.cos(th), np.sin(th)] for th in np.linspace(0, 2 * np.pi, 8, endpoint=False)]

    def action(self, action):
        action = self.thrust_vectors[action]
        return np.array(action)


class PolarizeAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def action(self, action):
        state = self.unwrapped.state
        r, v = state[:2], state[2:]
        dist = np.linalg.norm(r)
        rhat = r / dist
        rotation_matrix = np.array([[rhat[0], -rhat[1]], [rhat[1], rhat[0]]])

        return rotation_matrix @ action


class TangentialThrust(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.action_space = gym.spaces.Discrete(3)

        # self.thrust_levels = [-1, -0.1, -.01, 0, 0.01, 0.1, 1]

    def action(self, action):
        return np.array([0, action - 1])
        # return np.array([0, self.thrust_levels[action]])


class RadialThrust(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.action_space = gym.spaces.Discrete(3)

        # self.thrust_levels = [-1, -0.3, -0.1, 0, 0.1, 0.3, 1]

    def action(self, action):
        # return np.array([self.thrust_levels[action], 0])
        return np.array([action - 1, 0])


class PolarizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        r, v = obs[:2], obs[2:]
        dist = np.linalg.norm(r)
        rhat = r / dist
        rotation_matrix = np.array([[rhat[0], rhat[1]], [-rhat[1], rhat[0]]])
        obs = np.array([dist, *(rotation_matrix @ v)])
        return obs


class RadialObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        r, v = obs[:2], obs[2:]
        dist = np.linalg.norm(r)
        rhat = r / dist
        obs = np.array([dist, v @ rhat])
        return obs
