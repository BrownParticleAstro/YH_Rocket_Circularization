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
    Dict,
    Callable
)

import numpy as np
import matplotlib.pyplot as plt

import gym
from gym.spaces import Box
from gym.utils import seeding

from animation import RocketAnimation


def make(name):
    '''
    Initialize Rocket Circularization environment. Contains the environment hyperparameters

    `RocketCircularization-v0` has wall mechanics on. When the craft hits the boundary, it will
    loose all orthogonal velocity towards the boundary.
    `RocketCircularization-v1` has wall mechanics off. When the craft hits the boundary, it will
    pass through boundary and mark the state as truncated.

    Usage:
    ```python
    env = make('RocketCircularization-v0')
    ```
    or in notebooks
    ```python
    with make('RocketCircularization-v0') as env:
        ...
    ```
    '''
    if name == 'RocketCircularization-v0':
        init_func = varied_l(r_min=0.5, r_max=1.5)
        return RocketEnv(max_step=400, simulation_step=3, rmax=1.5, rmin=0.5,
                         init_func=init_func, max_thrust=.01,
                         oob_penalty=0, dt=0.03, wall_mechanics=True,
                         velocity_penalty_rate=0.1, thrust_penalty_rate=0.001)
    if name == 'RocketCircularization-v1':
        return RocketEnv(max_step=400, simulation_step=3, rmax=5, rmin=0.5, max_thrust=.1,
                         oob_penalty=0, dt=0.03, wall_mechanics=False,
                         velocity_penalty_rate=0.1, thrust_penalty_rate=0.001)
    else:
        raise ValueError(f'No environment {name}')


def uniform(r_min: float = 0.99, r_max: float = 1.01,
            rdot_min: float = -0.05, rdot_max: float = 0.05,
            thetadot_min: float = 0.99, thetadot_max: float = 1.01) \
        -> Callable[[], List[np.float32]]:
    '''
    Produces a function that generates initial conditions at different angles uniformly under those
    conditions

    r_min: minimum bound for radius
    r_max: maximum bound for radius
    rdot_min: minimum bound for radial velocity
    rdot_max: maximum bound for radial velocity
    thetadot_min: minimum bound for angular velocity
    thetadot_max: maximum bound for angular velocity

    Return:
        A function that when called, returns an initial condition
    '''
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


def varied_l(r_min: float = 0.9, r_max: float = 1.1,
             rdot_min: float = -0.5, rdot_max: float = 0.5,
             dl_min: float = -.5, dl_max: float = .5) \
        -> Callable[[], List[np.float32]]:
    '''
    Produces a function that generates initial conditions at different angles uniformly with
    a range of angular momentum settings

    r_min: minimum bound for radius
    r_max: maximum bound for radius
    rdot_min: minimum bound for radial velocity
    rdot_max: maximum bound for radial velocity
    dl_min: minimum deviation of angular momentum from target angular momentum
    dl_max: maximum deviation of angular momentum from target angular momentum

    Return:
        A function that when called, returns an initial condition
    '''
    def func():
        nonlocal r_min, r_max, rdot_min, rdot_max

        # r = np.random.uniform(r_min, r_max)
        r = np.random.uniform(1, 1)
        theta = np.random.uniform(0, 2 * np.pi)
        # rdot = np.random.uniform(rdot_min, rdot_max)
        rdot = np.random.uniform(1, 1)
        thetadot = (1 + np.random.uniform(1, 1)) / r ** 2
        # thetadot = (1 + np.random.uniform(dl_min, dl_max)) / r ** 2

        # pos = [r, 0]
        pos = [r, 0]
        vel = [0, 1]# setting velocity to 1

        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

        return [*(rot_mat @ pos), *(rot_mat @ vel)]
    return func


def target_l(r_min: float = 0.5, r_max: float = 1.5,
             rdot_min: float = -0.5, rdot_max: float = 0.5) \
        -> Callable[[], List[np.float32]]:
    '''
    Produces a function that generates initial conditions at different angles uniformly 
    with the target angular momentum

    r_min: minimum bound for radius
    r_max: maximum bound for radius
    rdot_min: minimum bound for radial velocity
    rdot_max: maximum bound for radial velocity

    Return:
        A function that when called, returns an initial condition
    '''
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


def quadratic_penalty(state: np.ndarray, action: np.ndarray, rtarget: float,
                      velocity_penalty_rate: float, thrust_penalty_rate: float,
                      G: float = 1, M: float = 1) -> np.float32:
    '''
    Calculates the Quadratic reward at the current state with the current action. Subject to change.

    reward = -(r - rtarget)^2 - velocity_penalty * (v_r^2 + (v_t - v_t,target)^2) - thrust_penalty * |u|^2

    state: the current game state
    action: actions performed to reach this state
    rtarget: target radius of the craft
    velocity_penalty_rate: ratio of velocity penalty to radius penalty
    thrust_penalty_rate: ratio of thrust penalty to radius penalty

    G: Gravitational Constant, default 1
    M: Mass of center object, default 1

    Return:
        Reward in this state
    '''
    vtarget = np.sqrt(G * M / rtarget)
    r, v = state[:2], state[2:]
    dist = np.linalg.norm(r)
    rhat = r / dist
    rotation_matrix = np.array([[rhat[0], rhat[1]], [-rhat[1], rhat[0]]])
    vpolar = rotation_matrix @ v

    return -((dist - rtarget)**2) \
        - velocity_penalty_rate * (vpolar[0] ** 2 + (vpolar[1] - vtarget)**2) \
        - thrust_penalty_rate * np.linalg.norm(action) ** 2

def basic_reward(state: np.ndarray, action: np.ndarray, rtarget: float,
                      velocity_penalty_rate: float, thrust_penalty_rate: float,
                      G: float = 1, M: float = 1) -> np.float32:
    vtarget = np.sqrt(G * M / rtarget)
    r, v = state[:2], state[2:]
    dist = np.linalg.norm(r)
    if abs(dist - rtarget)<0.1:
        return 1
    elif abs(dist -rtarget)<0.5:
        return 0.5
    else:
        return 0
    
def reward_function(state: np.ndarray, action: np.ndarray, rtarget: float,
                    velocity_penalty_rate: float, thrust_penalty_rate: float,
                    mode: str = 'Quadratic', G: float = 1, M: float = 1) -> np.float32:
    '''
    Calculates the reward at the current state with the current action. Subject to change.

    state: the current game state
    action: actions performed to reach this state
    rtarget: target radius of the craft
    velocity_penalty_rate: ratio of velocity penalty to radius penalty
    thrust_penalty_rate: ratio of thrust penalty to radius penalty
    mode: Mode of rewards, one of 'Quadratic' or 'Gaussian'
            Quadratic: -(r - rtarget)^2 - velocity_penalty * (v_r^2 + (v_t - v_t,target)^2) - thrust_penalty * |u|^2
            Gaussian: exp(-(r - rtarget)^2 - velocity_penalty * (v_r^2 + (v_t - v_t,target)^2) - thrust_penalty * |u|^2)

    G: Gravitational Constant, default 1
    M: Mass of center object, default 1

    Return:
        Reward in this state
    '''
    value = quadratic_penalty(state, action, rtarget,
                              velocity_penalty_rate, thrust_penalty_rate, G, M)
    return value
    if mode == 'Quadratic':
        return value
    elif mode == 'Gaussian':
        return np.exp(value)
    else:
        ValueError(f'Invalid reward mode {mode}')


def score(state: np.ndarray, rtarget: float,  velocity_penalty_rate: float,
          G: float = 1, M: float = 1) -> np.float32:
    '''
    DEPRECATED
    Calculates the reward at the current state without action penalty. Subject to change.
    This may be used for a differential reward structure

    score = -(r - rtarget)^2 - velocity_penalty * (v_r^2 + (v_t - v_t,target)^2)

    state: the current game state
    rtarget: target radius of the craft
    velocity_penalty_rate: ratio of velocity penalty to radius penalty

    G: Gravitational Constant, default 1
    M: Mass of center object, default 1

    Return:
        Score in this state
    '''
    vtarget = np.sqrt(G * M / rtarget)
    r, v = state[:2], state[2:]
    dist = np.linalg.norm(r)
    rhat = r / dist
    rotation_matrix = np.array([[rhat[0], -rhat[1]], [rhat[1], rhat[0]]])
    vtarget = rotation_matrix @ np.array([0, vtarget])

    return -np.abs(dist - rtarget) - velocity_penalty_rate * np.sum(np.abs(v - vtarget))


def reward_function(state: np.ndarray, action: np.ndarray, prev_score: np.float32, rtarget: float,
                    velocity_penalty_rate: float, thrust_penalty_rate: float,
                    G: float=1, M: float=1) -> np.float32:
    '''
    DEPRECATED
    Calculates the reward at the current state with action penalty. Subject to change.
    This may be used for a differential reward structure

    reward = current_score - prev_score + thrust_penalty * |u|^2

    state: the current game state
    action: actions performed to reach this state
    prev_score: score from the last time the reward is calculated
    rtarget: target radius of the craft
    velocity_penalty_rate: ratio of velocity penalty to radius penalty
    thrust_penalty_rate: ratio of thrust penalty to radius penalty

    G: Gravitational Constant, default 1
    M: Mass of center object, default 1

    Return:
        Reward at this state
    '''
    curr_score = score(state, rtarget, velocity_penalty_rate, G=G, M=M)
    return curr_score - prev_score - thrust_penalty_rate * np.sum(np.abs(action)), curr_score


def clip_by_norm(t: np.ndarray, mins: float, maxs: float) -> np.ndarray:
    '''
    Clip the vector by its l2 norm between an interval.

    t: the vector to clip
    mins: the minimum norm
    maxs: the maximum norm

    Return:
        Clipped vector

    Raises:
        ValueError: when norm of input vector is zero and minimum is not zero
    '''
    norm = np.linalg.norm(t)
    if np.count_nonzero(t) == 0 and mins > 0:
        raise ValueError('Trying to clip norm of zero vector')
    if norm < mins:
        t = t * mins / norm
    elif norm > maxs:
        t = t * maxs / norm

    return t


def wall_clip_velocity(v: np.ndarray, r: np.ndarray, mins: float, maxs: float):
    '''
    If the particle is moving towards the circular boundaries, cancel velocity perpendicular to the boundary

    v: velocity vector
    r: position vector
    mins: minimum radius
    maxs: maximum radius

    Return: 
        Velocity vector modified by the walls
    '''
    # Obtain the velocity component orthogonal to the circular boundaries
    direction = v @ r
    along = (v @ r) / (r @ r) * r
    ortho = v - along

    # Get the distance from origin to test if the object is at the bounds
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
    '''
    Open AI Gym environment for Rocket Circularization
    '''

    def __init__(self,
                 G: float = 1, M: float = 1, m: float = .01, dt: float = .01,
                 rmin: float = .1, rmax: float = 2, rtarget: float = 1, vmax: float = 10,
                 init_func: Callable[[], np.ndarray] = varied_l(), wall_mechanics: bool = True,
                 oob_penalty: float = 10, max_thrust: float = .01, clip_thrust: str = 'Ball',
                 velocity_penalty_rate: float = .001, thrust_penalty_rate: float = .0001,
                 max_step: int = 500, simulation_step: int = 1) -> None:
        '''
        Initializes the environment

        G: Gravitational Constant, default 1
        M: Mass of center object, default 1
        m: Mass of orbiting object, default .01
        dt: Simulation time step, default .01

        rmin: game space radius lower bound, default .1
        rmax: game space radius upper bound, default 2
        rtarget: the target radius the craft is supposed to reach, default 1
        vmax: maximum velocity allowed in the game space (implemented for simulation accuracy and network interpolation), default 10
        oob_penalty: penalty for being out of bounds, default 10
        init_func: function that returns an initial condition, default varied_l()
        wall_mechanics: whether the boundary acts as a wall.
                If true, all normal velocity towards the boundary will be canceled upon impact
                If false, the craft will pass through the wall and truncation will be marked true 

        max_thrust: The magnitude of the thrust, scales the action u
        clip_thrust: The way in which the action is clipped, Options: Box, Ball, None, default: Ball

        velocity_penalty_rate: the penalty of velocity as a ratio of radius penalty
        thrust_penalty_rate: the penalty of thrust as a ratio of radius penalty

        max_step: number of iterations in each episode if no early-termination is encountered, default: 500
        simulation_step: number of simulation steps for every game step. Reducing timestep and increasing simulation step
                increases simulation accuracy, but may be more computationally straining
        '''
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
        self.init_func = init_func
        self.wall_mechanics = wall_mechanics
        self.velocity_penalty_rate = velocity_penalty_rate
        self.thrust_penalty_rate = thrust_penalty_rate

        self.max_step, self.simulation_step = max_step, simulation_step
        self.iters = 0

        # Animation object
        lim = rmax * 1.1
        self.animation = RocketAnimation(r_min=rmin, r_target=rtarget, r_max=rmax, xlim=(-lim, lim), ylim=(-lim, lim),
                                         markersize=10, circle_alpha=1, t_vec_len=.1)

    def reset(self, seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        '''
        Resets the environment with a new state, return the new state as well as information
        if required.

        seed: NOT IMPLEMENTED, randomizer seed
        return_info: If information is returned
        options: some options for initialization
                init_func: the function used to initialize the state, provided uniform, target_l, varied_l

        Return:
            the initial state, shape (4,)
        '''
        # super().reset(seed=seed)

        if options is not None and 'init_func' in options:
            init_func = options['init_func']
        else:
            init_func = self.init_func

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
        '''
        Accept action and modify the states accordingly. 

        action: an array with shape (2,) representing the thrust component in the 2 cartesian directions.

        Return:
            state, shape (2,),
            reward from this state,
            if the game is done running, 
            if the agent is out-of-bounds, and
            some more info about the game
        '''
        if self.done:
            print('Warning: Stepping after done is True')

        # Clipp action if needed
        action = np.array(action)
        print(action)
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

        # For easy access from wrappers
        self.last_action = action

        r, v = self.state[:2], self.state[2:]
        reward = 0
        info = dict()

        # Simulate for a number of steps
        for _ in range(self.simulation_step):
            # Calculate total force
            gravitational_force = - (self.G * self.M * self.m) / \
                (np.power(np.linalg.norm(r), 3)) * r  # F = - GMm/|r|^3 * r
            thrust_force = action * self.m * self.max_thrust
            total_force = gravitational_force + thrust_force
            # Update position and location, this can somehow guarantee energy conservation
            # If the craft hits a wall, all normal velocity cancels
            v = v + total_force / self.m * self.dt
            # v = clip_by_norm(v, 0, self.vmax)
            r = r + v * self.dt
            if self.wall_mechanics:
                v = wall_clip_velocity(v, r, self.rmin, self.rmax)
                r = clip_by_norm(r, self.rmin, self.rmax)
            # Scored-based reward structure
            # reward += reward_function(np.array([*r, *v]), action,
            #                           self.rtarget, self.velocity_penalty_rate,
            #                           self.thrust_penalty_rate, 'Quadratic', self.G, self.M)
            # Differential-score-based Reward structure
            step_reward, self.prev_score = reward_function(np.array([*r, *v]), action, self.prev_score, self.rtarget, self.velocity_penalty_rate,
                                                           self.thrust_penalty_rate, self.G, self.M)
            reward += step_reward * self.dt

            # If out-of-bounds, end the game
            if self.wall_mechanics:
                # The game will not be truncated when wall-mechanics are disabled
                truncated = False
            else:
                # This condition may be changed into a controllability condition
                if np.linalg.norm(r) > self.rmax or np.linalg.norm(r) < self.rmin:
                    reward -= self.oob_penalty
                    truncated = True
                    # self.state = self.init_state
                else:
                    self.state = np.array([*r, *v])
                    truncated = False

        self.state = np.array([*r, *v])
        self.iters += 1

        if self.iters >= self.max_step:
            self.done = True

        return self.state, reward, self.done, truncated, info

    def render(self, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> None:
        '''
        Record frames of the animation. Need to be used in conjunction with the
        show() method.
        '''
        self.animation.render(self.state, self.last_action, self.last_action,
                              self.rmin, self.rtarget, self.rmax)

    def show(self, path: Optional[str] = None, summary: bool = False) -> None:
        '''
        Show the saved frames of the animation or produce a summary

        path: if the animation is saved, the path to which it is saved. If None, the 
                animation is shown in a pop-up window. Note that pop-up window 
                animation does not work in notebooks, and need to be closed for
                the execution to continue. default, None
        summary: if animation is not saved, if to produce a summary plot of the game 
                episode instead.
        '''
        if path is None:
            if summary:
                self.animation.summary_plot()
                plt.show()
            else:
                self.animation.show_animation()
        else:
            self.animation.save_animation(path)


class PolarizeAction(gym.ActionWrapper):
    '''
    Wrapper for RocketEnv. Convert polar thrust request to cartesian. Note that the
    cartesian thrust value will not rotate with the state during the simulation step.
    '''

    def __init__(self, env: gym.Env) -> None:
        '''
        Initialize the wrapper.

        env: The environment to wrap
        '''
        super().__init__(env)

    def action(self, action: np.ndarray) -> np.ndarray:
        '''
        Convert polar thrust requests to cartesian given the current position.

        action: 1D numpy array of shape (2,). The components corresponds to the radial
                and tangential components of the thrust respectively.

        Return:
            numpy array of shape (2,). The thrust in cartesian coordinates.
        '''
        state = self.unwrapped.state
        r, v = state[:2], state[2:]
        dist = np.linalg.norm(r)
        rhat = r / dist
        rotation_matrix = np.array([[rhat[0], -rhat[1]], [rhat[1], rhat[0]]])

        return rotation_matrix @ action


class DiscretiseAction(gym.ActionWrapper):
    '''
    Wrapper for RocketEnv. Provides 9 discrete thrust levels, no thrust, 4 cardinal directions, and 4 diagonal
    directions with unit length. Need to be combined with PolarizeAction to become polar.
    '''

    def __init__(self, env: gym.Env) -> None:
        '''
        Initialize the wrapper.

        env: The environment to wrap, preferably with polar thrust
        '''
        super().__init__(env)

        self.action_space = gym.spaces.Discrete(9)
        self.thrust_vectors = [
            [0, 0]] + [[np.cos(th), np.sin(th)] for th in np.linspace(0, 2 * np.pi, 8, endpoint=False)]

    def action(self, action: int) -> np.ndarray:
        '''
        Map integers to their correspnding thrust value.
        0 represents no thrust, 1-8 represents the other directions in counter-clockwise
        direction from the x-axis. 

        action: integer value thrust choice

        Return:
            1d numpy array of shape (2,), representing the thrust value for this iteration.
        '''
        action = self.thrust_vectors[action]
        return np.array(action)


class TangentialThrust(gym.ActionWrapper):
    '''
    Wrapper for RocketEnv. Provides 3 discrete thrust vectors in the tangential direction
    with unit length. Note that it needs to be used in conjuction with PolarizeAction for
    the thrust values to point in the tangential direction.
    '''

    def __init__(self, env: gym.Env):
        '''
        Initialize the wrapper.

        env: The environment to wrap, preferably with polar thrust
        '''
        super().__init__(env)

        self.action_space = gym.spaces.Discrete(3)

        # For more detailed control
        # self.thrust_levels = [-1, -0.1, -.01, 0, 0.01, 0.1, 1]

    def action(self, action):
        '''
        Map integers to their correspnding thrust value.
        0: clockwise thrust
        1: no thrust
        2: counter-clockwise thrust

        action: integer value thrust choice

        Return:
            1d numpy array of shape (2,), representing the thrust value for this iteration.
        '''
        return np.array([0, action - 1])

        # For more detailed control
        # return np.array([0, self.thrust_levels[action]])


class RadialThrust(gym.ActionWrapper):
    '''
    Wrapper for RocketEnv. Provides 3 discrete thrust vectors in the radial direction 
    with unit length. Need to be used in conjunction with PolarizeAction for the
    thrust values to point in the radial direction.
    '''

    def __init__(self, env: gym.Env):
        '''
        Initialize the wrapper.

        env: The environment to wrap, preferably with polar thrust
        '''
        super().__init__(env)

        self.action_space = gym.spaces.Discrete(3)

        # For more detailed control
        # self.thrust_levels = [-1, -0.3, -0.1, 0, 0.1, 0.3, 1]

    def action(self, action: int) -> np.ndarray:
        '''
        Map integers to their correspnding thrust value.
        0: thrust inwards
        1: no thrust
        2: thrust outwards

        action: integer value thrust choice

        Return:
            1d numpy array of shape (2,), representing the thrust value for this iteration.
        '''
        # For more detailed control
        # return np.array([self.thrust_levels[action], 0])

        return np.array([action - 1, 0])


class PolarizeObservation(gym.ObservationWrapper):
    '''
    Wrapper for RocketEnv. Wraps state to provide radius, radial velocity, and tangential
    velocity.
    '''

    def __init__(self, env: gym.Env) -> None:
        '''
        Initialize the wrapper.

        env: The environment to wrap
        '''
        super().__init__(env)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        '''
        Given the observation in cartesian coordinates, convert to polar observation.

        obs: numpy array with shape (4,). State vector with first 2 as cartesian position
                and last 2 as cartesian velocity

        Return
            numpy array with shape (3,) with radius, radial velocity, and tangential velocity,
            in that order.
        '''
        r, v = obs[:2], obs[2:]
        dist = np.linalg.norm(r)
        rhat = r / dist
        rotation_matrix = np.array([[rhat[0], rhat[1]], [-rhat[1], rhat[0]]])
        obs = np.array([dist, *(rotation_matrix @ v)])
        return obs


class RadialObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        '''
        Initialize the wrapper.

        env: The environment to wrap
        '''
        super().__init__(env)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        '''
        Given the observation in cartesian coordinates, convert to observation with
        only radial position and velocity.

        obs: numpy array with shape (4,). State vector with first 2 as cartesian position
                and last 2 as cartesian velocity

        Return
            numpy array with shape (2,) with radius and radial velocity, in that order.
        '''
        r, v = obs[:2], obs[2:]
        dist = np.linalg.norm(r)
        rhat = r / dist
        obs = np.array([dist, v @ rhat])
        return obs
