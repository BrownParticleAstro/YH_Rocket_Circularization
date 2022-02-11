import numpy as np
from animation import *


class RocketCircularization(object):
    '''
    A Rocket Circularization Game Environment
    '''

    def __init__(self, max_iter=500, radius_range=[0.1, 10], target_radius=1,
                 dt=0.01, M=1, m=0.01, G=1,
                 init_state=np.array([1, 0, 0, 1], dtype=np.float32),
                 thrust_vectors=np.array([[-.1, 0], [0, -.1], [.1, 0], [0, .1]], dtype=np.float32)):
        '''
        Initialize the Rocket Circularization game environment

        max_iter: int, indicating after how many successful iterations the game will end
        radius_range: list or tuple (2, ), indicating the maximum and minimum radius out of which the game is considered failed
        dt: float, the time step of the simulation
        M: float, the mass of the center mass
        m: float, the mass of the rocket (Useless in this context)
        G: float, the Gravitational constant 
        init_state: np.array (np.float32) shape: (state_space_dim,), indicating the position and velocity components of the rocket
        thrust_vectors: np.array (np.float32) shape: (num_thrusters, dims), indicating the thrust acceleration of each thrust on the rocket
        '''

        # initialize the board
        self.state = init_state
        self.state_space_dim = init_state.shape[0]
        if not self.state_space_dim % 2 == 0:
            raise ValueError(
                'The number of states should be divisible by 2, representing both position and velocity')
        self.dims = self.state_space_dim // 2
        self.max_iter = max_iter
        self.iters = 0
        self.min_radius = radius_range[0]
        self.max_radius = radius_range[1]
        self.target_radius = target_radius

        self.dt = dt
        self.M = M
        self.m = m
        self.G = G

        # Initialize thrusters
        self.thrust_vectors = thrust_vectors
        self.num_thrusters = self.thrust_vectors.shape[0]
        if not self.thrust_vectors.shape == (self.num_thrusters, self.dims):
            raise ValueError(
                f'The thrust vector should have shape {(self.num_thrusters, self.dims)} (num_thrusters, dims)')
        self.action_space_size = 2 ** self.num_thrusters

        # Calculate Thrust for each action
        self.thrust_accelerations = []
        for action in range(self.action_space_size):
            thrust_acceleration = np.zeros((self.dims,), dtype=np.float32)
            for thruster in range(self.num_thrusters):
                thrust_acceleration += (action % 2) * \
                    self.thrust_vectors[thruster]
                action //= 2
            self.thrust_accelerations.append(thrust_acceleration)
        self.thrust_accelerations = np.array(self.thrust_accelerations)

        self.done = False

        self.animation = RocketAnimation()

    def reset(self, init_state=np.array([1, 0, 0, 1.01], dtype=np.float32)):
        '''
        Reset the environment

        init_state: The initial state after the reset

        return: The state after update
        '''
        if not init_state.shape[0] == self.dims * 2:
            raise ValueError(f'The number of states should be {self.dims}')
        self.state = init_state
        self.iters = 0
        self.done = False
        
        # Initialize animation
        limits = (- self.max_radius - 0.2, self.max_radius + 0.2)
        self.animation = RocketAnimation(
            r_min=self.min_radius, r_max=self.max_radius,
            xlim=limits, ylim=limits)
        self.animation.render(init_state)
        return self.state

    def _reward(self, pos):
        '''
        Return the reward at a given position

        pos: np.array (self.dim, )
        '''
        return -np.absolute(np.linalg.norm(pos) - self.target_radius)

    def step(self, action, time_steps=10, evaluation_steps=2000):
        '''
        Move to the next step of the simulation with the forces provided. Calculates the new state and the rewards

        action: int, a number in [0, action_space_size), representing which thrusters are on
        time_steps: int, the number of dt's the simulation will iterate through

        Return: a tuple of 4 elements
          observation: np.array, shape: (state_space_dim,) indicating the new state of the rocket
          reward: float, indicating the reward corresponding to this action
          done: bool, if the simulation is finished
          info: dict, information about the environment (NOT IMPLEMENTED)
        '''
        if not (action >= 0 and action < self.action_space_size):
            raise ValueError(
                f'Action should be an integer between 0 and {self.action_space_size}')

        if self.done:
            print('Warning: Stepping after done is True')

        r, v = self.state[:self.state_space_dim //
                          2], self.state[self.state_space_dim//2:]
        done = False
        reward = 0
        info = dict()

        for _ in range(time_steps):
            # Calculate total force
            gravitational_force = - (self.G * self.M * self.m) / \
                np.power(np.linalg.norm(r), 3) * r  # F = - GMm/|r|^3 * r
            thrust_force = self.thrust_accelerations[action] * self.m
            total_force = gravitational_force + thrust_force
            # Update position and location, this can somehow guarantee energy conservation
            v = v + total_force / self.m * self.dt
            r = r + v * self.dt
            # reward += self._reward(r) * self.dt
            # If out-of-bounds, end the game
            if np.linalg.norm(r) > self.max_radius or np.linalg.norm(r) < self.min_radius:
                print('Out-of-Bounds')
                self.done = True
                break

        self.state = np.concatenate((r, v), axis=0)
        self.iters += 1
        if self.iters >= self.max_iter:
            self.done = True
            # Play for evaluation_steps after all has finished
            for _ in range(evaluation_steps):
                # Calculate total force
                gravitational_force = - \
                    (self.G * self.M * self.m) / \
                    np.power(np.linalg.norm(r), 3) * r  # F = - GMm/|r|^3 * r
                # Update position and location, this can somehow guarantee energy conservation
                v = v + gravitational_force / self.m * self.dt
                r = r + v * self.dt
                reward += self._reward(r) * self.dt
                # If out-of-bounds, end the game
                if np.linalg.norm(r) > self.max_radius or np.linalg.norm(r) < self.min_radius:
                    print('Out-of-Bounds')
                    reward -= 100
                    break
                
        self.animation.render(self.state)

        return self.state, reward, self.done, info
    
    def animate(self, ):
        '''
        Show animation
        '''
        self.animation.show_animation()

if __name__ == '__main__':
    env = RocketCircularization()
    done = False
    obs = env.reset(init_state=np.array([2, 0, 0, 0.75]))
    while not done:
        obs, _, done, _ = env.step(1)
        
    env.animate()
    