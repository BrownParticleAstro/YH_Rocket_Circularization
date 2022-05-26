from email.policy import default
import numpy as np
from animation import *
from bounds import Bounds, DEFAULT_BOUNDS


class RocketCircularization(object):
    '''
    A Rocket Circularization Game Environment
    '''

    def __init__(self, max_iter=1000, evaluation_steps=2000,  radius_range=[0.1, 10], target_radius=1,
                 dt=0.01, M=1, m=0.01, G=1, bound_config=DEFAULT_BOUNDS,
                 init_state=[1, 0, 0, 1], thrust_vectors=[[.1, 0], [0, .1], [-.1, 0], [0, -.1]],
                 evaluation_penalty=1, inbounds_reward=1, thrust_penalty=.1, t_vec_len=1, polar=False):
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
        if isinstance(init_state, list):
            init_state = np.array(init_state, dtype=np.float32)
        if isinstance(thrust_vectors, list):
            thrust_vectors = np.array(thrust_vectors, dtype=np.float32)
        
        self.init_state = init_state
        self.state = init_state
        self.state_space_dim = init_state.shape[0]
        if not self.state_space_dim % 2 == 0:
            raise ValueError(
                'The number of states should be divisible by 2, representing both position and velocity')
        self.dims = self.state_space_dim // 2
        self.max_iter = max_iter
        self.evaluation_steps = evaluation_steps
        self.iters = 0
        self.simulation_steps = 0
        self.min_radius = radius_range[0]
        self.max_radius = radius_range[1]
        self.target_radius = target_radius
        self.bounds = Bounds(**bound_config)

        self.dt = dt
        self.M = M
        self.m = m
        self.G = G
        
        self.evaluation_penalty = evaluation_penalty
        self.inbounds_reward = inbounds_reward
        self.thrust_penalty = thrust_penalty

        # Initialize thrusters
        self.thrust_vectors = thrust_vectors
        self.num_thrusters = self.thrust_vectors.shape[0]
        if not self.thrust_vectors.shape == (self.num_thrusters, self.dims):
            raise ValueError(
                f'The thrust vector should have shape {(self.num_thrusters, self.dims)} (num_thrusters, dims)')
        self.action_space_size = 2 ** self.num_thrusters

        # Calculate Thrust for each action
        self.thrust_accelerations = []
        self.thrust_penalties = []
        for action in range(self.action_space_size):
            thrust_acceleration = np.zeros((self.dims,), dtype=np.float32)
            thrust_penalty = 0
            for thruster in range(self.num_thrusters):
                thrust_acceleration += (action % 2) * \
                    self.thrust_vectors[thruster]
                thrust_penalty += action % 2
                action //= 2
            self.thrust_accelerations.append(thrust_acceleration)
            self.thrust_penalties.append(thrust_penalty)
        self.thrust_accelerations = np.array(self.thrust_accelerations)
        self.thrust_penalties = np.array(self.thrust_penalties)

        self.done = False

        self.animation = RocketAnimation()
        self.t_vec_len = t_vec_len
        self.polar = polar

    def reset(self, init_state=None):
        '''
        Reset the environment

        init_state: The initial state after the reset

        return: The state after update
        '''
        if init_state is None:
            init_state = self.init_state
        if not init_state.shape[0] == self.dims * 2:
            raise ValueError(f'The number of states should be {self.dims}')
        self.state = init_state
        self.iters = 0
        self.simulation_steps = 0
        self.done = False
        
        # Reset the bounds
        self.bounds.reset()
        self.min_radius, self.max_radius = self.bounds.get_bounds(self.iters)
        
        # Initialize animation
        limits = (- self.max_radius - 0.2, self.max_radius + 0.2)
        plt.close(self.animation.fig)
        self.animation = RocketAnimation(
            r_min=self.min_radius, r_target=self.target_radius, r_max=self.max_radius,
            xlim=limits, ylim=limits, t_vec_len=self.t_vec_len)
        self.animation.render(init_state, np.array([0, 0]), self.min_radius, self.target_radius, self.max_radius)
        
        return self.state

    def _reward(self, pos):
        '''
        Return the reward at a given position

        pos: np.array (self.dim, )
        '''
        return -np.absolute(np.linalg.norm(pos) - self.target_radius)
    
    def _cartesian_to_polar(self, state):
        pos = state[:2]
        vel = state[2:]
        r = np.linalg.norm(pos)
        r_hat = pos / r
        theta_hat = np.array([-r_hat[1], r_hat[0]])
        theta = np.arctan2(pos[1], pos[0])
        r_dot = vel @ r_hat
        theta_dot = vel @ theta_hat
        return np.array([r, theta, r_dot, theta_dot])

    def step(self, action, time_steps=10):
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
        v_hat = v / np.linalg.norm(v)
        rotation_matrix = np.array([[v_hat[0], -v_hat[1]], [v_hat[1], v_hat[0]]])
        done = False
        reward = 0
        info = dict()
        
        # Read the new bounds
        self.min_radius, self.max_radius = self.bounds.get_bounds(self.iters)

        for _ in range(time_steps):
            # Calculate total force
            gravitational_force = - (self.G * self.M * self.m) / \
                np.power(np.linalg.norm(r), 3) * r  # F = - GMm/|r|^3 * r
            # Point the thrust in the direction of travel
            thrust_acc = rotation_matrix @ self.thrust_accelerations[action]
            thrust_force = thrust_acc * self.m
            total_force = gravitational_force + thrust_force
            # Update position and location, this can somehow guarantee energy conservation
            v = v + total_force / self.m * self.dt
            r = r + v * self.dt
            # reward for staying inbounds 
            # reward += (self.inbounds_reward - self.thrust_penalties[action] * self.thrust_penalty) * self.dt
            reward += self.dt
            self.simulation_steps += 1
            # If out-of-bounds, end the game
            if np.linalg.norm(r) > self.max_radius or np.linalg.norm(r) < self.min_radius:
                print('Out-of-Bounds')
                # reward -= 1e6 / self.simulation_steps + 1e3
                self.done = True
                break
        

        self.state = np.concatenate((r, v), axis=0)
        self.iters += 1
        if self.iters >= self.max_iter:
            self.done = True
            # Play for evaluation_steps after all has finished
            for _ in range(self.evaluation_steps):
                # Calculate total force
                gravitational_force = - \
                    (self.G * self.M * self.m) / \
                    np.power(np.linalg.norm(r), 3) * r  # F = - GMm/|r|^3 * r
                # Update position and location, this can somehow guarantee energy conservation
                v = v + gravitational_force / self.m * self.dt
                r = r + v * self.dt
                # reward for staying inbounds
                reward += (self._reward(r) * self.evaluation_penalty + self.inbounds_reward) * self.dt
                self.simulation_steps += 1
                # If out-of-bounds, end the game
                if np.linalg.norm(r) > self.max_radius or np.linalg.norm(r) < self.min_radius:
                    print('Out-of-Bounds')
                    # reward -= 1e6 / self.simulation_steps + 1e3
                    break
                
        self.animation.render(self.state, thrust_acc, self.min_radius, self.target_radius, self.max_radius)
        
        if self.polar:
            state = self._cartesian_to_polar(self.state)
        else:
            state = self.state

        return state, reward, self.done, info
    
    def animate(self, ):
        '''
        Show animation
        '''
        self.animation.show_animation()
    
    def save(self, name):
        '''
        Save animation
        
        Parameter:
            name: str, the file path
        '''
        self.animation.save_animation(name)

if __name__ == '__main__':
    env = RocketCircularization()
    done = False
    obs = env.reset(init_state=np.array([2, 0, 0, 0.75]))
    while not done:
        obs, _, done, _ = env.step(1)
        
    env.save('test.mp4')
    