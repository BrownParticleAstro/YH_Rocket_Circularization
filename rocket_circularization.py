import numpy as np
from animation import RocketAnimation
from bounds import Bounds, DEFAULT_BOUNDS
from initial_condition import InitialCondition, DEFAULT_INITIAL_CONDITION


class RocketCircularization(object):
    '''
    A Rocket Circularization Game Environment
    '''

    def __init__(self, max_iter=1000, evaluation_steps=2000, iter_steps=10, radius_range=[0.1, 10], target_radius=1,
                 dt=0.01, M=1, m=0.01, G=1, bound_config=None, ignore_bounds=False,
                 init_state=[1, 0, 0, 1], thrust_vectors=[[.1, 0], [0, .1], [-.1, 0], [0, -.1]], max_thrust=.1,
                 evaluation_penalty=1, inbounds_reward=1, thrust_penalty=.1, circularization_penalty=1, ang_momentum_penalty=0,
                 penalty_mode='Absolute',
                 t_vec_len=1, circle_alpha=1, state_output_mode='Cartesian', state_target_r=False, state_target_l=False,
                 thrust_mode='On-off', thrust_direction='Polar', clip=True):
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
        if isinstance(thrust_vectors, list):
            thrust_vectors = np.array(thrust_vectors, dtype=np.float32)
        
        if isinstance(init_state, list) or isinstance(init_state, np.ndarray):
            self.init_state = InitialCondition('constant', {'value': init_state})
        elif isinstance(init_state, dict):
            self.init_state = InitialCondition(**init_state)
            
        self.state = self.init_state.get_initial_condition()
        
        self.state_space_dim = self.state.shape[0]
        if not self.state_space_dim % 2 == 0:
            raise ValueError(
                'The number of states should be divisible by 2, representing both position and velocity')
        self.dims = self.state_space_dim // 2
        self.max_iter = max_iter
        self.evaluation_steps = evaluation_steps
        self.iters = 0
        self.simulation_steps = 0
        self.iter_steps = iter_steps
        
        self._init_bounds(bound_config, radius_range)
        self.ignore_bounds = ignore_bounds
        self.target_radius = target_radius

        self.dt = dt
        self.M = M
        self.m = m
        self.G = G
        
        self.circularization_penalty = circularization_penalty
        self.ang_momentum_penalty = ang_momentum_penalty
        self.evaluation_penalty = evaluation_penalty
        self.inbounds_reward = inbounds_reward
        self.thrust_penalty = thrust_penalty
        
        penalty_functions = {
            'Absolute': np.absolute, 
            'Quadratic': np.square
        }
        self.penalty_function = penalty_functions[penalty_mode]

        # Initialize thrusters
        self.thrust_vectors = thrust_vectors
        self.num_thrusters = self.thrust_vectors.shape[0]
        if not self.thrust_vectors.shape == (self.num_thrusters, self.dims):
            raise ValueError(
                f'The thrust vector should have shape {(self.num_thrusters, self.dims)} (num_thrusters, dims)')

        self.thrust_mode = thrust_mode
        self.thrust_direction = thrust_direction
        self.clip = clip
        # Calculate Thrust for each action
        if self.thrust_mode == 'On-off':
            self.action_space_size = 2 ** self.num_thrusters
            self._thrust_acc_and_penalties()
        elif self.thrust_mode == 'Continuous':
            self.max_thrust = max_thrust
            self.action_space_size = 2
        else:
            print(f'Warning: Unknown thrust mode {self.thrust_mode}')

        self.done = False

        self.animation = None
        self.t_vec_len = t_vec_len
        self.circle_alpha = circle_alpha
        
        output_dims = {
            'Cartesian': self.state_space_dim,
            'Polar': self.state_space_dim,
            'No Theta': self.state_space_dim - 1
        }
        
        self.state_output_dims = output_dims[state_output_mode] + sum([state_target_r, state_target_l])
        self.state_target_r = state_target_r
        self.state_target_l = state_target_l
        self.polar = state_output_mode in {'Polar', 'No Theta'}
        self.state_output_mode = state_output_mode
        
        
    def get_state_dims(self):
        return self.state_output_dims
    
    def get_action_dims(self):
        return self.action_space_size

    def get_thrust_mode(self):
        thrust_modes = {
            'On-off': 'Discrete',
            'Continuous': 'Continuous'
        }
        return thrust_modes[self.thrust_mode]
    
    def _get_thrust_and_penalty(self, action):
        if self.thrust_mode == 'On-off':
            thrust_acc = self.thrust_accelerations[action]
            thrust_penalty = self.thrust_penalties[action]
        elif self.thrust_mode == 'Continuous':
            thrust_acc = action
            if self.clip:
                thrust_acc_magnitude = np.linalg.norm(thrust_acc)
                if thrust_acc_magnitude > 1:
                    thrust_acc = thrust_acc / thrust_acc_magnitude     
            thrust_acc = thrust_acc * self.max_thrust
            thrust_penalty = np.linalg.norm(thrust_acc)
        
        if self.thrust_direction == 'Polar':
            r = self.state[:self.state_space_dim // 2]
            r_hat = r / np.linalg.norm(r)
            rotation_matrix = np.array([[r_hat[0], -r_hat[1]], [r_hat[1], r_hat[0]]])
            thrust_acc = rotation_matrix @ thrust_acc
            
        return thrust_acc * self.m, thrust_penalty
        
    def _thrust_acc_and_penalties(self):
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
        
    def _init_bounds(self, bound_config, radius_range):
        if bound_config is None:
            bound_config = {
                'rmin_func': 'constant',
                'rmin_strategy': [
                    {
                        'name': 'constant',
                        'parameters': {'const': radius_range[0]}
                    }
                ],
                'rmax_func': 'constant',
                'rmax_strategy': [
                    {
                        'name': 'constant',
                        'parameters': {'const': radius_range[1]}
                    }
                ]
            }
            
        self.bounds = Bounds(**bound_config)

        self.min_radius, self.max_radius = self.bounds.get_bounds(0)

    def reset(self, init_state=None):
        '''
        Reset the environment

        init_state: The initial state after the reset

        return: The state after update
        '''
        if init_state is None:
            init_state = self.init_state.get_initial_condition()
        if not init_state.shape[0] == self.dims * 2:
            raise ValueError(f'The number of states should be {self.dims}')
        
        # Reset the initial condition
        self.state = init_state
        self.init_state.reset()
        
        self.iters = 0
        self.simulation_steps = 0
        self.done = False
        
        # Reset the bounds
        self.bounds.reset()
        self.min_radius, self.max_radius = self.bounds.get_bounds(self.iters)
        
        
        # Initialize animation
        limits = (- self.max_radius - 0.2, self.max_radius + 0.2)
        self.animation = RocketAnimation(
            r_min=self.min_radius, r_target=self.target_radius, r_max=self.max_radius,
            xlim=limits, ylim=limits, t_vec_len=self.t_vec_len, circle_alpha=self.circle_alpha)
        self.animation.render(init_state, np.array([0, 0]), self.min_radius, self.target_radius, self.max_radius)
        
        return self._state_transform()
    
    def _state_transform(self):
        if self.polar:
            state = self._cartesian_to_polar(self.state)
        else:
            state = self.state
            
        if self.state_output_mode == 'No Theta':
            state = state[[0, 2, 3]]
            
        if self.state_target_r:
            state = np.array([*state, self.target_radius])
            
        if self.state_target_l:
            print('Warning: Only Circular')
            state = np.array([*state, np.sqrt(self.target_radius * self.G * self.M)])
            
        return state

    def _reward(self, state):
        '''
        Return the reward at a given position

        pos: np.array (self.dim, )
        '''
        state = np.array(state)
        pos, vel = state[:2], state[2:]
        l0 = np.sqrt(self.target_radius * self.G * self.M)
        # r = np.linalg.norm(pos)
        # rhat = pos / r
        # thetahat = [-rhat[1], rhat[0]]
        # thetadot = vel @ thetahat / r
        l = pos[0] * vel[1] - pos[1] * vel[0]
        circularization_penalty =  -self.penalty_function(np.linalg.norm(pos) - self.target_radius) * self.circularization_penalty 
        ang_momentum_penalty = -self.penalty_function(l - l0) * self.ang_momentum_penalty
        
        return circularization_penalty + ang_momentum_penalty
    
    def _cartesian_to_polar(self, state):
        pos = state[:2]
        vel = state[2:]
        r = np.linalg.norm(pos)
        r_hat = pos / r
        theta_hat = np.array([-r_hat[1], r_hat[0]])
        theta = np.arctan2(pos[1], pos[0])
        r_dot = vel @ r_hat
        theta_dot = vel @ theta_hat / r
        return np.array([r, theta, r_dot, theta_dot])

    def step(self, action):
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
        if self.done:
            print('Warning: Stepping after done is True')

        r, v = self.state[:self.state_space_dim //
                          2], self.state[self.state_space_dim//2:]
        done = False
        reward = 0
        info = dict()
        
        # Read the new bounds
        self.min_radius, self.max_radius = self.bounds.get_bounds(self.iters)

        for _ in range(self.iter_steps):
            # Calculate total force
            gravitational_force = - (self.G * self.M * self.m) / \
                np.power(np.linalg.norm(r), 3) * r  # F = - GMm/|r|^3 * r
            # Point the thrust in the direction of travel
            thrust_force, thrust_penalty = self._get_thrust_and_penalty(action)
            total_force = gravitational_force + thrust_force
            # Update position and location, this can somehow guarantee energy conservation
            v = v + total_force / self.m * self.dt
            r = r + v * self.dt
            # reward for staying inbounds 
            reward += (self._reward([*r, *v]) + self.inbounds_reward - thrust_penalty * self.thrust_penalty) * self.dt
            self.simulation_steps += 1
            # If out-of-bounds, end the game
            if not self.ignore_bounds:
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
                
        self.animation.render(self.state, thrust_force, self.min_radius, self.target_radius, self.max_radius)
        
        state = self._state_transform()

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
    