import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
import multiprocessing

class OrbitalEnvironment:
    def __init__(self, r0=None, v0=None, max_steps=5000):
        self.G = 1.0
        self.M = 1.0
        self.dt = 0.01
        self.max_steps = max_steps

        self.init_r = r0 if r0 else np.random.uniform(0.6, 2.0)
        self.init_v = v0 if v0 else np.sqrt(self.G * self.M / self.init_r)

        self.trajectory = []
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.x = self.init_r + np.random.uniform(-0.5, 0.5)
        self.y = 0.0
        self.vx = 0.0
        self.vy = self.init_v + np.random.uniform(-0.5, 0.5)
        self.m = 1.0
        self.current_step = 0
        self.episode_reward = 0

        self.trajectory = [(self.x, self.y)]

        return self._get_state(), {}

    def _get_state(self):
        r = np.sqrt(self.x**2 + self.y**2)
        theta = np.arctan2(self.y, self.x)
        v_radial = (self.x * self.vx + self.y * self.vy) / r
        v_tangential = (self.x * self.vy - self.y * self.vx) / r
        angular_momentum = r * v_tangential

        return np.array([r, theta, v_radial, v_tangential, angular_momentum], dtype=np.float32)
    
    #@profile
    def rk4_integration(self, thrust):
        state = np.array([self.x, self.y, self.vx, self.vy])
        dt = self.dt

        def derivatives(s):
            x, y, vx, vy = s
            r = np.hypot(x, y)
            accel_gravity = -self.G * self.M * np.array([x, y]) / r**3
            accel_total = accel_gravity + thrust / self.m
            return np.array([vx, vy, *accel_total])

        k1 = derivatives(state)
        k2 = derivatives(state + 0.5 * dt * k1)
        k3 = derivatives(state + 0.5 * dt * k2)
        k4 = derivatives(state + dt * k3)

        state += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        self.x, self.y, self.vx, self.vy = state
        self.m -= np.linalg.norm(thrust) * dt * 0.001
        self.trajectory.append((self.x, self.y))

    #@profile
    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.item()) 

        r = np.sqrt(self.x**2 + self.y**2)
        thrust_magnitude = 0.1
        thrust_direction = np.array([-self.y, self.x])
        thrust_direction /= r

        thrust = {
            0: -thrust_magnitude * thrust_direction,
            1: np.array([0.0, 0.0]),
            2: thrust_magnitude * thrust_direction
        }[action]

        self.rk4_integration(thrust)
        reward = 1 if abs(r - 1.0) < 0.1 else 0
        if action == 0 or action == 2: reward -= 0.1

        self.episode_reward += reward
        terminated = r > 2.0 or r < 0.1
        truncated = self.current_step >= self.max_steps
        done = terminated or truncated

        info = {}
        if done:
            info['episode'] = {'r': self.episode_reward, 'l': self.current_step + 1}

        self.current_step += 1
        return self._get_state(), reward, terminated, truncated, info

    def render(self, episode_num=0):
        trajectory = np.array(self.trajectory)
        colors = cm.viridis(np.linspace(0, 1, len(trajectory)))

        plt.figure(figsize=(6, 6))
        for i in range(1, len(trajectory)):
            plt.plot(
                [trajectory[i-1, 0], trajectory[i, 0]],
                [trajectory[i-1, 1], trajectory[i, 1]],
                color=colors[i],
                alpha=0.7
            )

        plt.scatter(0, 0, color='orange', s=200, label='Central Body')
        circle = plt.Circle((0, 0), 1.0, color='blue', fill=False, linestyle='--', label='Target Orbit')
        plt.gca().add_artist(circle)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Episode {episode_num}')
        plt.legend()
        plt.grid()
        plt.show()

class OrbitalEnvWrapper(gym.Env):
    def __init__(self):
        super(OrbitalEnvWrapper, self).__init__()
        self.env = OrbitalEnvironment()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human', episode_num=0):
        self.env.render(episode_num=episode_num)

    def seed(self, seed=None):
        np.random.seed(seed)

""" python -m cProfile -o output.prof testing_new.py
snakeviz output.prof 
"""

""" sudo py-spy record -o profile.svg -- python testing_new.py
"""

def make_env(rank, seed=0):
    def _init():
        env = OrbitalEnvWrapper()
        env.seed(seed + rank)
        return env
    return _init

# Custom linear schedule
class LinearSchedule:
    def __init__(self, schedule_timesteps, initial_p, final_p):
        self.schedule_timesteps = schedule_timesteps
        self.initial_p = initial_p
        self.final_p = final_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    num_cpu = max(1, multiprocessing.cpu_count() - 1)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    exploration_initial_eps = 1.0
    exploration_final_eps = 0.05
    exploration_fraction = 1
    learning_rate_initial = 0.001
    learning_rate_final = 0.0001

    exploration_schedule = LinearSchedule(
        schedule_timesteps=int(1_000_000 * exploration_fraction),
        initial_p=exploration_initial_eps,
        final_p=exploration_final_eps
    )

    def custom_learning_rate(progress_remaining):
        return learning_rate_initial * progress_remaining + learning_rate_final * (1 - progress_remaining)

    model = DQN(
        'MlpPolicy',
        env,
        learning_rate=custom_learning_rate,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        exploration_fraction=exploration_fraction,
        verbose=1,
        tensorboard_log="./orbital_dqn_tensorboard/"
    )

    model.learn(total_timesteps=1_000_000)

    single_env = OrbitalEnvWrapper()
    for episode in range(10):
        obs, info = single_env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = single_env.step(action)
            done = terminated or truncated
        single_env.render(episode_num=episode)

    mean_reward, std_reward = evaluate_policy(model, single_env, n_eval_episodes=10)
    print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")