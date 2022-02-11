from csv import writer
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch as Arrow
plt.style.use('seaborn-pastel')


class RocketAnimation(object):
    def __init__(self, r_min=0.1, r_target=1, r_max=10, xlim=(-10.2, 10.2), ylim=(-10.2, 10.2), markersize=10, t_vec_len=20):
        '''
        Initialize Animation Object

        Parameters:
            r_min: the minimum radius circle
            r_target: the target radius circle
            r_max: the maximum radius circle
            x_lim: tuple of 2 elements, max and min bound of the axes on the x direction
            y_lim: tuple of 2 elements, max and min bound of the axes on the y direction
            markersize: int, the size of the marker indicating rocket
            t_vec_len: the scale of the thrust vector
        '''

        def _circle(radius):
            '''
            Create data for a circle with a certain radius

            Parameters:
                radius: the radius of the circle

            Return:
                tuple of np.ndarray representing the coordinates of each point
                on the circle
            '''
            theta = np.linspace(0, 2 * np.pi, 100)
            x, y = radius * np.cos(theta), radius * np.sin(theta)
            return x, y

        self.fig = plt.figure(figsize=(6, 6))
        self.ax = plt.axes(xlim=xlim, ylim=ylim)
        self.t_vec_len = t_vec_len
        self.line, = self.ax.plot([], [], marker='o', markersize=markersize)
        self.arrow = Arrow(posA=(0, 0), posB=(0, 0), arrowstyle='simple', mutation_scale=10, color='r')
        self.ax.add_patch(self.arrow)

        self.ax.plot(*_circle(r_min), '--', label='Minimum Radius')
        self.ax.plot(*_circle(r_target), '--', label='Target Orbit')
        self.ax.plot(*_circle(r_max), '--', label='Maximum Radius')

        self.ax.grid(True)
        self.ax.legend()

        self.states = list()
        self.thrusts = list()

    def _init(self,):
        '''
        Function used for generating the animation
        The first step in the animation

        Returns:
            line to update
        '''
        self.line.set_data([], [])
        return self.line,

    def _animate(self, i):
        '''
        Function used for generating the animation
        The update function run each time the animation advances

        Parameters:
            i: the number of frames of the animation

        Returns:
            line to update
        '''
        st = self.states[i]
        vec = -self.thrusts[i] * self.t_vec_len
        self.line.set_data([st[0]], [st[1]])
        self.arrow.set_positions(posA=st[:2], posB=st[:2] + vec)
        # self.arrow = self.ax.arrow(st[0], st[1], vec[0], vec[1])
        return self.line,

    def show_animation(self,):
        '''
        Shows the animation in a pop-up window
        '''
        anim = FuncAnimation(self.fig, self._animate, init_func=self._init, frames=len(
            self.states), interval=10, repeat=False)
        plt.show()


    def render(self, state, thrust):
        '''
        Records the current state in the animation for future rendering

        Parameters:
            state: the current state to render
        '''
        self.states.append(state)
        self.thrusts.append(thrust)


if __name__ == '__main__':
    pass
