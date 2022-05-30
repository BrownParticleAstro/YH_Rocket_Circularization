from matplotlib.patches import FancyArrowPatch as Arrow
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
plt.style.use('seaborn-pastel')


class RocketAnimation(object):
    def __init__(self, r_min=0.1, r_target=1, r_max=10, xlim=(-10.2, 10.2), ylim=(-10.2, 10.2), markersize=10, t_vec_len=1):
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

        self.fig = plt.figure(figsize=(4, 4), num=1, clear=True)
        self.ax = plt.axes(xlim=xlim, ylim=ylim)
        self.t_vec_len = t_vec_len
        self.line, = self.ax.plot([], [], marker='o', markersize=markersize)
        self.arrow = Arrow(posA=(0, 0), posB=(
            0, 0), arrowstyle='simple', mutation_scale=10, color='r')
        self.ax.add_patch(self.arrow)

        self.min_circle, = self.ax.plot(
            *self._circle(r_min), '--', label='Minimum Radius')
        self.target_circle, = self.ax.plot(
            *self._circle(r_target), '--', label='Target Orbit')
        self.max_circle, = self.ax.plot(
            *self._circle(r_max), '--', label='Maximum Radius')

        self.ax.grid(True)
        self.ax.legend()

        self.states = list()
        self.thrusts = list()

        self.rmin = list()
        self.rtarget = list()
        self.rmax = list()

        self.xlim = xlim
        self.ylim = ylim

    def _circle(self, radius):
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
        vec = -self.thrusts[i] * self.t_vec_len * (self.xlim[1] - self.xlim[0])

        self.line.set_data([st[0]], [st[1]])
        self.min_circle.set_data(*self._circle(self.rmin[i]))
        self.target_circle.set_data(*self._circle(self.rtarget[i]))
        self.max_circle.set_data(*self._circle(self.rmax[i]))

        self.arrow.set_positions(posA=st[:2], posB=st[:2] + vec)
        self.ax.set_title(f'Iteration: {i}')
        # self.arrow = self.ax.arrow(st[0], st[1], vec[0], vec[1])
        return self.line, self.min_circle, self.target_circle, self.max_circle

    def show_animation(self,):
        '''
        Shows the animation in a pop-up window
        '''
        anim = FuncAnimation(self.fig, self._animate, init_func=self._init, frames=len(
            self.states), blit=True, interval=100, repeat=False)
        plt.show()

    def save_animation(self, name):
        '''
        Save the animation in a file

        Parameter:
            name: str, the file name
        '''
        anim = FuncAnimation(self.fig, self._animate, init_func=self._init, frames=len(
            self.states), blit=True, interval=100, repeat=False)
        anim.save(name)

    def render(self, state, thrust, rmin, rtarget, rmax):
        '''
        Records the current state in the animation for future rendering

        Parameters:
            state: the current state to render
        '''
        self.states.append(state)
        self.thrusts.append(thrust)
        self.rmin.append(rmin)
        self.rtarget.append(rtarget)
        self.rmax.append(rmax)


if __name__ == '__main__':
    pass
