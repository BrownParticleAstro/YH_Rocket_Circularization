from csv import writer
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')


class RocketAnimation(object):
    def __init__(self, r_min=0.1, r_target=1, r_max=10, xlim=(-10.2, 10.2), ylim=(-10.2, 10.2), markersize=10):

        def _circle(radius):
            theta = np.linspace(0, 2 * np.pi, 100)
            x, y = radius * np.cos(theta), radius * np.sin(theta)
            return x, y

        self.fig = plt.figure(figsize=(6, 6))
        self.ax = plt.axes(xlim=xlim, ylim=ylim)
        self.line, = self.ax.plot([], [], marker='o', markersize=markersize)

        self.ax.plot(*_circle(r_min), '--', label='Minimum Radius')
        self.ax.plot(*_circle(r_target), '--', label='Target Orbit')
        self.ax.plot(*_circle(r_max), '--', label='Maximum Radius')

        self.ax.grid(True)
        self.ax.legend()

        self.record = list()

    def init(self,):
        self.line.set_data([], [])
        return self.line,

    def animate(self, i):
        self.line.set_data([self.record[i][0]], [self.record[i][1]])
        return self.line,

    def show_animation(self,):
        anim = FuncAnimation(self.fig, self.animate, init_func=self.init, frames=len(
            self.record), interval=10)
        plt.show()

    def render(self, state):
        self.record.append(state)


if __name__ == '__main__':
    pass
