from csv import writer
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')


def RocketAnimation(object):
    def __init__(self, r_min=0.1, r_target=1, r_max=10, xlim=(-10.2, 10.2), ylim=(-10.2, 10.2), markersize=10):

        def _circle(radius):
            theta = np.linspace(0, 2 * np.pi, 100)
            x, y = radius * np.cos(theta), radius * np.sin(theta)
            return x, y

        self.fig = plt.figure(figsize=(6, 6))
        self.ax = plt.axes(xlim=xlim, ylim=ylim)
        self.line, = self.ax.plot([], [], marker='o', markersize=markersize)

        self.ax.plot(*_circle(0.1), '--', label='Minimum Radius')
        self.ax.plot(*_circle(1), '--', label='Target Orbit')
        self.ax.plot(*_circle(10), '--', label='Maximum Radius')

        self.ax.grid(True)
        self.ax.legend()

        self.x_record = list()
        self.y_record = list()

    def init(self,):
        self.line.set_data([], [])
        return self.line,

    def animate(self, i):
        self.line.set_data([self.x_record[i]], [self.y_record[i]])
        return self.line,
    
    def show_animation(self,):
        anim = FuncAnimation(self.fig, self.animate, init_func=self.init, frames=500, interval=20)
        plt.show()


if __name__ == '__main__':
    pass
