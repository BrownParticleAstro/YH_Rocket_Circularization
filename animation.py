import numpy as np
from matplotlib.patches import FancyArrowPatch as Arrow
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
plt.style.use('bmh')


class RocketAnimation(object):
    def __init__(self, r_min=0.1, r_target=1, r_max=10, xlim=(-10.2, 10.2), ylim=(-10.2, 10.2), markersize=10, circle_alpha=1, t_vec_len=1):
        self.r_min = r_min
        self.r_target = r_target
        self.r_max = r_max

        self.marker_size = markersize
        self.circle_alpha = circle_alpha
        self.t_vec_len = t_vec_len

        self.states = []
        self.thrusts = []
        self.requested_thrusts = []

        self.rmin = []
        self.rtarget = []
        self.rmax = []

        self.Us = []  # List to store potential energy values

        self.xlim = xlim
        self.ylim = ylim

    def _circle(self, radius):
        theta = np.linspace(0, 2 * np.pi, 100)
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        return x, y

    def _init(self):
        self.arrow = Arrow(posA=(0, 0), posB=(0, 0), arrowstyle='simple', mutation_scale=10, color='r')
        self.ax.add_patch(self.arrow)
        self.line, = self.ax.plot([], [], marker='o', markersize=self.marker_size, alpha=self.circle_alpha)
        self.min_circle, = self.ax.plot(*self._circle(self.r_min), '--', label='Minimum Radius')
        self.target_circle, = self.ax.plot(*self._circle(self.r_target), '--', label='Target Orbit')
        self.max_circle, = self.ax.plot(*self._circle(self.r_max), '--', label='Maximum Radius')
        
        self.energy_line, = self.energy_ax.plot([], [], label='Potential Energy')  # Line for potential energy

        self.ax.grid(True)
        self.ax.legend(loc='upper left')

        self.energy_ax.grid(True)
        self.energy_ax.legend(loc='upper right')

        return (self.line, self.min_circle, self.target_circle, self.max_circle, self.energy_line)

    def _animate(self, i):
        st = self.states[i]
        r = np.linalg.norm(st[:2])
        U = -((2 * np.pi) ** 2) / r
        self.Us.append(U)

        self.line.set_data([st[0]], [st[1]])
        self.min_circle.set_data(*self._circle(self.rmin[i]))
        self.target_circle.set_data(*self._circle(self.rtarget[i]))
        self.max_circle.set_data(*self._circle(self.rmax[i]))
        self.energy_line.set_data(range(i + 1), self.Us)

        self.energy_ax.set_xlim(0, len(self.Us) + 1)
        self.energy_ax.set_ylim(min(self.Us) * 1.1, 0)  # Assuming potential energy is always negative

        return (self.line, self.min_circle, self.target_circle, self.max_circle, self.energy_line)

    def render(self, state, thrust, requested_thrust, rmin, rtarget, rmax):
        self.states.append(state)
        self.thrusts.append(thrust)
        self.requested_thrusts.append(requested_thrust)
        self.rmin.append(rmin)
        self.rtarget.append(rtarget)
        self.rmax.append(rmax)

        r = np.linalg.norm(state[:2])
        U = -((2 * np.pi) ** 2) / r
        self.Us.append(U)  # Calculate and store potential energy

if __name__ == '__main__':
    pass
