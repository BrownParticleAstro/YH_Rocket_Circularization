from matplotlib.patches import FancyArrowPatch as Arrow
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
plt.style.use('bmh')


class RocketAnimation(object):
    def __init__(self, r_min=0.1, r_target=1, r_max=10, xlim=(-10.2, 10.2), ylim=(-10.2, 10.2), markersize=10, circle_alpha=1, t_vec_len=1):
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
        self.r_min = r_min
        self.r_target = r_target
        self.r_max = r_max

        self.marker_size = markersize
        self.circle_alpha = circle_alpha
        self.t_vec_len = t_vec_len

        self.states = list()
        self.thrusts = list()
        self.requested_thrusts = list()

        self.rmin = list()
        self.rtarget = list()
        self.rmax = list()

        self.Us = list()

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

        self.t_vec_len = self.t_vec_len
        self.arrow = Arrow(posA=(0, 0), posB=(
            0, 0), arrowstyle='simple', mutation_scale=10, color='r')
        self.ax.add_patch(self.arrow)
        self.line, = self.ax.plot(
            [], [], marker='o', markersize=self.marker_size, alpha=self.circle_alpha)

        self.min_circle, = self.ax.plot(
            *self._circle(self.r_min), '--', label='Minimum Radius')
        self.target_circle, = self.ax.plot(
            *self._circle(self.r_target), '--', label='Target Orbit')
        self.max_circle, = self.ax.plot(
            *self._circle(self.r_max), '--', label='Maximum Radius')

        self.ax.grid(True)
        if not hasattr(self, 'ax_legend_created'):
            self.ax.legend(loc='upper left')
            self.ax_legend_created = True

        self.thrustr, = self.thrustax.plot([], [], label='thrust magnitude')
        self.requested_thrustr, = self.thrustax.plot(
            [], [], label='requested thrust magnitude')
        self.thrustax.grid(True)
        if not hasattr(self, 'thrustax_legend_created'):
            self.thrustax.legend(loc='upper right')
            self.thrustax_legend_created = True

        self.stater, = self.stateax.plot([], [], label='state r')
        self.statetheta, = self.stateax.plot([], [], label='state $\\theta$')
        self.stateax.grid(True)
        if not hasattr(self, 'stateax_legend_created'):
            self.stateax.legend(loc='upper right')
            self.stateax_legend_created = True

        self.energy_line, = self.energyax.plot([], [], label='Potential Energy')  # Line for potential energy
        self.energyax.grid(True)
        if not hasattr(self, 'energyax_legend_created'):
            self.energyax.legend(loc='upper right')
            self.energyax_legend_created = True

        return self.line, self.min_circle, self.target_circle, self.max_circle, \
            self.thrustr, self.requested_thrustr,\
            self.stater, self.statetheta, self.energy_line

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
        vec = self.thrusts[i] * self.t_vec_len * (self.xlim[1] - self.xlim[0])

        self.line.set_data([st[0]], [st[1]])
        self.min_circle.set_data(*self._circle(self.rmin[i]))
        self.target_circle.set_data(*self._circle(self.rtarget[i]))
        self.max_circle.set_data(*self._circle(self.rmax[i]))

        self.arrow.set_positions(posA=st[:2], posB=st[:2] + vec)
        self.fig.suptitle(f'Iteration: {i}')

        self.thrustr.set_data([range(i)], self.thrusts_norm[:i])
        self.requested_thrustr.set_data(
            [range(i)], self.requested_thrusts_norm[:i])

        max_value = np.max([self.thrusts_norm, self.requested_thrusts_norm])
        self.thrustax.set_xlim(-0.5, len(self.thrusts_norm) + 0.5)
        self.thrustax.set_ylim(-max_value*0.1, max_value*1.1)

        self.stater.set_data([range(i)], self.rs[:i])
        max_value = np.max(np.abs(self.rs))
        min_value = np.min(np.abs(self.rs))
        self.stateax.set_xlim(-0.5, len(self.rs) + 0.5)
        self.stateax.set_ylim(min_value - max_value * .1, max_value*1.1)

        self.energy_line.set_data([range(i)], self.Us[:i])
        max_value = np.max(np.abs(self.Us))
        min_value = np.min(np.abs(self.Us))
        self.energyax.set_xlim(-0.5, len(self.Us) + 0.5)
        self.energyax.set_ylim(min_value - max_value * .1, max_value*1.1)

        return self.line, self.min_circle, self.target_circle, self.max_circle,\
            self.thrustr, self.requested_thrustr, \
            self.stater, self.statetheta, self.energy_line

    def show_animation(self, step=1):
        '''
        Shows the animation in a pop-up window
        '''
        self._transform_vectors()
        self.fig = plt.figure(figsize=(10, 5), num=1,
                              clear=True, tight_layout=True)
        self.ax = self.fig.add_subplot(1, 2, 1)
        self.thrustax = self.fig.add_subplot(3, 2, 2)
        self.stateax = self.fig.add_subplot(3, 2, 4)
        self.energyax = self.fig.add_subplot(3, 2, 6)
        frames_to_show = range(0, len(self.states), step)
        anim = FuncAnimation(self.fig, self._animate, init_func=self._init,
                            frames=frames_to_show, blit=True, interval=100, repeat=False)
        plt.show()

    def save_animation(self, name, step=1):
        '''
        Save the animation in a file

        Parameter:
            name: str, the file name
        '''
        self._transform_vectors()
        self.fig = plt.figure(figsize=(10, 7), num=1,
                              clear=True, tight_layout=True)
        self.ax = self.fig.add_subplot(1, 2, 1)
        self.thrustax = self.fig.add_subplot(3, 2, 2)
        self.stateax = self.fig.add_subplot(3, 2, 4)
        self.energyax = self.fig.add_subplot(3, 2, 6)
        frames_to_show = range(0, len(self.states), step)
        anim = FuncAnimation(self.fig, self._animate, init_func=self._init,
                             frames=frames_to_show, blit=True, interval=100, repeat=False)
        anim.save(name, dpi=80)

    def _get_transforms(self, states):

        transforms = list()
        rs = list()
        thetas = list()
        for st in states:
            pos, vel = st[:2], st[2:]
            r = np.linalg.norm(pos)
            theta = np.arctan2(pos[1], pos[0])
            rhat = pos / r
            rot_mat = np.array([[rhat[0], -rhat[1]], [rhat[1], rhat[0]]])
            transforms.append(rot_mat)
            rs.append(r)
            thetas.append(theta)

        return transforms, rs, thetas

    def _forward_transform(self, transforms, vecs):
        return [tr @ vec for tr, vec in zip(transforms, vecs)]

    def _inverse_transform(self, transforms, vecs):
        return [tr.T @ vec for tr, vec in zip(transforms, vecs)]

    def _transform_vectors(self, ):
        transforms, self.rs, self.thetas = self._get_transforms(self.states)
        self.vel_polar = self._inverse_transform(
            transforms, [st[2:] for st in self.states])
        self.thrusts_polar = self._inverse_transform(transforms, self.thrusts)
        self.requested_thrusts_polar = self._inverse_transform(
            transforms, self.requested_thrusts)
        self.thrusts_norm = [np.linalg.norm(thrust) for thrust in self.thrusts]
        self.requested_thrusts_norm = [np.linalg.norm(
            thrust) for thrust in self.requested_thrusts]
        self.thrust_direction = [np.arctan2(
            thrust[1], thrust[0]) for thrust in self.thrusts_polar]
        self.requested_thrust_direction = [np.arctan2(
            thrust[1], thrust[0]) for thrust in self.requested_thrusts_polar]

    def render(self, state, thrust, requested_thrust, rmin, rtarget, rmax):
        '''
        Records the current state in the animation for future rendering

        Parameters:
            state: the current state to render
        '''
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
