import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

# Renderer is responsible for dynamically rendering the episode data from the test phase using matplotlib.
class Renderer:
    def __init__(self, model_save_path):
        """
        Args: model_save_path: Path where the test episode data is stored (str).
        Returns: None. Initializes internal state and figure handles.
        """
        self.model_save_path = model_save_path
        self.radius_history = []
        self.action_history = []
        self.timestep_history = []
        self.fig = None
        self.ax_radius = None
        self.ax_action = None
        self.ax_orbit = None
        self.spaceship_plot = None
        self.velocity_arrow = None
        self.thrust_arrow = None
        self.star_plot = None
        self.orbit_circles = []
        self.state = None

    def load_data(self, episode_num):
        """
        Loads the state, action, and timestep data from a storage file.
        Args: episode_num: Episode number to load (int).
        Returns: None. Updates internal state variables with loaded data.
        """
        test_data_dir = os.path.join(self.model_save_path, "testing")
        filepath = os.path.join(test_data_dir, f'episode_{episode_num}.npz')
        data = np.load(filepath)
        
        x = data['x']
        y = data['y']
        vx = data['vx']
        vy = data['vy']
        actions = data['action']
        timesteps = data['timestep']
        
        # Compute the radius from x and y values
        radius = np.sqrt(x**2 + y**2)
        
        # Store the data for rendering
        self.radius_history = radius
        self.action_history = actions
        self.timestep_history = timesteps
        
        # Store the state to render at each step
        self.state = np.stack([x, y, vx, vy], axis=1)

    def render(self, mode='human', episode_num=1):
        """
        Dynamically renders the episode data, including the spaceship's position, velocity, and thrust over time.
        Args:
            mode: Render mode. Default is 'human' (str).
            episode_num: Episode number to render (int).

        Returns:
            None. Updates the plot dynamically.
        """
        matplotlib.use('TkAgg')
        plt.ion()

        if self.state is None:
            self.load_data(episode_num)

        x, y, vx, vy = self.state[0]
        r = np.sqrt(x**2 + y**2)
        v_radial = (x * vx + y * vy) / r
        v_tangential = (x * vy - y * vx) / r

        # Update radius and action histories
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 6))
            gs = GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1], figure=self.fig)
            self.ax_radius = self.fig.add_subplot(gs[0, 0])
            self.ax_action = self.fig.add_subplot(gs[1, 0])
            self.ax_orbit = self.fig.add_subplot(gs[:, 1])

            # Initialize static plot elements (star and orbit)
            self.star_plot, = self.ax_orbit.plot(0, 0, marker='*', markersize=15, color='red', label='Star (0,0)')
            
            # Add circles at integer radii with weaker color
            max_radius = 5
            for radius in range(1, max_radius + 1):
                circle = Circle((0, 0), radius, color=(0.5, 0.5, 0.5, 0.5), fill=False, linestyle='--')
                self.orbit_circles.append(circle)
                self.ax_orbit.add_artist(circle)

            # Initialize dynamic plot elements
            self.spaceship_plot, = self.ax_orbit.plot([], [], marker='o', markersize=10, color='blue')
            self.velocity_arrow = self.ax_orbit.arrow(0, 0, 0, 0, head_width=0.05, head_length=0.1, fc='green', ec='green')
            self.thrust_arrow = None
            plt.ion()

        # Update the spaceship and velocity arrow dynamically for each timestep
        for timestep_idx in range(len(self.timestep_history)):
            x, y, vx, vy = self.state[timestep_idx]
            r = self.radius_history[timestep_idx]
            action = self.action_history[timestep_idx]
            action = action.item() if isinstance(action, np.ndarray) else action

            # Update velocity arrow
            if self.velocity_arrow:
                self.velocity_arrow.remove()
            self.velocity_arrow = self.ax_orbit.arrow(x, y, vx, vy, head_width=0.05, head_length=0.1, fc='green', ec='green')

            # Update thrust arrow
            if action is not None:
                theta = np.arctan2(y, x)
                ax_x = -np.sin(theta) * action
                ax_y = np.cos(theta) * action

                if self.thrust_arrow:
                    self.thrust_arrow.remove()
                self.thrust_arrow = self.ax_orbit.arrow(x, y, ax_x, ax_y, head_width=0.05, head_length=0.1, fc='orange', ec='orange')

            # Generate the spaceship label
            spaceship_label = f'Spaceship \nx: {x:.2f}, y: {y:.2f}, \nr: {r:.3f}, \ntimestep: {timestep_idx}'
            self.spaceship_plot.set_data([x], [y])
            self.spaceship_plot.set_label(spaceship_label)

            # Update legends
            velocity_legend = Line2D([0], [0], color='green', lw=2,
                                     label=f'Velocity \nvx: {vx:.2f}, vy: {vy:.2f}, \nv_radial: {v_radial:.2f}, \nv_tangential: {v_tangential:.2f}')
            orbit_legend = Line2D([0], [0], color='grey', linestyle='--', label='Orbit (r=1)')
            thrust_legend = Line2D([0], [0], color='orange', lw=2, label=f'Thrust \nax_x: {ax_x:.3f}, ax_y: {ax_y:.3f}')

            # Prepare handles and labels for the legend
            handles = [self.star_plot, orbit_legend, self.spaceship_plot, velocity_legend]
            labels = [h.get_label() for h in handles]

            if action is not None:
                handles.append(thrust_legend)
                labels.append(thrust_legend.get_label())

            self.ax_orbit.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

            # Adjust plot limits dynamically
            margin = 0.2
            new_xlim = [min(x - margin, -1.5), max(x + margin, 1.5)]
            new_ylim = [min(y - margin, -1.5), max(y + margin, 1.5)]

            self.ax_orbit.set_xlim(new_xlim)
            self.ax_orbit.set_ylim(new_ylim)
            self.ax_orbit.set_aspect('equal')

            # Plot radius over time
            self.ax_radius.clear()
            self.ax_radius.plot(self.timestep_history[:timestep_idx+1], self.radius_history[:timestep_idx+1], color='blue')
            self.ax_radius.set_title('Radius Over Time')
            self.ax_radius.set_xlabel('Timestep')
            self.ax_radius.set_ylabel('Radius')

            # Plot action over time
            self.ax_action.clear()
            self.ax_action.plot(self.timestep_history[:timestep_idx+1], self.action_history[:timestep_idx+1], color='orange')
            self.ax_action.set_title('Action Over Time')
            self.ax_action.set_xlabel('Timestep')
            self.ax_action.set_ylabel('Action')

            # Adjust subplot spacing and margins
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.3)
            self.fig.tight_layout()

            # Draw and pause for dynamic rendering
            self.fig.canvas.draw()
            plt.pause(0.1)

    def close(self):
        """
        Closes the rendering window.
        Returns: None
        """
        plt.close(self.fig)