import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

# Renderer is responsible for dynamically rendering the episode data from the test phase using matplotlib.
class Renderer:
    def __init__(self, model_save_path, custom_render_callback=None):
        """
        Args: 
            model_save_path: Path where the test and training episode data is stored (str).
            custom_render_callback: Optional callback function for custom rendering logic (callable). 
                                    If provided, it overrides the default rendering logic.
        Returns: None. Initializes internal state and figure handles.
        """
        self.model_save_path = model_save_path
        self.custom_render_callback = custom_render_callback
        self.data = {
            "training": [],
            "testing": []
        }
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

    def load_data(self):
        """
        Loads all training and testing episode data from storage and stores them into a data structure.
        
        Returns:
            None. Populates the `data` dictionary with training and testing episode data.
        """
        # Load training episodes
        training_data_dir = os.path.join(self.model_save_path, "training")
        self._load_episodes_from_directory(training_data_dir, "training")
        
        # Load testing episodes
        testing_data_dir = os.path.join(self.model_save_path, "testing")
        self._load_episodes_from_directory(testing_data_dir, "testing")
        
        print(f"Training & Testing episodes loaded!")

    def _load_episodes_from_directory(self, directory_path, data_type):
        """
        Helper method to load episode data from a specified directory and append it to the data structure.
        
        Args:
            directory_path: Path to the directory containing episode data (str).
            data_type: Either "training" or "testing" (str), specifies where the data will be stored.

        Returns:
            None. Updates the `data` dictionary with the loaded episodes.
        """
        # Check if the directory exists
        if not os.path.exists(directory_path):
            return
        
        # Iterate through each .npz file in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.npz'):
                filepath = os.path.join(directory_path, filename)
                if not os.path.exists(filepath):
                    continue

                episode_data = self._load_episode_file(filepath)
                
                if episode_data is not None:
                    self.data[data_type].append(episode_data)
    
    def _load_episode_file(self, filepath):
        """
        Loads a single episode data file (.npz) and returns its contents as a dictionary.
        
        Args:
            filepath: Path to the episode file (str).
        
        Returns:
            Dictionary with episode data, or None if loading failed.
        """
        try:
            data = np.load(filepath)
            episode_data = {
                "x": data['x'],
                "y": data['y'],
                "vx": data['vx'],
                "vy": data['vy'],
                "timestep": data['timestep'],
                "action": data['action']
            }
            return episode_data
        except Exception as e:
            return None

    def render(self, mode='human', episode_num=1, data_type='testing'):
        """
        Dynamically renders the episode data, including the spaceship's position, velocity, and thrust over time.
        Args:
            mode: Render mode. Default is 'human' (str).
            episode_num: Episode number to render (int).
            data_type: Either "training" or "testing" (str).
        
        Returns:
            None. Updates the plot dynamically.
        """
        matplotlib.use('TkAgg')
        plt.ion()
        self.load_data()

        # Ensure that the episode number is within the range of available data
        if episode_num > len(self.data[data_type]) or episode_num <= 0:
            print(f"Episode number {episode_num} is out of range.")
            return

        # Extract the data for the specified episode
        episode_data = self.data[data_type][episode_num - 1]
        self.state = list(zip(episode_data['x'], episode_data['y'], episode_data['vx'], episode_data['vy'])) # list of tuples: (x, y, vx, vy)
        self.timestep_history = episode_data['timestep']
        self.action_history = episode_data['action']
        self.radius_history = [np.sqrt(x**2 + y**2) for x, y in zip(episode_data['x'], episode_data['y'])]

        # Check if a custom render callback is provided
        if self.custom_render_callback:
            print("Using custom rendering callback.")
            self.custom_render_callback(self.state, self.radius_history, self.action_history, self.timestep_history)
            return

        # Otherwise, proceed with the default rendering logic
        self._default_render()

    def _default_render(self):
        """
        Default rendering logic if no custom rendering function is provided.
        """
        if self.state is None or len(self.state) == 0:
            print(f"No data available for rendering.")
            return

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
