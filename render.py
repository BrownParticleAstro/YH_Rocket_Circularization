import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec

# Renderer is responsible for dynamically rendering the episode data from the test phase using matplotlib.
class Renderer:
    def __init__(self, model_save_path):
        """
        Args: model_save_path: Path where the episode data is stored (str).
        Returns: None. Initializes internal state and figure handles.
        """
        self.model_save_path = model_save_path
        self.radius_history = []
        self.action_history = []
        self.timestep_history = []
        self.fig = None
        self.state = None

        # Dictionary to store figure generation functions by name
        self.fig_generators = {
            "combined": self._generate_combined_fig,  # Default combined figure
            # You can add more figure types here in the future
        }

    def load_data(self, episode_num, data_type="testing"):
        """
        Loads the state, action, and timestep data from a storage file.
        Args: 
            episode_num: Episode number to load (int).
            data_type: Folder from which to load data ('testing' or 'training') (str, optional).
        Returns: None. Updates internal state variables with loaded data.
        """
        data_dir = os.path.join(self.model_save_path, data_type)
        filepath = os.path.join(data_dir, f'episode_{episode_num}.npz')
        data = np.load(filepath)
        
        x = data['x']
        y = data['y']
        vx = data['vx']
        vy = data['vy']
        actions = data['action']
        timesteps = data['episode_step']
        
        # Compute the radius from x and y values
        radius = np.sqrt(x**2 + y**2)
        
        # Store the data for rendering
        self.radius_history = radius
        self.action_history = actions
        self.timestep_history = timesteps
        
        # Store the state to render at each step
        self.state = np.stack([x, y, vx, vy], axis=1)

    def render(self, episode_num=1, fig_name="combined", interval=1, filter_func=None, data_type="testing"):
        """
        Renders the animation for the specified figure type.
        Args:
            episode_num: Episode number to render (int).
            fig_name: The name of the figure configuration to use (str).
            interval: Frame processing interval. Only every interval-th frame will be processed into the animation (int).
            filter_func: A callable that takes an episode number and returns a boolean. If provided, episodes where this function 
                         returns True will be rendered. If None, the provided episode_num will be used (function, optional).
            data_type: Folder from which to load data ('testing' or 'training') (str, optional).
        """
        # Get list of episode numbers from the specified folder (testing or training)
        data_dir = os.path.join(self.model_save_path, data_type)
        episode_files = [f for f in os.listdir(data_dir) if f.startswith('episode_') and f.endswith('.npz')]

        # Extract episode numbers from filenames
        episode_numbers = [int(f.split('_')[1].split('.')[0]) for f in episode_files]

        # Apply filter_func if provided
        if filter_func is not None:
            episode_numbers = list(filter(filter_func, episode_numbers))
        else:
            # Use only the provided episode_num if no filter_func is given
            episode_numbers = [episode_num]

        # Render each episode that satisfies the filter condition
        for ep_num in episode_numbers:
            print(f"Rendering episode {ep_num} from {data_type} data...")
            # Load the data for the current episode from the specified folder
            self.load_data(ep_num, data_type)

            # Call the selected figure generator function
            if fig_name in self.fig_generators:
                self.fig_generators[fig_name](interval, ep_num, data_type)
            else:
                raise ValueError(f"Figure type '{fig_name}' is not registered.")

    def _generate_combined_fig(self, interval, episode_num, data_type):
        """
        Generates the default combined figure with radius, action, and orbit plots.
        Args:
            interval: Frame processing interval (int).
            episode_num: Episode number being rendered (int).
            data_type: Folder from which to load data ('testing' or 'training') (str, optional).
        """
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 6))
            gs = plt.GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1], figure=self.fig)
            self.ax_radius = self.fig.add_subplot(gs[0, 0])
            self.ax_action = self.fig.add_subplot(gs[1, 0])
            self.ax_orbit = self.fig.add_subplot(gs[:, 1])

            # Initialize static plot elements (star and orbit)
            self.ax_orbit.plot(0, 0, marker='*', markersize=15, color='red', label='Star (0,0)')
            
            # Add orbit circles at integer radii with weaker color (no labels)
            max_radius = 5
            for radius in range(1, max_radius + 1):
                circle = Circle((0, 0), radius, color=(0.5, 0.5, 0.5, 0.5), fill=False, linestyle='--')
                self.ax_orbit.add_artist(circle)

            # Initialize dynamic plot elements
            self.spaceship_plot, = self.ax_orbit.plot([], [], marker='o', markersize=10, color='blue', label='Spaceship')
            self.velocity_arrow = None
            self.thrust_arrow = None

            # Set titles and labels
            self.ax_radius.set_title('Radius Over Time')
            self.ax_radius.set_xlabel('Timestep')
            self.ax_radius.set_ylabel('Radius')

            self.ax_action.set_title('Action Over Time')
            self.ax_action.set_xlabel('Timestep')
            self.ax_action.set_ylabel('Action')

            self.ax_orbit.set_aspect('equal')
            self.ax_orbit.set_xlim([-max_radius - 1, max_radius + 1])
            self.ax_orbit.set_ylim([-max_radius - 1, max_radius + 1])
            self.ax_orbit.set_title('Orbit Over Time')

            def update(timestep_idx):
                print(f"timestep_idx: {timestep_idx}")
                x, y, vx, vy = self.state[timestep_idx]
                r = self.radius_history[timestep_idx]
                action = self.action_history[timestep_idx]
                action = action.item() if isinstance(action, np.ndarray) else action
                timestep = self.timestep_history[timestep_idx]

                # Compute v_radial and v_tangential
                r = np.sqrt(x**2 + y**2)
                v_radial = (x * vx + y * vy) / r
                v_tangential = (x * vy - y * vx) / r

                # Update spaceship position
                self.spaceship_plot.set_data([x], [y])

                # Set spaceship label with x, y, r, and timestep
                spaceship_label = f'Spaceship \nx: {x:.2f}, y: {y:.2f}, \nr: {r:.3f}, \nt: {timestep}'
                self.spaceship_plot.set_label(spaceship_label)

                # Update velocity arrow
                if self.velocity_arrow:
                    self.velocity_arrow.remove()
                self.velocity_arrow = self.ax_orbit.arrow(x, y, vx, vy, head_width=0.05, head_length=0.1, fc='green', ec='green')
                velocity_label = f'Velocity \nvx: {vx:.2f}, vy: {vy:.2f}, \nv_rad: {v_radial:.2f}, \nv_tan: {v_tangential:.2f}'

                # Update thrust arrow
                if action is not None:
                    theta = np.arctan2(y, x)
                    ax_x = -np.sin(theta) * action
                    ax_y = np.cos(theta) * action

                    if self.thrust_arrow:
                        self.thrust_arrow.remove()
                    self.thrust_arrow = self.ax_orbit.arrow(x, y, ax_x, ax_y, head_width=0.05, head_length=0.1, fc='orange', ec='orange')
                    thrust_label = f'Thrust \ntx: {ax_x:.2f}, ty: {ax_y:.2f}'

                # Dynamically adjust orbit plot limits
                arrow_fraction = 0.5
                max_x = max(abs(x + arrow_fraction * vx), abs(x + arrow_fraction * ax_x)) + 0.1
                max_y = max(abs(y + arrow_fraction * vy), abs(y + arrow_fraction * ax_y)) + 0.1
                max_dim = max(max_x, max_y)
                            
                self.ax_orbit.set_xlim([-max_dim, max_dim])
                self.ax_orbit.set_ylim([-max_dim, max_dim])

                # Update radius over time (line plot)
                self.ax_radius.clear()
                self.ax_radius.plot(self.timestep_history[:timestep_idx+1], self.radius_history[:timestep_idx+1], color='blue')
                self.ax_radius.set_title('Radius Over Time')
                self.ax_radius.set_xlabel('Timestep')
                self.ax_radius.set_ylabel('Radius')

                # Update action over time (line plot)
                self.ax_action.clear()
                self.ax_action.plot(self.timestep_history[:timestep_idx+1], self.action_history[:timestep_idx+1], color='orange')
                self.ax_action.set_title('Action Over Time')
                self.ax_action.set_xlabel('Timestep')
                self.ax_action.set_ylabel('Action')

                # Re-add the correct legend for velocity and thrust
                self.ax_orbit.legend([self.spaceship_plot, self.velocity_arrow, self.thrust_arrow], 
                                    [spaceship_label, velocity_label, thrust_label], loc='upper right')

                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.3)
                self.fig.tight_layout()

        frames_to_use = range(0, len(self.timestep_history), interval)
        ani = animation.FuncAnimation(self.fig, update, frames=frames_to_use, interval=50, repeat=False)

        animation_save_path = os.path.join(self.model_save_path, data_type, f"episode_{episode_num}_animation_interval_{interval}.mp4")
        ani.save(animation_save_path, writer='ffmpeg')

        print(f"Animation saved to {animation_save_path} with interval {interval}")

    def close(self):
        """
        Closes the rendering window.
        Returns: None
        """
        plt.close(self.fig)
