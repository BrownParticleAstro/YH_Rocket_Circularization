# Orbital Simulation Project

This project simulates 2D orbital maneuvers using reinforcement learning (PPO) and includes dynamic visualization, customizable training, and testing setups.

## Overview

The primary goal of this project is to simulate spacecraft orbital control using a reinforcement learning approach, specifically through the use of the PPO algorithm from the `stable-baselines3` library. The simulation focuses on controlling a spacecraft in orbit using prograde and retrograde thrust, with the goal of achieving stable orbits or other predefined objectives.

The project provides functionality for:

- **Training**: Learning optimal control strategies through PPO.
- **Testing**: Evaluating the performance of trained models in unseen scenarios.
- **Rendering**: Visualizing the simulation, including spacecraft position, velocity vectors, and thrust vectors.

## File Structure

- **`environment.py`**  
  Defines the `OrbitalEnv` and `OrbitalEnvWrapper` classes, which handle the simulation of spacecraft orbits, including state transitions, actions (thrust), and reward calculations. The wrapper allows for flexible configuration of the environment parameters.

  **Customization**:  
  - Adjust physics-related parameters like thrust, gravity, and timestep.
  - Modify the reward function to encourage different spacecraft behaviors, such as circularizing an orbit, minimizing fuel usage, or maintaining a stable orbit.

- **`model.py`**  
  Contains the neural network used for the PPO agent, including separate neural networks for processing various features like position, velocity, angular momentum, and energy.

  **Customization**:  
  - Modify the architecture of the model or feature extractor to process additional inputs or change how orbital states are handled.
  - Experiment with different neural network configurations to improve learning efficiency for specific tasks.

- **`render.py`**  
  This file handles dynamic rendering of the spacecraft’s trajectory, actions, and other relevant data using `matplotlib`. It provides a flexible system for generating custom visualizations by allowing users to register their own figure generation functions. This system enables the user to extend the rendering functionality beyond the default visualization provided.

  **Customization**:  
  Users can write their own figure generation functions and register them with the `Renderer` class using the `register_figure` method. This allows for tailored visualizations of different aspects of the spacecraft’s behavior (e.g., velocity vectors, action distribution over time, or energy changes).

  The only implemented figure generator at the moment is `"combined"`, which provides a visualization that includes:
  - **Radius Over Time**: A line plot that shows the radial distance of the spacecraft from the central body as a function of timesteps.
  - **Action Over Time**: A line plot showing the magnitude of the actions taken (thrust) over the episode’s timesteps.
  - **Orbit Plot**: A dynamic plot that shows the spacecraft’s position in 2D space along with its velocity vector and thrust vector at each timestep.

  The `"combined"` figure provides an overall view of the spacecraft’s orbit, actions, and trajectory over time, making it useful for analyzing the agent's decision-making and performance in the environment.

  **Example of registering a custom figure**:  
  Users can add new custom figures by defining their own figure generation function and registering it in the `Renderer` class. This allows for specialized visualizations based on different aspects of the simulation.

- **`run_and_view_episode.py`**  
  A utility to run and visualize a single episode of the simulation. Useful for debugging or exploring the behavior of the trained model in specific scenarios.

  **Customization**:  
  - Adjust the environment or model for targeted test runs.
  - Visualize different aspects of the spacecraft’s behavior during an episode (e.g., focus on velocity or thrust patterns).

- **`train.py`**  
  This script trains the PPO model using `stable-baselines3` in the `OrbitalEnvWrapper` environment. The training process includes saving the best model based on average episode reward.

  **Customization**:  
  - Modify training hyperparameters such as learning rate, gamma, or the total number of timesteps.
  - Switch between different reward functions to encourage varied spacecraft control strategies.
  - Customize the `SaveBest` callback to adjust what data is saved during training and how performance is tracked.

- **`test.py`**  
  This script tests a trained PPO model by running multiple episodes in the environment and saving the results for later analysis.

  **Customization**:  
  - Adjust the number of episodes to run during testing.
  - Load different model weights to compare the performance of various training runs.

## Customization

This project is highly flexible and can be adapted to explore different orbital control strategies by modifying:

- **Environment dynamics**: You can adjust the thrust capabilities, gravity model, or reward structure to create different orbital challenges for the agent to solve.
- **Neural network architecture**: Modify the `create_model` function to explore different architectures for processing orbital states or handling the PPO policy.
- **Training setup**: Adjust PPO hyperparameters like the learning rate, gamma, or number of timesteps to optimize the training process.
- **Visualization**: Create new plots and visualizations for analyzing the agent's performance.

## Example Usage

### Training

Train the model by creating the training environment and calling the `train_model` function. The trained model and data will be saved to a specified directory.

### Testing

Load the trained model and test it in the environment. Test results will be saved for further analysis.

### Rendering

Create a renderer instance using the trained model's save path and render specific episodes from the training or testing data. Custom figure generation is supported by registering new figure generators in the `Renderer` class.