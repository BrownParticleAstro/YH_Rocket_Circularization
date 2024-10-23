# Orbital Simulation Project

## Overview
This project simulates 2D orbital maneuvers using reinforcement learning (PPO) and includes dynamic visualization, customizable training, and testing setups.

## File Structure

- **`environment.py`**  
  Defines the orbital environment (`OrbitalEnv`, `OrbitalEnvWrapper`) and handles state transitions, action effects, and reward calculations.

  **Customization:**  
  Adjust physics parameters like thrust, gravity, timestep, or modify the reward function to encourage different behaviors.

- **`model.py`**  
  Contains the custom neural network used for PPO, including separate processing for orbital states like velocity and angular momentum.

  **Customization:**  
  Modify the network architecture or feature extractor to accommodate new inputs or improve learning efficiency.

- **`render.py`**  
  Handles visualization of the spacecraft’s trajectory and actions using matplotlib.

  **Customization:**  
  Adjust visual elements (e.g., colors, plot details) or the frequency of rendering updates.

- **`run_and_view_episode.py`**  
  Runs and visualizes a single episode. Useful for debugging model behavior.

  **Customization:**  
  Change the environment or model for specific test runs.

- **`train.py`**  
  Trains the PPO model using stable-baselines3.

  **Customization:**  
  Modify hyperparameters (learning rate, gamma, etc.) or switch out the reward function to experiment with training outcomes.

- **`test.py`**  
  Tests a trained model over multiple episodes and saves the results.

  **Customization:**  
  Adjust the number of test episodes or load different model weights for performance comparison.

## Customization
This project is designed for flexibility. You can easily modify environment dynamics, neural network architecture, or reinforcement learning parameters to explore different orbital control strategies.
