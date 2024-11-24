from stable_baselines3 import PPO
import numpy as np
import os

""" Runs a test episode of the trained model on the environment and saves the results. """
def test_model(env, model_path, model_save_path, episode_num=1):
    """
    Load the trained model, run test episodes, collect data, and save it.
    
    Args:
        env: The environment to test on.
        model_path: Path to the saved model.
        model_save_path: Path where to save the test data.
        episode_num: Number of episodes to run.
    """
    # Load the trained model
    model = PPO.load(model_path)
    
    # Prepare to save data
    test_save_path = os.path.join(model_save_path, 'testing')
    os.makedirs(test_save_path, exist_ok=True)
    
    for ep in range(1, episode_num+1):
        obs = env.reset()
        done = False
        episode_data = []
        step = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Collect data
            state = info['state']  # (x, y, vx, vy)
            action_value = action[0]
            r_err_norm = info.get('r_err_norm', 0)
            d_r_err_norm = info.get('d_r_err_norm', 0)
            int_r_err_norm = info.get('int_r_err_norm', 0)
            
            episode_data.append((
                *state, step, action_value, reward, r_err_norm, d_r_err_norm, int_r_err_norm
            ))
            
            step += 1
            
        # Save episode data
        np.savez(os.path.join(test_save_path, f'episode_{ep}.npz'),
            x=np.array([d[0] for d in episode_data]),
            y=np.array([d[1] for d in episode_data]),
            vx=np.array([d[2] for d in episode_data]),
            vy=np.array([d[3] for d in episode_data]),
            episode_step=np.array([d[4] for d in episode_data]),
            action=np.array([d[5] for d in episode_data]),
            reward=np.array([d[6] for d in episode_data]),
            r_err_norm=np.array([d[7] for d in episode_data]),
            d_r_err_norm=np.array([d[8] for d in episode_data]),
            int_r_err_norm=np.array([d[9] for d in episode_data]))
    print(f"Test episode(s) completed and saved in {test_save_path}")