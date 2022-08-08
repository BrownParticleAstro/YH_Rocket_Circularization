# Orbital Circularization with Reinforcement Learning

This code base is an attempt at solving an orbital circularization control
problem for a small craft around a fixed massive object with Reinforcement
Learning

## Requirements

- Python version: 3.8.12
- Numpy version: 1.19.5
- Matplotlib version: 3.5.0
- Tensorflow version: 2.6.0
- ffmpeg version: 4.2.2
- gym version: 0.21.0

Remark: These runs were completed on an Intel MacBook Pro with MacOS 12.3 
Monterey in a Conda environment. Compatibility may differ for other 
machines. 

## Models

### Deep Q Network (DQN)

#### General Environment

The notebook `rocket_dqn.ipynb` and file `rocket_dqn.py` both readily run 
the Deep Q Network implementations. They contain the newest hyperparameters
but may not be readily functional. Due to the implementation of the 
animation, the notebook version is not able to generate an animation. If 
you need animation at run time, please run `rocket_dqn.py`. We suggest 
training in the notebook and load the model to see the animated gameplay in 
`rocket_dqn.py`. For execution instructions, see [Execution](#Execution)

For hyperparameters along with versions with those parameters,
please refer to the 
[running notes](https://docs.google.com/document/d/1E03SGqcWgLNUoU_IawYxPLhMxpBwwk5Vu3pEcD3UJ4E/edit?usp=sharing)

#### Radial Thrust Environment

The 2D version of the circularization problem does not work at the moment. 
We transformed the problem into a 1D problem with only radial motion
and radial thrust. It has a separate gym environment. To train, use the
`radial_rocket.ipynb` notebook. The environment only provides a summary
chart, but no animation.

For an animation, use `rocket_dqn.py` with the 
2D environment. Note that for radial stabilization to work, we need
to set initialization function `init_func` to `target_l()` in the `reset` 
method in `rocket_gym.py`. This ensures that the rocket has the desired
angular momentum of a circular orbit at the target radius.
Additionally, the environment should also be configured with the class
decorators
```python
env = rocket_gym.RadialObservation(
    rocket_gym.RadialThrust(
        rocket_gym.PolarizeAction(env)))
```
and network settings
```python
model = DeepQNetwork(dims=[2, 128, 128, 3],
                    epsilon=1.0, epsilon_decay=.1, gamma=.95,
                    memory=100000, start_updating=10000,
                    batch_size=32, learning_rate=1e-4, descent_frequency=16, update_frequency=1,
                    use_target=True, target_frequency=8)
```

We will work on making that more customizable in the future. 

### Vanilla Policy Gradient (VPG, REINFORCE with Baseline)

A VPG implementation is also available, but without Experience Replay, it is
not as efficient as DQN. Some improved policy networks may work better, 
but they are not well-explored.

To run the VPG implementation, make sure that `wandb` is installed, and run

```bash
python main.py
```

As an alternative, the notebook `run.ipynb` can also run the VPG model.

### Linear Quadratic Regulator (LQR)

LQR is a type of simplified Optimal Control Problem. By linearizing the
dynamics near an equilibrium, we can approximate the Rocket Circularization
problem to an LQR problem. It works for states close enough to the equilibrium. For more information, checkout the [Google Colab demonstration](https://colab.research.google.com/drive/1gU3B9EPqj-WBo_FEwV74bbFS9yBnYnCg?usp=sharing).

To run LQR, uncomment the LQR code in `main.py` and run the file like before.

## Environment

### Dynamics

According to Newton's Law of Gravitation (Inverse Square Law),
we have the following second-order vector ODE. 

$$m\ddot{\mathbf{r}} + \frac{GMm}{r^3}\mathbf{r} = F\mathbf{u}$$

where $\mathbf{r}$ is the position vector with $r$ as the magnitude,
$G$, $M$, $m$ are the gravitational constant, the mass of the center
mass, and the mass of the craft respectively. $F$ is the magnitude 
of the force, with $\mathbf{u}$ being a control to the system with
magnitude $|\mathbf{u}|\leq 1$. To prevent overflows and underflows, 
we set natural units for the system, i.e. $G=1$, $M=1$. Additionally,
we set $m=.01$.

We use the Euler-Cromer Method to approximate the motion:
$$
\begin{cases}
\mathbf{v}_{t+1} = \mathbf{v}_t + \mathbf{a}_{net, t} \cdot \Delta t \\
\mathbf{r}_{t+1} = \mathbf{r}_t + \mathbf{v}_{t+1} \cdot \Delta t
\end{cases}
$$
where $\mathbf{r_t}$, $\mathbf{v_t}$ are position and velocity of the 
object at timestep $t$ respectively. $a_{net, t}$ is the total acceleration 
of the object at timestep $t$. The timestep is $\Delta t$. The timestep
may be subject to change based on simulation accuracy and simulation
speed, but it is around the magnitude of $.01$ or $.1$ based on the values
given above.

To ensure that observations stay in a resonable range for the network, we
clipped velocity as well as the radius with the norm. Additionally, when the 
craft hits some boundaries, it will loose all the velocity normal
to the boundary in an inelastic collision. This ensures that the craft stays
inbounds.

## Open AI Gym

Open AI gym is one of the standard APIs for Reinforcement Learning. 
Other APIs differ slightly but usually have a similar format.
For a detailed specification of Gym Environments, checkout [this link](https://www.gymlibrary.ml/content/api/). In this repository, `rocket_gym.py`
provides the gym environment. To use the environment, call the `make` 
function. More customization can be done in the `make` function itself.
```python
import rocket_gym
with rocket_gym.make('RocketCircularization-v0') as env:
    # Simulation Loop
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs, rwd, done, info = env.step(u)
    # Note that to produce an animation or a summary,
    # the show method must be run
    env.show()
```

The state is a vector of length `4`. The first two elements show the 
position, and the last two show the velocity, both in cartesian.
This `u` is a control vector in cartesian space. To obtain polar observations
and give polar controls, use the class wrappers provided in `rocket_gym`.

### Class Wrappers

To use class wrappers, simply apply them to the environment instance.
```python
env = rocket_gym.PolarizeAction(env)
```
Note that the order they are applied may matter depending on how they are 
implemented.

Reference the documentation in `rocket_gym.py` for more details on the class 
wrappers.

### Ongoing Experimentation

The details of the environment are not certain yet. This may include choice 
of simulation parameters, number of timesteps, reward structure, etc.
Reference running notes and documentation for more details.

### Deprecated Environment

`rocket_circularization.py` is an environment previously used for training. 
Without modifications, VPG and LQR still use this environment. It is also 
NOT an Open AI Gym environment, and is NOT guaranteed to work with the 
wrappers, but it does have settings that accomplish the those functions 
and possibly more.

## Code Structure

## Execution

To run the notebook, using VS Code with the `Jupyter Notebook` 
plugin should be sufficient. Otherwise, type `jupyter notebook` in the 
command line with the environment activated. Navigate to the file and it should run. 

Note that for changes to the imported files to apply, the notebook kernel
 need to be restarted to clear the module cache. If necessary, the `./__pycache__` folder can also be deleted and the notebook restarted.

To run the python file, use the command `python rocket_dqn.py`.