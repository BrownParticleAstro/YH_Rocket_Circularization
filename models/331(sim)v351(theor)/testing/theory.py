from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

# Define the ODE system for nonlinear gravitational acceleration
def gravitational_fall(t, state, GM):
    r, v = state  # r: radial position, v: radial velocity
    drdt = v
    dvdt = -GM / (r**2)  # Gravitational acceleration
    return [drdt, dvdt]

# Initial conditions and parameters
r0 = 1.0  # Initial radius
v0 = 0.0  # Initial velocity
GM = 1.0  # Gravitational constant
t_span = (0, 1.0)  # Time span from t=0 to t=1 second
initial_state = [r0, v0]

# Solve the ODE using a numerical integrator
solution = solve_ivp(gravitational_fall, t_span, initial_state, args=(GM,), method='RK45', dense_output=True)

# Get the radial position at t=1 second
t_eval = [1.0]  # Evaluate at t=1.0
r_at_1_second = solution.sol(t_eval)[0][0]
r_at_1_second

# Define the time points for evaluation
t_eval_full = np.linspace(0, 1.0, 100)

# Solve the ODE for the full time span
solution_full = solve_ivp(gravitational_fall, t_span, initial_state, args=(GM,), t_eval=t_eval_full)

# Extract radius values over time
radius_over_time = solution_full.y[0]  # First row corresponds to r(t)
time_points = solution_full.t

# Plot the radius over time
plt.figure(figsize=(10, 6))
plt.plot(time_points, radius_over_time, label='Radius (r) vs. Time (t)', color='blue')
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8, label='Central Mass (r=0)')
plt.title('Radial Position Over Time Under Nonlinear Gravity', fontsize=14)
plt.xlabel('Time (t) [s]', fontsize=12)
plt.ylabel('Radius (r)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()
