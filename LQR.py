from scipy.linalg import solve_continuous_are as solve_care
from scipy.linalg import solve_discrete_are as solve_dare
import numpy as np

class LQR:
  def __init__(self, mu):
    self.mu = mu
    self.r0 = None
    self.S = None
    self.K = None

  def insert_target(self, r0, l0=None, mu=1):
    l0 = np.sqrt(r0 * self.mu)

    if r0 == self.r0:
      return self.K, self.S

    self.r0 = r0
    
    A = np.array([0, 0, 0, 0, 0, 1, -self.mu / np.power(r0, 3), 2*np.sqrt(self.mu)/np.power(r0, 5/2), 0]).reshape(3, 3)
    B = np.array([0, r0, 0, 0, 1, 0]).reshape(3, 2)
    Q = np.diag([1/r0**2, 1/l0**2, 0]).reshape(3, 3)
    R = 0.1 * np.eye(2)

    self.S = solve_care(A, B, Q, R)
    self.K = np.linalg.inv(R) @ B.T @ self.S

    return self.K, self.S

  def act(self, state):
    r, rdot, thetadot, r0 = state
    K, _ = self.insert_target(r0)
    x = np.array([r**2 * thetadot - np.sqrt(self.mu * r0), r - r0, rdot])
    return -K @ x
