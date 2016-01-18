from __future__ import division

import numpy as np
import pylab as plt

PLOT_STEP = 5


def ic(x):
    q0 = np.zeros_like(x)
    q0[(x > 0) & (x < 5120)] = 1
    return q0


def ctcs_advection(nt, u=80, dt=2, nx=100, L=25600, K=0):
    dx = L / nx
    c = u * dt / dx
    print('c={}'.format(c))
    x = np.linspace(0, L, nx + 1)
    q0 = ic(x)
    qs = [q0]
    t = 0
    ts = [t]
    q1 = (q0 
          - c * (q0 - np.roll(q0, 1)) 
          + K * dt * (np.roll(q0, -1) - 2 * q0 + np.roll(q0, 1)) / dx**2)
    qs.append(q1)
    ts.append(t + dt)
    for timestep in range(2, nt + 1):
        t = timestep * dt
        ts.append(t)
        q2 = (q0 
              - c * (np.roll(q1, -1) - np.roll(q1, 1))
              + 2 * K * dt * (np.roll(q0, -1) - 2 * q0 + np.roll(q0, 1)) / dx**2)
        qs.append(q2)

        q0 = q1
        q1 = q2
    return x, qs, ts


def plot_qs(x, u, qs, ts, L=25600, step=PLOT_STEP):
    for q, t in zip(qs, ts)[::step]:
        plt.clf()
        plt.plot(x, q)
        q_an = ic((x - u * t) % L)
        plt.plot(x, q_an)
        plt.ylim((-0.5, 1.5))
        plt.pause(0.01)
    raw_input('Press enter to continue...')


if __name__ == '__main__':
    u = 80 # km hr^-1
    # Task B: oscillations due to num. dispersion
    x, qs, ts = ctcs_advection(160, dt=2, u=u)
    plot_qs(x, u, qs, ts)

    # Task C: c > 1 therefore unstable
    x, qs, ts = ctcs_advection(5, dt=4, u=u)
    plot_qs(x, u, qs, ts)

    # Task D: Diffusion
    x, qs, ts = ctcs_advection(160, u=u, K=540)
    plot_qs(x, u, qs, ts)

    # Task E: Diffusion weak diffusion
    x, qs, ts = ctcs_advection(160, u=u, K=540/5)
    plot_qs(x, u, qs, ts)

    # Task E: Diffusion strong diffusion
    x, qs, ts = ctcs_advection(160, u=u, K=540 * 5)
    plot_qs(x, u, qs, ts)

    # Task E: Diffusion overly strong diffusion - unstable.
    x, qs, ts = ctcs_advection(160, u=u, K=540 * 10)
    plot_qs(x, u, qs, ts)
