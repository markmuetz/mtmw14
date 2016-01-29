from __future__ import division

import numpy as np
from scipy.linalg import solve


def enso_oscillator(nt, method='rk4', dt=0.01, T0=1.125, h0=0, mu=2/3, b0=2.5, gamma=0.75, c=1, r=0.25, alpha=0.125, epsilon=0, xi=0):
    b = b0 * mu
    R = gamma * b - c
    T, h = T0, h0
    T, h = T0/7.5, h0/150
    Ts, hs = [T], [h]
    X = np.array([T, h])
    M = np.array([[1 - dt * R, -dt * gamma], [dt * alpha * b, 1 + dt * r]])

    def fT(t, T, h):
        return (R * T + gamma * h - epsilon * (h + b * T)**3)

    def fh(t, T, h):
        return -(r * h + alpha * b * T)

    for timestep in range(nt + 1):
        t = timestep * dt
        print(t)
        if method == 'ft':
            # Forward time, explicit.
            T = T + dt * (R * T + gamma * h - epsilon * (h + b * T)**3)
            h = h - dt * (r * h + alpha * b * T)
            Ts.append(T)
            hs.append(h)
        elif method == 'bt':
            # Backward time, implicit.
            X = solve(M, X)
            Ts.append(X[0])
            hs.append(X[1])
        elif method == 'rk4':
            # RK4.
            k1 = fT(t, T, h)
            l1 = fh(t, T, h)

            k2 = fT(t + dt/2, T + dt/2 * k1, h + dt/2 * l1)
            l2 = fh(t + dt/2, T + dt/2 * k1, h + dt/2 * l1)

            k3 = fT(t + dt/2, T + dt/2 * k2, h + dt/2 * l2)
            l3 = fh(t + dt/2, T + dt/2 * k2, h + dt/2 * l2)

            k4 = fT(t + dt, T + dt * k3, h + dt * l3)
            l4 = fh(t + dt, T + dt * k3, h + dt * l3)

            T = T + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            h = h + dt/6 * (l1 + 2*l2 + 2*l3 + l4)

            Ts.append(T)
            hs.append(h)

    return np.array(Ts) * 7.5, np.array(hs) * 150
    #return np.array(Ts), np.array(hs)


def enso_annual_oscillator(nt, dt=1, T0=1.125, h0=0, mu0=0.75, mu_ann=0.2, tau=6, b0=2.5, gamma=0.75, c=1, r=0.25, alpha=0.125, epsilon=0.1):
    T, h = T0/7.5, h0/150
    Ts, hs = [T], [h]
    for timestep in range(1, nt + 1):
        t = timestep * dt
        print(t)

        mu = mu0 * (1 + mu_ann * np.cos(2 * np.pi * t / tau - 5 * np.pi / 6))
        b = b0 * mu
        R = gamma * b - c
        xi = f_ann * np.cos(2 * np.pi / tau) + f_ran * W * tau_cor/dt

        T = T + dt * (R * T + gamma * h - epsilon * (h + b * T)**3)
        h = h - dt * (r * h + alpha * b * T)
        Ts.append(T)
        hs.append(h)
    return np.array(Ts) * 7.5, np.array(hs) * 150

def run():
    return enso_oscillator(5000)


if __name__ == '__main__':
    Ts, hs = run()
