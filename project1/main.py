from __future__ import division
import random

import numpy as np
from scipy.linalg import solve
import pylab as plt

TIME_SCALE = 2
T_SCALE = 7.5
H_SCALE = 150

def fT(t, T, h, R, gamma, epsilon, b):
    return (R * T + gamma * h - epsilon * (h + b * T)**3)

def fh(t, T, h, r, alpha, b, xi):
    return -(r * h + alpha * b * T + alpha * xi)

def enso_oscillator(timelength, nt, method='rk4', T0=1.125, h0=0, mu=2/3, b0=2.5, gamma=0.75, c=1, r=0.25, alpha=0.125, epsilon=0, xi=0):
    timelength_non_dim = timelength / TIME_SCALE
    timesteps = np.linspace(0, timelength_non_dim, nt)
    dt = timesteps[1] - timesteps[0]

    T, h = T0/T_SCALE, h0/H_SCALE
    Ts, hs = [T], [h]

    b = b0 * mu
    R = gamma * b - c

    if method == 'bt':
        X = np.array([T, h])
        M = np.array([[1 - dt * R, -dt * gamma], [dt * alpha * b, 1 + dt * r]])

    for t in timesteps:
        print('t_non_dim={}, t_actual={}'.format(t, t * TIME_SCALE))
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
            k1 = fT(t, T, h, R, gamma, epsilon, b)
            l1 = fh(t, T, h, r, alpha, b, xi)

            k2 = fT(t + dt/2, T + dt/2 * k1, h + dt/2 * l1, R, gamma, epsilon, b)
            l2 = fh(t + dt/2, T + dt/2 * k1, h + dt/2 * l1, r, alpha, b, xi)

            k3 = fT(t + dt/2, T + dt/2 * k2, h + dt/2 * l2, R, gamma, epsilon, b)
            l3 = fh(t + dt/2, T + dt/2 * k2, h + dt/2 * l2, r, alpha, b, xi)

            k4 = fT(t + dt, T + dt * k3, h + dt * l3, R, gamma, epsilon, b)
            l4 = fh(t + dt, T + dt * k3, h + dt * l3, r, alpha, b, xi)

            T = T + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            h = h + dt/6 * (l1 + 2*l2 + 2*l3 + l4)

            Ts.append(T)
            hs.append(h)

    return timesteps * TIME_SCALE, np.array(Ts) * T_SCALE, np.array(hs) * H_SCALE
    #return np.array(Ts), np.array(hs)


def enso_oscillator_ann(timelength, nt, method='rk4', T0=1.125, h0=0, mu0=0.75, mu_ann=0.2, tau=12, b0=2.5, gamma=0.75, c=1, r=0.25, alpha=0.125, epsilon=0, xi=0):
    T, h = T0/7.5, h0/150
    timelength_non_dim = timelength / TIME_SCALE
    timesteps = np.linspace(0, timelength_non_dim, nt)
    dt = timesteps[1] - timesteps[0]

    T, h = T0/T_SCALE, h0/H_SCALE
    Ts, hs = [T], [h]
    tau = tau / TIME_SCALE

    for t in timesteps:
        print('t_non_dim={}, t_actual={}'.format(t, t * TIME_SCALE))
        mu = mu0 * (1 + mu_ann * np.cos(2 * np.pi * t / tau - 5 * np.pi / 6))
        b = b0 * mu
        R = gamma * b - c
        # xi = f_ann * np.cos(2 * np.pi / tau) + f_ran * W * tau_cor/dt

        # RK4.
        k1 = fT(t, T, h, R, gamma, epsilon, b)
        l1 = fh(t, T, h, r, alpha, b, xi)

        k2 = fT(t + dt/2, T + dt/2 * k1, h + dt/2 * l1, R, gamma, epsilon, b)
        l2 = fh(t + dt/2, T + dt/2 * k1, h + dt/2 * l1, r, alpha, b, xi)

        k3 = fT(t + dt/2, T + dt/2 * k2, h + dt/2 * l2, R, gamma, epsilon, b)
        l3 = fh(t + dt/2, T + dt/2 * k2, h + dt/2 * l2, r, alpha, b, xi)

        k4 = fT(t + dt, T + dt * k3, h + dt * l3, R, gamma, epsilon, b)
        l4 = fh(t + dt, T + dt * k3, h + dt * l3, r, alpha, b, xi)

        T = T + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        h = h + dt/6 * (l1 + 2*l2 + 2*l3 + l4)

        Ts.append(T)
        hs.append(h)

    return timesteps * TIME_SCALE, np.array(Ts) * T_SCALE, np.array(hs) * H_SCALE


def enso_oscillator_ann_with_forcing(timelength, nt, method='rk4', T0=1.125, h0=0, mu0=0.75, mu_ann=0.2, tau=12, b0=2.5, gamma=0.75, c=1, r=0.25, alpha=0.125, epsilon=0, xi=0, f_ann=0.02, f_ran=0.2, tau_cor=1/30):
    T, h = T0/7.5, h0/150
    timelength_non_dim = timelength / TIME_SCALE
    timesteps = np.linspace(0, timelength_non_dim, nt)
    dt = timesteps[1] - timesteps[0]

    T, h = T0/T_SCALE, h0/H_SCALE
    Ts, hs = [T], [h]
    tau = tau / TIME_SCALE

    W = random.uniform(-1, 1)
    t_last = 0

    for t in timesteps:
        if t - t_last > tau_cor:
            W = random.uniform(-1, 1)
            t_last = t

        xi = f_ann * np.cos(2 * np.pi / tau) + f_ran * W * tau_cor/dt
        print('t_non_dim={}, t_actual={}'.format(t, t * TIME_SCALE))
        mu = mu0 * (1 + mu_ann * np.cos(2 * np.pi * t / tau - 5 * np.pi / 6))
        b = b0 * mu
        R = gamma * b - c

        # RK4.
        k1 = fT(t, T, h, R, gamma, epsilon, b)
        l1 = fh(t, T, h, r, alpha, b, xi)

        k2 = fT(t + dt/2, T + dt/2 * k1, h + dt/2 * l1, R, gamma, epsilon, b)
        l2 = fh(t + dt/2, T + dt/2 * k1, h + dt/2 * l1, r, alpha, b, xi)

        k3 = fT(t + dt/2, T + dt/2 * k2, h + dt/2 * l2, R, gamma, epsilon, b)
        l3 = fh(t + dt/2, T + dt/2 * k2, h + dt/2 * l2, r, alpha, b, xi)

        k4 = fT(t + dt, T + dt * k3, h + dt * l3, R, gamma, epsilon, b)
        l4 = fh(t + dt, T + dt * k3, h + dt * l3, r, alpha, b, xi)

        T = T + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        h = h + dt/6 * (l1 + 2*l2 + 2*l3 + l4)

        Ts.append(T)
        hs.append(h)

    return timesteps * TIME_SCALE, np.array(Ts) * T_SCALE, np.array(hs) * H_SCALE


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


def ensemble(h0_min=-0.1, h0_max=0.1, n=100):
    for h0 in np.linspace(h0_min, h0_max, n):
        ts, Ts, hs = enso_oscillator_ann_with_forcing(41*4, h0=h0, nt=1000, mu0=0.75, mu_ann=0.2, tau=12, epsilon=0.1, method='rk4')
        plt.plot(ts, Ts[1:])


if __name__ == '__main__':
    Ts, hs = run()
