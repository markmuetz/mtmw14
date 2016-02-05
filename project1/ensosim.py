from __future__ import division, print_function
import random

import numpy as np
from scipy.linalg import solve

TIME_SCALE = 2 # months
T_SCALE = 7.5 # K
H_SCALE = 150 # m

def init_settings(**kwargs):
    # Initial settings.
    # TODO: Comment and add units.
    settings = dict(
        # Base settings. Following are non-dimensional:
        T0=1.125/T_SCALE, # init. temperature
        h0=0/H_SCALE, # init. thermocline depth
        mu0=2/3, # relative coupling coefficient
        b0=2.5, # coupling parameter
        gamma=0.75,  # thermocline/SST gradient feedback
        c=1, # SST anomaly damping rate
        r=0.25, # damping of upper ocean heat content
        alpha=0.125, 
        epsilon=0, 
        # annual_cycle settings.
        mu_ann=0, 
        tau=0, 
        tau_cor=0, 
        xi=0, 
        f_ann=0, 
        f_ran=0, 
        # control.
        mode='base', 
        scheme='rk4',
        debug=False)

    if kwargs:
        print('setting: ', end='')
        for key, value in kwargs.iteritems():
            if key not in settings:
                raise ValueError('key {} not recognised'.format(key))
            print('{}={}, '.format(key, value), end='')
            settings[key] = value
        print('')
    return settings


def print_settings(settings, keys):
    for key in keys:
        print('{}: {}'.format(key, settings[key]))


def fT(T, h, R, gamma, epsilon, b, xi):
    return R * T + gamma * h - epsilon * (h + b * T)**3 + gamma * xi


def fh(T, h, r, alpha, b, xi):
    return -r * h - alpha * b * T - alpha * xi


def enso_oscillator(timelength, nt, 
                    T0, h0, mu0, b0, gamma, c, r, alpha, epsilon, # base
                    mu_ann, tau, tau_cor, f_ann, f_ran, xi, # annual_cycle
                    mode='base', scheme='rk4', debug=False):
    T, h = T0, h0

    timesteps = np.linspace(0, timelength, nt)
    dt = timesteps[1] - timesteps[0]
    Ts, hs = [T], [h]

    mu = mu0
    if mode == 'annual_cycle':
        W = random.uniform(-1, 1)
        t_last = 0

    if scheme == 'bt':
        # Backward time, implicit.
        # TODO won't work unless mode == 'base'.
        if mode != 'base':
            raise ValueError("Cannot use scheme='bt' with mode!='base'")
        b = b0 * mu
        R = gamma * b - c
        X = np.array([T, h])
        M = np.array([[1 - dt * R, -dt * gamma], [dt * alpha * b, 1 + dt * r]])

    for t in timesteps[1:]:
        if debug:
            print('t_non_dim={}, t_actual={}'.format(t, t * TIME_SCALE))

        if mode == 'annual_cycle':
            if t - t_last >= tau_cor:
                W = random.uniform(-1, 1)
                t_last = t
                if debug:
                    print('Setting W={}'.format(W))
            mu = mu0 * (1 + mu_ann * np.cos(2 * np.pi * t / tau - 5 * np.pi / 6))
            xi = f_ann * np.cos(2 * np.pi / tau) + f_ran * W * tau_cor/dt
        b = b0 * mu
        R = gamma * b - c

        if scheme == 'ft':
            # Forward time, explicit.
            T = T + dt * fT(T, h, R, gamma, epsilon, b, xi)
            h = h + dt * fh(T, h, r, alpha, b, xi)
            Ts.append(T)
            hs.append(h)
        elif scheme == 'bt':
            # Backward time, implicit.
            X = solve(M, X)
            Ts.append(X[0])
            hs.append(X[1])
        elif scheme == 'rk4':
            # RK4.
            k1 = fT(T, h, R, gamma, epsilon, b, xi)
            l1 = fh(T, h, r, alpha, b, xi)

            k2 = fT(T + dt/2 * k1, h + dt/2 * l1, R, gamma, epsilon, b, xi)
            l2 = fh(T + dt/2 * k1, h + dt/2 * l1, r, alpha, b, xi)

            k3 = fT(T + dt/2 * k2, h + dt/2 * l2, R, gamma, epsilon, b, xi)
            l3 = fh(T + dt/2 * k2, h + dt/2 * l2, r, alpha, b, xi)

            k4 = fT(T + dt * k3, h + dt * l3, R, gamma, epsilon, b, xi)
            l4 = fh(T + dt * k3, h + dt * l3, r, alpha, b, xi)

            T = T + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            h = h + dt/6 * (l1 + 2*l2 + 2*l3 + l4)

            Ts.append(T)
            hs.append(h)
        else:
            raise ValueError('Unrecognized scheme: {}'.format(scheme))

    return timesteps, np.array(Ts), np.array(hs)


def run_enso_ensemble(timelength, nt, tau, tau_cor, 
                      h0_min=-0.05, h0_max=0.05, nh=21):
    settings = init_settings(epsilon=0.1, mu0=0.75, mu_ann=0.2, tau=tau,
                             f_ann=0.02, f_ran=0.2, tau_cor=tau_cor,
                             mode='annual_cycle', debug=False)
    results = []
    results.append(enso_oscillator(timelength, nt, **settings))
    for h0 in np.linspace(h0_min, h0_max, nh):
        print('{}, '.format(h0), end='')
        settings['h0'] = h0
        results.append((enso_oscillator(timelength, nt, **settings)))
    return results
