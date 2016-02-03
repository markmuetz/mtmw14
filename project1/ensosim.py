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
        # base settings.
        T0=1.125, # init. temperature, K
        h0=0, # init. thermocline depth, m
        # Following are non-dimensional:
        mu0=2/3, # relative coupling coefficient
        b0=2.5, # coupling parameter
        gamma=0.75,  # thermocline/SST gradient feedback
        c=1, # SST anomaly damping rate
        r=0.25, # damping of upper ocean heat content
        alpha=0.125, # 
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
        method='rk4',
        debug=False)
    print('setting: ', end='')
    for key, value in kwargs.iteritems():
	if key not in settings:
	    raise ValueError('key {} not recognised'.format(key))
	print('{}={}, '.format(key, value), end='')
	settings[key] = value
    print('')
    return settings


def fT(t, T, h, R, gamma, epsilon, b, xi):
    return R * T + gamma * h - epsilon * (h + b * T)**3 + gamma * xi


def fh(t, T, h, r, alpha, b, xi):
    return -r * h - alpha * b * T - alpha * xi


def enso_oscillator(timelength, nt, 
                    T0, h0, mu0, b0, gamma, c, r, alpha, epsilon, # base
                    mu_ann, tau, tau_cor, f_ann, f_ran, xi, # annual_cycle
                    mode='base', method='rk4', debug=False):
    # Non-dimensionalise time, T and h:
    timelength_non_dim = timelength / TIME_SCALE
    T, h = T0/T_SCALE, h0/H_SCALE

    timesteps = np.linspace(0, timelength_non_dim, nt)
    dt = timesteps[1] - timesteps[0]
    Ts, hs = [T], [h]

    mu = mu0
    if mode == 'annual_cycle':
        W = random.uniform(-1, 1)
        t_last = 0
    else:
        b = b0 * mu
        R = gamma * b - c

    if method == 'bt':
        # Backward time, implicit.
        # TODO won't work unless mode == 'base'.
        if mode != 'base':
            raise ValueError("Cannot use method='bt' with mode!='base'")
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
            b = b0 * mu
            R = gamma * b - c
            xi = f_ann * np.cos(2 * np.pi / tau) + f_ran * W * tau_cor/dt

        if method == 'ft':
            # Forward time, explicit.
            T = T + dt * fT(t, T, h, R, gamma, epsilon, b, xi)
            h = h - dt * fh(t, T, h, r, alpha, b, xi)
            Ts.append(T)
            hs.append(h)
        elif method == 'bt':
            # Backward time, implicit.
            X = solve(M, X)
            Ts.append(X[0])
            hs.append(X[1])
        elif method == 'rk4':
            # RK4.
            k1 = fT(t, T, h, R, gamma, epsilon, b, xi)
            l1 = fh(t, T, h, r, alpha, b, xi)

            k2 = fT(t + dt/2, T + dt/2 * k1, h + dt/2 * l1, R, gamma, epsilon, b, xi)
            l2 = fh(t + dt/2, T + dt/2 * k1, h + dt/2 * l1, r, alpha, b, xi)

            k3 = fT(t + dt/2, T + dt/2 * k2, h + dt/2 * l2, R, gamma, epsilon, b, xi)
            l3 = fh(t + dt/2, T + dt/2 * k2, h + dt/2 * l2, r, alpha, b, xi)

            k4 = fT(t + dt, T + dt * k3, h + dt * l3, R, gamma, epsilon, b, xi)
            l4 = fh(t + dt, T + dt * k3, h + dt * l3, r, alpha, b, xi)

            T = T + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            h = h + dt/6 * (l1 + 2*l2 + 2*l3 + l4)

            Ts.append(T)
            hs.append(h)

    # Redimensionalise before returning.
    return timesteps * TIME_SCALE, np.array(Ts) * T_SCALE, np.array(hs) * H_SCALE


def ensemble(settings, h0_min=-0.1, h0_max=0.1, n=100):
    for h0 in np.linspace(h0_min, h0_max, n):
        print(h0)
	settings['h0'] = h0
        ts, Ts, hs = enso_oscillator(41*4, nt=1000, **settings)
        # plt.plot(ts, Ts[1:])
