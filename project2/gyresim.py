'''Implementation of Eulerian and semi-Lagrangian schemes for SWEs on Arakawa-C grid.
And supporting functions.'''
from __future__ import division, print_function
from collections import OrderedDict

import numpy as np
import scipy.interpolate as interp
from scipy.integrate import simps

from gyreresults import ResultsManager
from gyreanalysis import plot_timestep


def init_settings(**kwargs):
    settings = OrderedDict()
    settings['f0'] = 1e-4 # s^-1
    settings['B'] = 1e-11 # m^-1 s^-1
    settings['g'] = 10 # m s^-2
    settings['gamma'] = 1e-6 # s^-1
    settings['rho'] = 1e3 # kg m^-3
    settings['H'] = 1e3 # m
    settings['tau0'] = 0.2 # N m^-2
    settings['L'] = 1e6 # m
    settings['plot'] = False
    if kwargs:
        # Sets and prints any settings that have been overridden.
        print('setting: ', end='')
        for key, value in kwargs.iteritems():
            if key not in settings:
                raise ValueError('key {} not recognised'.format(key))
            print('{}={}, '.format(key, value), end='')
            settings[key] = value
        print('')
    return settings


def energy(eta, u, v, rho, H, g, x, y):
    'Calculate the energy from equation (12) in writup'
    Ez =  0.5 * (rho * (H * (u**2 + v**2) + g * eta**2))
    E = simps(simps(Ez, y), x) # 2d integration using Simpson's rule.
    return E


def calc_c(g, H, dt, dx, dy):
    'Calculate c (CFL criterion) from equation (10) in writup'
    c_x, c_y = np.sqrt(g * H) * dt/dx, np.sqrt(g * H) * dt/dy
    c = np.sqrt(c_x**2 + c_y**2)
    return c


def run_tasks(run_controls):
    'Run tasks defined by run_controls, using saved results if they already exist'
    rm = ResultsManager()
    settings = init_settings()
    settings['plot'] = True

    results = OrderedDict()
    for task, mode, timelength, dx, dt in run_controls:
        dy = dx
        key = '{}:{}-{}-{}x{}'.format(mode, timelength, dt, dx, dy)
        if mode == 'gyre_sim':
            sim = gyre_sim
        elif mode == 'gyre_sim_semi_lag':
            sim = gyre_sim_semi_lag

        if rm.exists(key):
            result = rm.get(key)
        else:
            X, Y, times, eta, u, v, Es = sim(0, timelength, dt, dx, dy, 
                                             plot_timestep=plot_timestep, **settings)

            eta_st, u_st, v_st = analytical_steady_state(eta[0, eta.shape[1]/2], X, Y, **settings)

            u_grid = (u[:-1, :] + u[1:, :]) / 2
            v_grid = (v[:, :-1] + v[:, 1:]) / 2

            up = u_grid - u_st
            vp = v_grid - v_st
            etap = eta - eta_st
            E_diff = energy(etap, up, vp, settings['rho'], settings['H'], settings['g'], X[:, 0], Y[0, :])
            result = {}
            result['sim'] = (times, eta, u, v, Es)
            result['ana'] = (eta_st, u_st, v_st)
            result['res'] = 'energy_{}x{}'.format(dx, dy)
            result['grid'] = (X, Y)
            result['energy_diff'] = E_diff

            rm.save(key, result)

        results[key] = result
    return results


def gyre_sim_semi_lag(t0, timelength, dt, dx, dy, 
                      f0, B, g, gamma, rho, H, tau0, L, 
                      plot, plot_timestep):
    'Solve SWEs using semi-Lagrangian method on Arakawa-C grid'
    # Set up fields.
    nx = L / dx + 1
    ny = nx

    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)

    X, Y = np.meshgrid(x, y, indexing='ij')

    eta0 = np.zeros_like(X)
    u0 = np.zeros((X.shape[0] + 1, X.shape[1]))
    v0 = np.zeros((X.shape[0], X.shape[1] + 1))

    eta = eta0
    u = u0
    v = v0
    nt = timelength / dt
    times = np.linspace(t0, t0 + timelength, nt + 1)
    dt = times[1] - times[0]
    dx = X[1, 0] - X[0, 0]
    dy = Y[0, 1] - Y[0, 0]

    x = X[:, 0]
    y = Y[0, :]
    x_u = np.linspace(X[0, 0] - dx/2, X[-1, 0] + dx/2, X.shape[0] + 1)
    y_u = np.linspace(Y[0, 0], Y[0, -1], Y.shape[1])
    X_u, Y_u = np.meshgrid(x_u, y_u, indexing='ij')

    x_v = np.linspace(X[0, 0], X[-1, 0], X.shape[0])
    y_v = np.linspace(Y[0, 0] - dy/2, Y[0, -1] + dy/2, Y.shape[1] + 1)
    X_v, Y_v = np.meshgrid(x_v, y_v, indexing='ij')

    print('dx={}, dy={}, dt={}'.format(dx, dy, dt))
    c = calc_c(g, H, dt, dx, dy)
    print('c={}'.format(c))
    if c > 1.0:
        print('Warning, running with c={} (>1.0)'.format(c))
    tau_x = - tau0 * np.cos(np.pi * Y_u / L)
    tau_y = np.zeros_like(Y_v)

    v_u = np.zeros_like(u0)
    u_v = np.zeros_like(v0)

    u_grid = (u[:-1, :] + u[1:, :]) / 2
    v_grid = (v[:, :-1] + v[:, 1:]) / 2
    Es = [energy(eta, u_grid, v_grid, rho, H, g, x, y)]

    u_prev = u.copy()
    v_prev = v.copy()
    for i, t in enumerate(times[1::2]):

        for j in range(2):
	    # Ensure 2nd order accuracy by calc'ing u, v at 1/2 timestep.
	    # Work out u, v, at n+1/2.
	    u0p5 = 1.5 * u - 0.5 * u_prev
	    v0p5 = 1.5 * v - 0.5 * v_prev

	    u_prev = u.copy()
	    v_prev = v.copy()

	    # Calc intermediate x, y.
	    u_grid = (u[:-1, :] + u[1:, :]) / 2
	    v_grid = (v[:, :-1] + v[:, 1:]) / 2
	    Xi = X - u_grid * dt/2
	    Yi = Y - v_grid * dt/2

	    v_u[1:-1, :] = (v[:-1, :-1] + v[1:, :-1] +
			    v[:-1, 1:] + v[1:, 1:]) / 4
	    u_v[:, 1:-1] = (u[:-1, :-1] + u[1:, :-1] +
			    u[:-1, 1:] + u[1:, 1:]) / 4
	    Xi_u = X_u - u * dt/2
	    Yi_u = Y_u - v_u * dt/2

	    Xi_v = X_v - u_v * dt/2
	    Yi_v = Y_v - v * dt/2

	    # Work out interp'd u, v at n+1/2
	    # Interp. handles values outside domain naturally
	    ui0p5 = interp.RectBivariateSpline(x_u, y_u, u0p5).ev(Xi, Yi)
	    vi0p5 = interp.RectBivariateSpline(x_v, y_v, v0p5).ev(Xi, Yi)

	    ui0p5_u = interp.RectBivariateSpline(x_u, y_u, u0p5).ev(Xi_u, Yi_u)
	    vi0p5_u = interp.RectBivariateSpline(x_v, y_v, v0p5).ev(Xi_u, Yi_u)

	    ui0p5_v = interp.RectBivariateSpline(x_u, y_u, u0p5).ev(Xi_v, Yi_v)
	    vi0p5_v = interp.RectBivariateSpline(x_v, y_v, v0p5).ev(Xi_v, Yi_v)

	    # Calc x dep at n
	    Xd = X - ui0p5 * dt
	    Yd = Y - vi0p5 * dt

	    Xd_u = X_u - ui0p5_u * dt
	    Yd_u = Y_u - vi0p5_u * dt

	    Xd_v = X_v - ui0p5_v * dt
	    Yd_v = Y_v - vi0p5_v * dt

	    # Calc u, v, eta at dep points.
	    u_tilde = interp.RectBivariateSpline(x_u, y_u, u).ev(Xd_u, Yd_u)
	    v_tilde = interp.RectBivariateSpline(x_v, y_v, v).ev(Xd_v, Yd_v)
	    eta_tilde = interp.RectBivariateSpline(x, y, eta).ev(Xd, Yd)
	    
	    dudx = (u[1:, :] - u[:-1, :]) / dx
	    dvdy = (v[:, 1:] - v[:, :-1]) / dy
	    dudx_tilde = interp.RectBivariateSpline(x, y, dudx).ev(Xd, Yd)
	    dvdy_tilde = interp.RectBivariateSpline(x, y, dvdy).ev(Xd, Yd)

	    eta = eta_tilde - H * dt * (dudx_tilde + dvdy_tilde)

	    # Update u, v, eta (switching u, v calc.).
            if (i + j) % 2 == 0:
                v_u[1:-1, :] = (v_tilde[:-1, :-1] + v_tilde[1:, :-1] +
				v_tilde[:-1, 1:] + v_tilde[1:, 1:]) / 4
                deta_dx_u = (eta[1:, :] - eta[:-1, :]) / dx

		u[1:-1, :] = (+ u_tilde[1:-1, :]
                              + (f0 + B * Y_u[1:-1, :]) * dt * v_u[1:-1, :] 
                              - g * dt * deta_dx_u
			      - gamma * dt * u_tilde[1:-1, :]
                              + tau_x[1:-1, :] * dt / (rho * H))
            else:
                u_v[:, 1:-1] = (u_tilde[:-1, :-1] + u_tilde[1:, :-1] +
				u_tilde[:-1, 1:] + u_tilde[1:, 1:]) / 4
                deta_dy_v = (eta[:, 1:] - eta[:, :-1]) / dy

		v[:, 1:-1] = (+ v_tilde[:, 1:-1]
                              - (f0 + B * Y_v[:, 1:-1]) * dt * u_v[:, 1:-1] 
                              - g * dt * deta_dy_v
			      - gamma * dt * v_tilde[:, 1:-1]
                              + tau_y[:, 1:-1] * dt / (rho * H))

            # Kinematic BCs: no normal flow.
            u[0, :] = 0
            u[-1, :] = 0
            v[:, 0] = 0
            v[:, -1] = 0

	    u_grid = (u[:-1, :] + u[1:, :]) / 2
	    v_grid = (v[:, :-1] + v[:, 1:]) / 2
	    Es.append(energy(eta, u_grid, v_grid, rho, H, g, x, y))

        if i % 100 == 0:
            print('{}: energy={}'.format(t / 86400, Es[-1]))
            if plot:
                plot_timestep(X, Y, u_grid, v_grid)

    return X, Y, times, eta, u, v, Es


def gyre_sim(t0, timelength, dt, dx, dy, 
             f0, B, g, gamma, rho, H, tau0, L, 
             plot, plot_timestep):
    'Solve SWEs using Eulerian method on Arakawa-C grid'
    # Set up fields.
    nx = L / dx + 1
    ny = nx

    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)

    X, Y = np.meshgrid(x, y, indexing='ij')

    eta0 = np.zeros_like(X)
    u0 = np.zeros((X.shape[0] + 1, X.shape[1]))
    v0 = np.zeros((X.shape[0], X.shape[1] + 1))

    eta = eta0
    u = u0
    v = v0
    nt = timelength / dt
    times = np.linspace(t0, t0 + timelength, nt + 1)
    x, y = X[:, 0], Y[0, :]
    dx = X[1, 0] - X[0, 0]
    dy = Y[0, 1] - Y[0, 0]
    print('dx={}, dy={}, dt={}'.format(dx, dy, dt))
    c_x, c_y = np.sqrt(g * H) * dt/dx, np.sqrt(g * H) * dt/dy
    c = np.sqrt(c_x**2 + c_y**2)
    print('c={}'.format(c))
    if c > 1.0:
        print('Warning, running with c={} (>1.0)'.format(c))
    tau_x = - tau0 * np.cos(np.pi * Y / L)
    tau_y = np.zeros_like(Y)

    v_u = np.zeros_like(u0)
    u_v = np.zeros_like(v0)

    u_grid = (u[:-1, :] + u[1:, :]) / 2
    v_grid = (v[:, :-1] + v[:, 1:]) / 2
    Es = [energy(eta, u_grid, v_grid, rho, H, g, x, y)]

    for i, t in enumerate(times[1::2]):
        eta = eta - H * dt * ((u[1:, :] - u[:-1, :]) / dx + 
                              (v[:, 1:] - v[:, :-1]) / dy)
        for j in range(2):
            if (i + j) % 2 == 0:
                v_u[1:-1, :] = (v[:-1, :-1] + v[1:, :-1] +
                                v[:-1, 1:]  + v[1:, 1:]) / 4
                u[1:-1, :] = (+ u[1:-1, :] 
                              + (f0 + B * (Y[:-1, :] + Y[1:, :]) / 2) * dt * v_u[1:-1, :] 
                              - g * dt * (eta[1:, :] - eta[:-1, :]) / dx 
                              - gamma * dt * u[1:-1, :] 
                              + (tau_x[:-1, :] + tau_x[1:, :]) * dt / ( 2 * rho * H))
            else:
                u_v[:, 1:-1] = (u[:-1, :-1] + u[1:, :-1] +
                                u[:-1, 1:]  + u[1:, 1:]) / 4
                v[:, 1:-1] = (+ v[:, 1:-1] 
                              - (f0 + B * (Y[:, :-1] + Y[:, 1:]) / 2) * dt * u_v[:, 1:-1] 
                              - g * dt * (eta[:, 1:] - eta[:, :-1]) / dy 
                              - gamma * dt * v[:, 1:-1] 
                              + (tau_y[:, :-1] + tau_y[:, 1:]) * dt / (2 * rho * H))

            # Kinematic BCs: no normal flow.
            u[0, :] = 0
            u[-1, :] = 0
            v[:, 0] = 0
            v[:, -1] = 0

	    u_grid = (u[:-1, :] + u[1:, :]) / 2
	    v_grid = (v[:, :-1] + v[:, 1:]) / 2
	    Es.append(energy(eta, u_grid, v_grid, rho, H, g, x, y))

        if i % 1000 == 0:
            print('{}: energy={}'.format(t / 86400, Es[-1]))
            if plot:
                plot_timestep(X, Y, u_grid, v_grid)

    return X, Y, times, eta, u, v, Es


def analytical_steady_state(eta0, X, Y, L, f0, B, g, gamma, rho, H, tau0, plot):
    'Calculate the steady state solution as in Musgrave 1985'
    epsilon = gamma / (L * B)
    a = (-1 - np.sqrt(1 + (2 * np.pi * epsilon)**2)) / (2 * epsilon)
    b = (-1 + np.sqrt(1 + (2 * np.pi * epsilon)**2)) / (2 * epsilon)

    def f1(X):
        return np.pi * (1 + ((np.exp(a) - 1) * np.exp(b * X) + 
                             (1 - np.exp(b)) * np.exp(a * X)) /
                             (np.exp(b) - np.exp(a)))

    def f2(X):
        return (((np.exp(a) - 1) * b * np.exp(b * X) + 
                (1 - np.exp(b)) * a * np.exp(a * X)) /
                     (np.exp(b) - np.exp(a)))
    u_st = (- tau0 / (np.pi * gamma * rho * H) *
            + f1(X / L) * np.cos(np.pi * Y / L))
    v_st = (+ tau0 / (np.pi * gamma * rho * H) * 
            + f2(X / L) * np.sin(np.pi * Y / L))
    eta_st = (+ eta0 + tau0 / (np.pi * gamma * rho * H) * f0 * L / g *
              + (gamma / (f0 * np.pi) * f2(X / L) * np.sin(np.pi * Y / L)
                 + 1 / np.pi * f1(X / L) *
                 + (np.sin(np.pi * Y / L) * (1 + B * Y / f0) 
                 + B * L / (f0 * np.pi) * np.cos(np.pi * Y / L))))

    return eta_st, u_st, v_st 
