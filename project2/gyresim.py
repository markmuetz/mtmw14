'''Systems supports gravity waves, this is the fastest thing in the sim, this
will allow me to work out CFL conditions. This will be the phase speed of the
gravity waves.'''
from __future__ import division
import numpy as np
import pylab as plt
import scipy.interpolate as interp
#from numba.decorators import jit, autojit                             

def init_settings(**kwargs):
    settings = {}
    settings['f0'] = 1e-4 # s^-1
    settings['B'] = 1e-11 # m^-1 s^-1
    settings['g'] = 10 # m s^-2
    settings['gamma'] = 1e-6 # s^-1
    settings['rho'] = 1e3 # kg m^-3
    settings['H'] = 1e3 # m
    settings['tau0'] = 0.2 # N m^-2
    settings['L'] = 1e6 # m
    settings['plot'] = False
    return settings


def energy(eta, u, v, rho, H, g, dx, dy):
    E = 0.5 * (rho * (H * (u**2 + v**2) + g * eta**2) * dx * dy).sum()
    return E


def gyre_sim_sl2(t0, eta0, u0, v0, timelength, nt, X, Y, 
                 f0, B, g, gamma, rho, H, tau0, L, plot):
    # TODO: Not working. Looks like it's almost there, but Coriolis force not working?
    # Turns into clockwise gyre (as it should), but remains symmetric in x whereas
    # should develop into asymmetric with 0 vel moving W (left).
    eta = eta0
    u = u0
    v = v0
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
    c_x, c_y = np.sqrt(g * H) * dt/dx, np.sqrt(g * H) * dt/dy
    # TODO: how does squaring change dimd c?
    c2 = c_x**2 + c_y**2
    print('c2={}'.format(c2))
    if c2 > 0.25:
        print('Warning, running with c2={} (>0.25)'.format(c2))
    # TODO: How to calc dimd phi?
    # phi = f0 * dt
    #if phi > 1:
        #print('Warning, runnint with phi={} (>1)'.format(phi))
    tau_x = - tau0 * np.cos(np.pi * Y / L)
    tau_y = np.zeros_like(Y)

    v_u = np.zeros_like(u0)
    u_v = np.zeros_like(v0)

    u_grid = (u[:-1, :] + u[1:, :]) / 2
    v_grid = (v[:, :-1] + v[:, 1:]) / 2
    Es = [energy(eta, u_grid, v_grid, rho, H, g, dx, dy)]

    tau_x_u = (tau_x[:-1, :] + tau_x[1:, :]) / 2
    tau_y_v = (tau_y[:, :-1] + tau_y[:, 1:]) / 2

    u_prev = u.copy()
    v_prev = v.copy()
    for i, t in enumerate(times[1:]):
        # Work out u, v, at n+1/2.
        u0p5 = 1.5 * u - 0.5 * u_prev
        v0p5 = 1.5 * v - 0.5 * v_prev

        u_prev = u.copy()
        v_prev = v.copy()

        # Calc intermediate x, y.
        u_grid = (u[:-1, :] + u[1:, :]) / 2
        v_grid = (v[:, :-1] + v[:, 1:]) / 2
        Xi1 = X - u_grid * dt/2
        Yi1 = Y - v_grid * dt/2

        # Work out interp'd u, v at n+1/2
        ui0p5 = interp.RectBivariateSpline(x_u, y_u, u0p5).ev(Xi1, Yi1)
        vi0p5 = interp.RectBivariateSpline(x_v, y_v, v0p5).ev(Xi1, Yi1)

        # Calc x dep at n
        Xd = X - ui0p5 * dt
        Yd = Y - vi0p5 * dt

        u_tilde = interp.RectBivariateSpline(x_u, y_u, u).ev(Xd, Yd)
        v_tilde = interp.RectBivariateSpline(x_v, y_v, v).ev(Xd, Yd)
        eta_tilde = interp.RectBivariateSpline(x, y, eta).ev(Xd, Yd)
        
        dudx = (u[1:, :] - u[:-1, :]) / dx
        dvdy = (v[:, 1:] - v[:, :-1]) / dy
        dudx_tilde = interp.RectBivariateSpline(x, y, dudx).ev(Xd, Yd)
        dvdy_tilde = interp.RectBivariateSpline(x, y, dvdy).ev(Xd, Yd)

        eta = eta_tilde - H * dt * (dudx_tilde + dvdy_tilde)

        for j in range(2):
            if (i + j) % 2 == 0:
                v_u[1:-1, :] = (v_tilde[:-1, :] + v_tilde[1:, :]) / 2
                u_tilde_u = (u_tilde[:-1, :] + u_tilde[1:, :]) / 2
                deta_dx_u = (eta[1:, :] - eta[:-1, :]) / dx

                u[1:-1, :] = (+ u_tilde_u
                              + (f0 + B * Y_u[1:-1, :]) * dt * v_u[1:-1, :] 
                              - g * dt * deta_dx_u
                              - gamma * dt * u_tilde_u
                              + tau_x_u * dt / (rho * H))
            else:
                u_v[:, 1:-1] = (u_tilde[:, :-1] + u_tilde[:, 1:]) / 2
                v_tilde_v = (v_tilde[:, :-1] + v_tilde[:, 1:]) / 2
                deta_dy_v = (eta[:, 1:] - eta[:, :-1]) / dy

                v[:, 1:-1] = (+ v_tilde_v
                              - (f0 + B * Y_v[:, 1:-1]) * dt * u_v[:, 1:-1] 
                              - g * dt * deta_dy_v
                              - gamma * dt * v_tilde_v
                              + tau_y_v * dt / (rho * H))

            # Kinematic BCs: no normal flow.
            u[0, :] = 0
            u[-1, :] = 0
            v[:, 0] = 0
            v[:, -1] = 0

        u_grid = (u[:-1, :] + u[1:, :]) / 2
        v_grid = (v[:, :-1] + v[:, 1:]) / 2
        Es.append(energy(eta, u_grid, v_grid, rho, H, g, dx, dy))

        if i % 100 == 1:
            print('{}'.format(t))
            if plot:
                plt.figure(1)
                plt.clf()
                cs = plt.contour(X, Y, u_grid)
                plt.figure(2)
                plt.clf()
                cs = plt.contour(X, Y, v_grid)
                plt.figure(3)
                plt.clf()
                plt.quiver(X[::5, ::5], Y[::5, ::5], u_grid[::5, ::5], v_grid[::5, ::5])
                plt.pause(0.01)

    return times, eta, u, v, Es


def gyre_sim(t0, eta0, u0, v0, timelength, nt, X, Y, f0, B, g, gamma, rho, H, tau0, L, plot):
    eta = eta0
    u = u0
    v = v0
    times = np.linspace(t0, t0 + timelength, nt + 1)
    dt = times[1] - times[0]
    dx = X[1, 0] - X[0, 0]
    dy = Y[0, 1] - Y[0, 0]
    print('dx={}, dy={}, dt={}'.format(dx, dy, dt))
    c_x, c_y = np.sqrt(g * H) * dt/dx, np.sqrt(g * H) * dt/dy
    # TODO: how does squaring change dimd c?
    c2 = c_x**2 + c_y**2
    print('c2={}'.format(c2))
    if c2 > 0.25:
        print('Warning, running with c2={} (>0.25)'.format(c2))
    # TODO: How to calc dimd phi?
    # phi = f0 * dt
    #if phi > 1:
        #print('Warning, runnint with phi={} (>1)'.format(phi))
    tau_x = - tau0 * np.cos(np.pi * Y / L)
    tau_y = np.zeros_like(Y)

    v_u = np.zeros_like(u0)
    u_v = np.zeros_like(v0)

    u_grid = (u[:-1, :] + u[1:, :]) / 2
    v_grid = (v[:, :-1] + v[:, 1:]) / 2
    Es = [energy(eta, u_grid, v_grid, rho, H, g, dx, dy)]

    for i, t in enumerate(times[1:]):
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
        Es.append(energy(eta, u_grid, v_grid, rho, H, g, dx, dy))

        if i % 100 == 0:
            print('{}: energy={}'.format(t, Es[-1]))
            if plot:
                plt.figure(1)
                plt.clf()
                cs = plt.contour(X, Y, u_grid)
                plt.figure(2)
                plt.clf()
                cs = plt.contour(X, Y, v_grid)
                plt.figure(3)
                plt.clf()
                plt.quiver(X[::5, ::5], Y[::5, ::5], u_grid[::5, ::5], v_grid[::5, ::5])
                plt.pause(0.01)

    return times, eta, u, v, Es


def analytical_steady_state(eta0, X, Y, L, f0, B, g, gamma, rho, H, tau0, plot):
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

def calc_analytical(eta0, X, Y, **settings):
    eta_st, u_st, v_st = analytical_steady_state(eta0=eta0, X=X, Y=Y, **settings)
    if settings['plot']:
        plt.figure(21)
        plt.clf()
        plt.title('u')
        cs = plt.contour(X, Y, u_st)
        plt.clabel(cs, inline=1, fontsize=10)
        plt.figure(22)
        plt.clf()
        plt.title('v')
        cs = plt.contour(X, Y, v_st)
        plt.figure(23)
        plt.clabel(cs, inline=1, fontsize=10)
        plt.clf()
        plt.quiver(X[::5, ::5], Y[::5, ::5], u_st[::5, ::5], v_st[::5, ::5])

    return eta_st, u_st, v_st

def analyse_results(results):
    # After 1 day:
    X, Y = results[2]['grid'][0], results[2]['grid'][1]
    eta = results[2]['sim'][1]
    u, v = results[2]['sim'][2], results[2]['sim'][3]

    plt.figure(91)
    plt.clf()
    plt.title('u vs x southern edge')
    plt.plot(X[:, 0], (u[:-1, 0] + u[1:, 0]) / 2)

    plt.figure(92)
    plt.clf()
    plt.title('v vs y western edge')
    plt.plot(Y[0, :], (v[0, :-1] + v[0, 1:])/ 2)

    plt.figure(93)
    plt.clf()
    plt.title('$\eta$ vs x middle')
    plt.plot(X[:, 0], eta[:, eta.shape[1]/2])

    plt.figure(94)
    plt.clf()
    plt.title('Contour plot of $\eta$')
    cs = plt.contour(X, Y, eta)
    plt.clabel(cs, inline=1, fontsize=10)

    plt.figure(95)
    plt.clf()
    plt.title('Energy vs time')
    plt.plot(results[0]['sim'][0], results[0]['sim'][4], 
             label=results[0]['res'])
    plt.plot(results[1]['sim'][0], results[1]['sim'][4], 
             label=results[1]['res'])
    print('{}: energy_diff={}'.format(results[0]['res'], results[0]['energy_diff']))
    print('{}: energy_diff={}'.format(results[1]['res'], results[1]['energy_diff']))


def TaskABC():
    settings = init_settings()
    L = settings['L']
    settings['plot'] = True

    results = []
    for timelength, nx, nt in [(86400 * 50, 51, 8640 * 1.42 * 5),
                               (86400 * 50, 101, 8640 * 2 * 5 * 1.42),
                               (86400, 51, 864 * 1.42)]:
        ny = nx
        x = np.linspace(0, L, nx)
        y = np.linspace(0, L, ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        X, Y = np.meshgrid(x, y, indexing='ij')

        eta0 = np.zeros_like(X)
        u0 = np.zeros((X.shape[0] + 1, X.shape[1]))
        v0 = np.zeros((X.shape[0], X.shape[1] + 1))

        #gyre_sim_jit = autojit(gyre_sim)
        times, eta, u, v, Es = gyre_sim(0, eta0, u0, v0, timelength, nt, X, Y, **settings)

        eta_st, u_st, v_st = calc_analytical(eta[0, eta.shape[1]/2], X, Y, **settings)

        u_grid = (u[:-1, :] + u[1:, :]) / 2
        v_grid = (v[:, :-1] + v[:, 1:]) / 2

        up = u_grid - u_st
        vp = v_grid - v_st
        etap = eta - eta_st
        E = energy(etap, up, vp, settings['rho'], settings['H'], settings['g'], dx, dy)
        result = {}
        result['sim'] = (times, eta, u, v, Es)
        result['ana'] = (eta_st, u_st, v_st)
        result['res'] = 'energy_{}x{}'.format(nx, ny)
        result['grid'] = (X, Y)
        result['energy_diff'] = E

        results.append(result)

    analyse_results(results)


def TaskD():
    settings = init_settings()
    settings['plot'] = True
    L = settings['L']

    results = []
    for timelength, nx, nt in [(100000, 51, 10000)]:
        ny = nx
        x = np.linspace(0, L, nx)
        y = np.linspace(0, L, ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        X, Y = np.meshgrid(x, y, indexing='ij')

        eta0 = np.zeros_like(X)
        u0 = np.zeros((X.shape[0] + 1, X.shape[1]))
        v0 = np.zeros((X.shape[0], X.shape[1] + 1))

        times, h, u, v, Es = gyre_sim_sl2(0, eta0, u0, v0, timelength, nt, X, Y, **settings)

if __name__ == '__main__':
    plt.ion()
    #TaskABC()
    TaskD()
