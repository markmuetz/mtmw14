'''Systems supports gravity waves, this is the fastest thing in the sim, this
will allow me to work out CFL conditions. This will be the phase speed of the
gravity waves.'''
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

def gyre_sim_semi_lag(t0, h0, u0, v0, timelength, nt, X, Y, f0, B, g, gamma, rho, H, tau0, L, plot):
    h = h0
    u = u0
    v = v0

    times = np.linspace(t0, t0 + timelength, nt + 1)
    dt = times[1] - times[0]

    u00 = np.zeros((X.shape[0] + 1, X.shape[1]))
    v00 = np.zeros((X.shape[0], X.shape[1] + 1))

    _, h1, u1, v1, Es =  gyre_sim(t0, h0, u00, v00, dt, 1, X, Y, f0, B, g, gamma, rho, H, tau0, L, plot)
    u1 = (u1[:-1, :] + u1[1:, :]) / 2
    v1 = (v1[:, :-1] + v1[:, 1:]) / 2

    dt = times[1] - times[0]
    dx = X[1, 0] - X[0, 0]
    dy = Y[0, 1] - Y[0, 0]
    x, y = X[:, 0], Y[0, :]
    print('dx={}, dy={}, dt={}'.format(dx, dy, dt))

    tau_x = - tau0 * np.cos(np.pi * Y / L)
    tau_y = np.zeros_like(Y)

    for step, t in enumerate(times[2:]):
        # Work out u, v, at n+1/2.
        u0p5 = 1.5 * u1 - 0.5 * u0
        v0p5 = 1.5 * v1 - 0.5 * v0

        # Calc intermediate x, y.
        xi1 = x - u1 * dt/2
        yi1 = y - v1 * dt/2

        # Work out interp'd u, v at n+1/2
        ui0p5 = interp.RectBivariateSpline(x, y, u0p5).ev(xi1, yi1)
        vi0p5 = interp.RectBivariateSpline(x, y, v0p5).ev(xi1, yi1)

        # Calc x dep at n
        xd1 = X - ui0p5 * dt
        yd1 = Y - vi0p5 * dt
        
        # Calc intermediate x, y (for n-1).
        xi2 = X - u0 * dt
        yi2 = Y - v0 * dt

        # Interp u at n to new u.
        ui0 = interp.RectBivariateSpline(x, y, u1).ev(xi2, yi2)
        vi0 = interp.RectBivariateSpline(x, y, v1).ev(xi2, yi2)

        # Calc x dep at n-1.
        xd0 = X - ui0 * 2 * dt
        yd0 = Y - vi0 * 2 * dt

        # Calc u- etc at n-1.
        u_minus = interp.RectBivariateSpline(x, y, u0).ev(xd0, yd0)
        v_minus = interp.RectBivariateSpline(x, y, v0).ev(xd0, yd0)
        h_minus = interp.RectBivariateSpline(x, y, h0).ev(xd0, yd0)

        # Calc u0 etc at n.
        u_0 = interp.RectBivariateSpline(x, y, u1).ev(xd1, yd1)
        v_0 = interp.RectBivariateSpline(x, y, v1).ev(xd1, yd1)
        h_0 = interp.RectBivariateSpline(x, y, h1).ev(xd1, yd1)

        # Calc du/dx, du/dy at n, interp x, y dep at n.
        dudx = (np.roll(u1, -1, axis=0) - np.roll(u1, 1, axis=0)) / (2 * dx)
        dudx0 = interp.RectBivariateSpline(x, y, dudx).ev(xd1, yd1)

        dvdy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dy)
        dvdy0 = interp.RectBivariateSpline(x, y, dvdy).ev(xd1, yd1)

        # Calc h derivs at n.
        dhdx = (np.roll(h1, -1, axis=0) - np.roll(h1, 1, axis=0)) / (2 * dx)
        dhdy = (np.roll(h1, -1, axis=1) - np.roll(h1, 1, axis=1)) / (2 * dy)
        # Interp x, y dep using h derivs.
        dhdx0 = interp.RectBivariateSpline(x, y, dhdx).ev(xd1, yd1)
        dhdy0 = interp.RectBivariateSpline(x, y, dhdy).ev(xd1, yd1)

        # Update h.
        h_plus = h_minus - h_0 * 2 * dt * (dudx0 + dvdy0)
        # Update u, v.
        u_plus = u_minus - 2 * dt * g * dhdx0
        v_plus = v_minus - 2 * dt * g * dhdy0

        # Cascade values down.
        h0, u0, v0 = h1, u1, v1
        h1, u1, v1 = h_plus, u_plus, v_plus

        if step % 1 == 0:
            print('{}'.format(t))
            if plot:
                #plt.figure(1)
                #plt.clf()
                #cs = plt.contour(X, Y, u_plus)
                #plt.figure(2)
                #plt.clf()
                #cs = plt.contour(X, Y, v_plus)
                plt.figure(3)
                plt.clf()
                plt.quiver(X[::5, ::5], Y[::5, ::5], u_plus[::5, ::5], v_plus[::5, ::5])
                plt.pause(0.01)
                raw_input()

    return times, h, u, v


def gyre_sim(t0, eta0, u0, v0, timelength, nt, x, y, f0, B, g, gamma, rho, H, tau0, L, plot):
    eta = eta0
    u = u0
    v = v0
    times = np.linspace(t0, t0 + timelength, nt + 1)
    dt = times[1] - times[0]
    dx = x[1, 0] - x[0, 0]
    dy = y[0, 1] - y[0, 0]
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
    tau_x = - tau0 * np.cos(np.pi * y / L)
    tau_y = np.zeros_like(y)

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
                              + (f0 + B * (y[:-1, :] + y[1:, :]) / 2) * dt * v_u[1:-1, :] 
                              - g * dt * (eta[1:, :] - eta[:-1, :]) / dx 
                              - gamma * dt * u[1:-1, :] 
                              + tau_x[:-1, :] * dt / (rho * H))
            else:
                u_v[:, 1:-1] = (u[:-1, :-1] + u[1:, :-1] +
                                u[:-1, 1:]  + u[1:, 1:]) / 4
                v[:, 1:-1] = (+ v[:, 1:-1] 
                              - (f0 + B * (y[:, :-1] + y[:, 1:]) / 2) * dt * u_v[:, 1:-1] 
                              - g * dt * (eta[:, 1:] - eta[:, :-1]) / dy 
                              - gamma * dt * v[:, 1:-1] 
                              + tau_y[:, :-1] * dt / (rho * H))
            # Kinematic BCs: no normal flow.
            u[0, :] = 0
            u[-1, :] = 0
            v[:, 0] = 0
            v[:, -1] = 0

        u_grid = (u[:-1, :] + u[1:, :]) / 2
        v_grid = (v[:, :-1] + v[:, 1:]) / 2
        Es.append(energy(eta, u_grid, v_grid, rho, H, g, dx, dy))

        if i % 1000 == 0:
            print('{}: energy={}'.format(t, Es[-1]))
            if plot:
                plt.figure(1)
                plt.clf()
                cs = plt.contour(x, y, u_grid)
                plt.figure(2)
                plt.clf()
                cs = plt.contour(x, y, v_grid)
                plt.figure(3)
                plt.clf()
                plt.quiver(x[::5, ::5], y[::5, ::5], u_grid[::5, ::5], v_grid[::5, ::5])
                plt.pause(0.01)

    return times, eta, u, v, Es


def analytical_steady_state(eta0, x, y, L, f0, B, g, gamma, rho, H, tau0, plot):
    epsilon = gamma / (L * B)
    a = (-1 - np.sqrt(1 + (2 * np.pi * epsilon)**2)) / (2 * epsilon)
    b = (-1 + np.sqrt(1 + (2 * np.pi * epsilon)**2)) / (2 * epsilon)

    def f1(x):
        return np.pi * (1 + ((np.exp(a) - 1) * np.exp(b * x) + 
                             (1 - np.exp(b)) * np.exp(a * x)) /
                             (np.exp(b) - np.exp(a)))

    def f2(x):
        return (((np.exp(a) - 1) * b * np.exp(b * x) + 
                (1 - np.exp(b)) * a * np.exp(a * x)) /
                     (np.exp(b) - np.exp(a)))
    u_st = (- tau0 / (np.pi * gamma * rho * H) *
            + f1(x / L) * np.cos(np.pi * y / L))
    v_st = (+ tau0 / (np.pi * gamma * rho * H) * 
            + f2(x / L) * np.sin(np.pi * y / L))
    eta_st = (+ eta0 + tau0 / (np.pi * gamma * rho * H) * f0 * L / g *
              + (gamma / (f0 * np.pi) * f2(x / L) * np.sin(np.pi * y / L)
                 + 1 / np.pi * f1(x / L) *
                 + (np.sin(np.pi * y / L) * (1 + B * y / f0) 
                 + B * L / (f0 * np.pi) * np.cos(np.pi * y / L))))

    return eta_st, u_st, v_st 

def calc_analytical(eta0, x, y, **settings):
    eta_st, u_st, v_st = analytical_steady_state(eta0=eta0, x=x, y=y, **settings)
    if settings['plot']:
        plt.figure(21)
        plt.clf()
        plt.title('u')
        cs = plt.contour(x, y, u_st)
        plt.clabel(cs, inline=1, fontsize=10)
        plt.figure(22)
        plt.clf()
        plt.title('v')
        cs = plt.contour(x, y, v_st)
        plt.figure(23)
        plt.clabel(cs, inline=1, fontsize=10)
        plt.clf()
        plt.quiver(x[::5, ::5], y[::5, ::5], u_st[::5, ::5], v_st[::5, ::5])

    return eta_st, u_st, v_st

def analyse_results(results):
    # After 1 day:
    x, y = results[2]['grid'][0], results[2]['grid'][1]
    eta = results[2]['sim'][1]
    u, v = results[2]['sim'][2], results[2]['sim'][3]

    plt.figure(91)
    plt.clf()
    plt.title('u vs x southern edge')
    plt.plot(x[:, 0], (u[:-1, 0] + u[1:, 0]) / 2)

    plt.figure(92)
    plt.clf()
    plt.title('v vs y western edge')
    plt.plot(y[0, :], (v[0, :-1] + v[0, 1:])/ 2)

    plt.figure(93)
    plt.clf()
    plt.title('$\eta$ vs x middle')
    plt.plot(x[:, 0], eta[:, eta.shape[1]/2])

    plt.figure(94)
    plt.clf()
    plt.title('Contour plot of $\eta$')
    cs = plt.contour(x, y, eta)
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

    results = []
    for timelength, nx, nt in [(86400 * 50, 51, 8640 * 1.42 * 5),
                               (86400 * 50, 101, 8640 * 2 * 5 * 1.42),
                               (86400, 51, 864 * 1.42)]:
        ny = nx
        x = np.linspace(0, L, nx)
        y = np.linspace(0, L, ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        x, y = np.meshgrid(x, y, indexing='ij')

        eta0 = np.zeros_like(x)
        u0 = np.zeros((x.shape[0] + 1, x.shape[1]))
        v0 = np.zeros((x.shape[0], x.shape[1] + 1))

        #gyre_sim_jit = autojit(gyre_sim)
        times, eta, u, v, Es = gyre_sim(0, eta0, u0, v0, timelength, nt, x, y, **settings)

        eta_st, u_st, v_st = calc_analytical(eta[0, eta.shape[1]/2], x, y, **settings)

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
        result['grid'] = (x, y)
        result['energy_diff'] = E

        results.append(result)

    analyse_results(results)


def TaskD():
    settings = init_settings()
    settings['plot'] = True
    L = settings['L']

    results = []
    for timelength, nx, nt in [(500, 51, 50)]:
        ny = nx
        x = np.linspace(0, L, nx)
        y = np.linspace(0, L, ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        x, y = np.meshgrid(x, y, indexing='ij')

        h0 = np.zeros_like(x)
        u0 = np.zeros_like(x)
        v0 = np.zeros_like(x)

        times, h, u, v = gyre_sim_semi_lag(0, h0, u0, v0, timelength, nt, x, y, **settings)

if __name__ == '__main__':
    plt.ion()
    #TaskABC()
    TaskD()
