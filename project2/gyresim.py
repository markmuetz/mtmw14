'''Systems supports gravity waves, this is the fastest thing in the sim, this
will allow me to work out CFL conditions. This will be the phase speed of the
gravity waves.'''
import numpy as np
import pylab as plt

from numba.decorators import jit, autojit                             


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
    return settings


def gyre_sim(eta0, u0, v0, timelength, nt, x, y, f0, B, g, gamma, rho, H, tau0, L):
    eta = eta0
    u = u0
    v = v0
    #etas, us, vs = [eta0], [u0], [v0]
    times = np.linspace(0, timelength, nt + 1)
    dt = times[1] - times[0]
    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]
    print('dx={}, dy={}, dt={}'.format(dx, dy, dt))
    c_x, c_y = dt/dx, dt/dy
    c = c_x**2 + c_y**2
    phi = f0 * dt
    print('c={}, , phi={}'.format(c, phi))
    if c > 0.5:
        print('Warning, runnint with c={} (>0.5)'.format(c))
    if phi > 1:
        print('Warning, runnint with phi={} (>1)'.format(phi))
    tau_x = tau0 * - np.cos(np.pi * y / L)
    tau_y = np.zeros_like(y)

    v_u = np.zeros_like(u0)
    u_v = np.zeros_like(v0)

    for i, t in enumerate(times):
        eta = eta - H * dt * ((u[1:, :] - u[:-1, :]) / dx + 
                              (v[:, 1:] - v[:, :-1]) / dy)
        for j in range(2):
            if (i + j) % 2 == 0:
                v_u[1:-1, :] = (v[:-1, :-1] + v[1:, :-1] +
                                v[-1:, 1:]  + v[1:, 1:]) / 4
                u[1:-1, :] = (+ u[1:-1, :] 
                              + (f0 + B * (y[:-1, :] + y[1:, :]) / 2) * dt * v_u[1:-1, :] 
                              - g * dt * (eta[1:, :] - eta[:-1, :]) / dx 
                              - gamma * dt * u[1:-1, :] 
                              + tau_x[:-1, :] * dt / (rho * H))
            else:
                u_v[:, 1:-1] = (u[:-1, :-1] + u[1:, :-1] +
                                u[-1:, 1:]  + u[1:, 1:]) / 4
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

        #eta[0, :] = 0
        #eta[-1, :] = 0
        #eta[:, 0] = 0
        #eta[:, -1] = 0

        if i % 100 == 0:
            print(t)
            plt.clf()
            plt.title('u')
            try:
                cs = plt.contour(x, y, u[:-1, :])
            except:
                pass
            plt.clabel(cs, inline=1, fontsize=10)
            plt.pause(0.1)
            #r = raw_input()
            r = 'c'
            if r == 'q':
                break

        #etas.append(eta)
        #us.append(u)
        #vs.append(v)
    return eta, u, v




def analytical_steady_state(x, y, L, f0, B, g, gamma, rho, H, tau0):
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
    eta0 = 0

    u_st = (-tau0 / (np.pi * gamma * rho * H) *
            f1(x / L) * np.cos(np.pi * y / L))
    v_st = (tau0 / (np.pi * gamma * rho * H) * 
            f2(x / L) * np.sin(np.pi * y / L))
    eta_st = (eta0 + tau0 / (np.pi * gamma * rho * H) * f0 * L / g *
              (gamma / (f0 * np.pi) * f2(x / L) * np.sin(np.pi * y / L) +
               1 / np.pi * f1(x / L) *
               (np.sin(np.pi * y / L) * (1 + B * y / f0) + 
                B * L / (f0 * np.pi) * np.cos(np.pi * y / L))))

    return u_st, v_st, eta_st

if __name__ == '__main__':
    plt.ion()
    settings = init_settings()
    nx, ny = 1000, 1000
    L = 1e6
    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    x, y = np.meshgrid(x, y)
    if False:
        u_st, v_st, eta_st = analytical_steady_state(x=x, y=y, **settings)
        plt.figure(1)
        plt.clf()
        plt.quiver(x, y, u_st, v_st)
        plt.figure(2)
        plt.clf()
        plt.title('u')
        cs = plt.contour(x, y, u_st)
        plt.clabel(cs, inline=1, fontsize=10)
        plt.figure(3)
        plt.clf()
        plt.title('v')
        cs = plt.contour(x, y, v_st)
        plt.clabel(cs, inline=1, fontsize=10)
    else:
        timelength = 86400
        nt = 2000
        eta0 = np.zeros_like(x)
        u0 = np.zeros((x.shape[0] + 1, x.shape[1]))
        v0 = np.zeros((x.shape[0], x.shape[1] + 1))
        gyre_sim_jit = autojit(gyre_sim)
        eta, u, v = gyre_sim(eta0, u0, v0, timelength, nt, x, y, **settings)

