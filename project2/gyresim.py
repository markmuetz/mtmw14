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
    tau_x = tau0 * - np.cos(np.pi * y / L)
    tau_y = 0
    for i, t in enumerate(times):
        print(t)
        eta = eta - H * dt * ((np.roll(u, -1, axis=1) - u)/ dx + 
                              (np.roll(v, -1, axis=0) - v)/ dy)
        if i % 2 == 0:
            v_u = (v + np.roll(v, -1, axis=0) + np.roll(v, 1, axis=1) + 
                   np.roll(np.roll(v, -1, axis=0), 1, axis=1)) / 4
            u = (u + (f0 + B * y) * dt * v_u - 
                 g * dt / dx * (eta - np.roll(eta, 1, axis=1)) -
                 gamma * dt * u +
                 tau_x * dt / (rho * H))
            u_v = (u + np.roll(u, -1, axis=0) + np.roll(u, 1, axis=1) + 
                   np.roll(np.roll(u, -1, axis=0), 1, axis=1)) / 4
            v = (v - (f0 + B * y) * dt * u_v - 
                 g * dt / dy * (eta - np.roll(eta, 1, axis=0)) -
                 gamma * dt * v +
                 tau_y * dt / (rho * H))
        else:
            u_v = (u + np.roll(u, -1, axis=0) + np.roll(u, 1, axis=1) + 
                   np.roll(np.roll(u, -1, axis=0), 1, axis=1)) / 4
            v = (v - (f0 + B * y) * dt * u_v - 
                 g * dt / dy * (eta - np.roll(eta, 1, axis=0)) -
                 gamma * dt * v +
                 tau_y * dt / (rho * H))
            v_u = (v + np.roll(v, -1, axis=0) + np.roll(v, 1, axis=1) + 
                   np.roll(np.roll(v, -1, axis=0), 1, axis=1)) / 4
            u = (u + (f0 + B * y) * dt * v_u - 
                 g * dt / dx * (eta - np.roll(eta, 1, axis=1)) -
                 gamma * dt * u +
                 tau_x * dt / (rho * H))

        if i % 10 == 0:
            plt.clf()
            plt.title('u')
            cs = plt.contour(x, y, u)
            plt.clabel(cs, inline=1, fontsize=10)
            plt.pause(0.1)
            r = raw_input()
            if r == 'q':
                break

        #etas.append(eta)
        #us.append(u)
        #vs.append(v)
    return eta, u, v




def analytical_steady_state(x, y, f0, B, g, gamma, rho, H, tau0):
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
    nx, ny = 500, 500
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
        u0 = np.zeros_like(x)
        v0 = np.zeros_like(x)
        #gyre_sim_jit = autojit(gyre_sim)
        eta, u, v = gyre_sim(eta0, u0, v0, timelength, nt, x, y, **settings)

