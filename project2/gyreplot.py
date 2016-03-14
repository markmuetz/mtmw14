import pylab as plt

def analyse_one_day_result(result):
    # After 1 day:
    X, Y = result['grid'][0], result['grid'][1]
    eta = result['sim'][1]
    u, v = result['sim'][2], result['sim'][3]

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

def analyse_diff_res(res1, res2):
    plt.figure(95)
    plt.clf()
    plt.title('Energy vs time')
    plt.plot(res1['sim'][0], res1['sim'][4], 
             label=res1['res'])
    plt.plot(res2['sim'][0], res2['sim'][4], 
             label=res2['res'])
    #plt.legend()
    print('{}: energy_diff={}'.format(res1['res'], res1['energy_diff']))
    print('{}: energy_diff={}'.format(res2['res'], res2['energy_diff']))


def plot_analytical():
    settings = init_settings()
    L = settings['L']
    settings['plot'] = True
    nx = 20
    ny = nx
    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y, indexing='ij')

    eta0 = np.zeros_like(X)
    u0 = np.zeros((X.shape[0] + 1, X.shape[1]))
    v0 = np.zeros((X.shape[0], X.shape[1] + 1))

    eta_st, u_st, v_st = calc_analytical(0, X, Y, **settings)
def plot_analytical(eta0, X, Y, **settings):
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
        plt.figure(24)
        plt.clf()
        plt.contourf(X, Y, eta_st, 100)
        plt.colorbar()

    return eta_st, u_st, v_st

def plot_timestep(X, Y, u_grid, v_grid):
    try:
        plt.figure(1)
        plt.clf()
        cs = plt.contour(X, Y, u_grid)
        plt.figure(2)
        plt.clf()
        cs = plt.contour(X, Y, v_grid)
        plt.figure(3)
        plt.clf()
        plt.quiver(X[::2, ::2], Y[::2, ::2], u_grid[::2, ::2], v_grid[::2, ::2])
        plt.pause(0.01)
    except:
        pass


