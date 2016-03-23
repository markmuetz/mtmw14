import pylab as plt
import numpy as np


def analyse_one_day_result(result):
    # After 1 day:
    X, Y = result['grid'][0], result['grid'][1]
    eta = result['sim'][1]
    u, v = result['sim'][2], result['sim'][3]

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    f.subplots_adjust(hspace=0.45, wspace=0.2)

    ax1.set_title('(a) u vs x southern edge')
    ax1.plot(X[:, 0] / 1e3, (u[:-1, 0] + u[1:, 0]) / 2)
    ax1.set_xlabel('x (km)')
    ax1.set_ylabel('u (ms$^{-1}$)')

    ax2.set_title('(b) v vs y western edge')
    ax2.plot(Y[0, :] / 1e3, (v[0, :-1] + v[0, 1:])/ 2)
    ax2.set_xlabel('y (km)')
    ax2.set_ylabel('v (ms$^{-1}$)')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    ax3.set_title('(c) $\eta$ vs x middle')
    ax3.plot(X[:, 0] / 1e3, eta[:, eta.shape[1]/2])
    ax3.set_xlabel('x (km)')
    ax3.set_ylabel('$\eta$ (m)')

    ax4.set_title('(d) Contour plot of $\eta$')
    cs = plt.contour(X / 1e3, Y / 1e3, eta)
    ax4.clabel(cs, inline=1, fontsize=10)
    ax4.set_xlabel('x (km)')
    ax4.set_ylabel('y (km)')
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")

    plt.savefig('writeup/figures/task_a.png')

def analyse_diff_res(res1, res2):
    plt.figure(95)
    plt.clf()
    #plt.title('Energy vs time')
    f, (ax1, ax2) = plt.subplots(2, 1)
    f.subplots_adjust(hspace=0.25)
    ax1.plot(res1['sim'][0] / 86400, np.array(res1['sim'][4][:-1]) / 1e15, 
             label=res1['res'])
    ax1.plot(res2['sim'][0] / 86400, np.array(res2['sim'][4][:-1]) / 1e15, 
             label=res2['res'])
    ax1.set_xlabel('time (days)')
    ax1.set_ylabel('energy (PJ)')
    ax1.legend(loc='lower right')

    ax2.plot(res1['sim'][0] / 86400, np.array(res1['sim'][4][:-1]) / 1e15, 
             label=res1['res'])
    ax2.plot(res2['sim'][0] / 86400, np.array(res2['sim'][4][:-1]) / 1e15, 
             label=res2['res'])
    ax2.set_xlim((60, 200))
    ax2.set_ylim((2.65, 2.85))
    ax2.set_xlabel('time (days)')
    ax2.set_ylabel('energy (PJ)')
    print('{}: energy_diff={}'.format(res1['res'], res1['energy_diff']))
    print('{}: energy_diff={}'.format(res2['res'], res2['energy_diff']))

    pd1 = (np.array(res1['sim'][4]).max() - res1['sim'][4][-1] ) / np.array(res1['sim'][4]).max() * 100
    pd2 = (np.array(res2['sim'][4]).max() - res2['sim'][4][-1] ) / np.array(res2['sim'][4]).max() * 100
    print('% diff max 1 {}'.format(pd1))
    print('% diff max 2 {}'.format(pd2))

    l1 = len(res1['sim'][4])
    pd3 = (res1['sim'][4][7 * l1 / 10] - res1['sim'][4][-1] ) / res1['sim'][4][-1] * 100
    pd4 = (res2['sim'][4][7 * l1 / 10] - res2['sim'][4][-1] ) / res2['sim'][4][-1] * 100
    #pd2 = (np.array(res2['sim'][4]).max() - res2['sim'][4][-1] ) / np.array(res2['sim'][4]).max() * 100
    print('% diff 70 day 1 {}'.format(pd3))
    print('% diff 70 day 2 {}'.format(pd4))

    plt.savefig('writeup/figures/task_b_energy.png')


def analyse_diff_res2(res1, res2, res3, res4):
    plt.figure(95)
    plt.clf()
    #plt.title('Energy vs time')
    f, (ax1, ax2) = plt.subplots(2, 1)
    f.subplots_adjust(hspace=0.25)
    ax1.plot(res3['sim'][0] / 86400, np.array(res3['sim'][4]) / 1e15, 
             label='Eulerian')
    ax1.plot(res4['sim'][0] / 86400, np.array(res4['sim'][4]) / 1e15, 
             label='semi-Lagrangian')
    ax1.set_xlabel('time (days)')
    ax1.set_ylabel('energy (PJ)')
    ax1.legend(loc='lower right')

    ax2.plot(res3['sim'][0] / 86400, np.array(res3['sim'][4]) / 1e15, 
             label='Eulerian')
    ax2.plot(res4['sim'][0] / 86400, np.array(res4['sim'][4]) / 1e15, 
             label='semi-Lagrangian')
    ax2.set_xlim((60, 100))
    ax2.set_ylim((2.65, 2.75))
    ax2.set_xlabel('time (days)')
    ax2.set_ylabel('energy (PJ)')

    print('{}: energy_diff={}'.format(res3['res'], res3['energy_diff']))
    print('{}: energy_diff={}'.format(res4['res'], res4['energy_diff']))

    pd1 = (np.array(res3['sim'][4]).max() - res3['sim'][4][-1] ) / np.array(res3['sim'][4]).max() * 100
    pd2 = (np.array(res4['sim'][4]).max() - res4['sim'][4][-1] ) / np.array(res4['sim'][4]).max() * 100
    print('% diff 1 {}'.format(pd1))
    print('% diff 2 {}'.format(pd2))

    plt.savefig('writeup/figures/task_d_energy.png')

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    f.subplots_adjust(hspace=0.25, wspace=0.25, right=0.8)

    X, Y = res1['grid']
    ax1.set_title('(a) $\eta$ after 1 day - Eul.')
    ax1.contourf(X/1e3, Y/1e3, res1['sim'][1], 100, vmin=-0.01, vmax=0.01)
    ax2.set_title('(b) $\eta$ after 1 day - S-L')
    cf2 = ax2.contourf(X/1e3, Y/1e3, res2['sim'][1], 100, vmin=-0.01, vmax=0.01)

    cbar_ax1 = f.add_axes([0.85, 0.55, 0.05, 0.4])
    f.colorbar(cf2, cax=cbar_ax1)

    ax3.set_title('(c) $\eta$ after 100 days - Eul.')
    ax3.contourf(X/1e3, Y/1e3, res3['sim'][1], 100, vmin=-0.15, vmax=0.2)
    ax4.set_title('(d) $\eta$ after 100 days - S-L')
    cf4 = ax4.contourf(X/1e3, Y/1e3, res4['sim'][1], 100, vmin=-0.15, vmax=0.2)

    ax3.set_xlabel('x (km)') 
    ax4.set_xlabel('x (km)') 
    ax1.set_ylabel('y (km)') 
    ax3.set_ylabel('y (km)') 

    cbar_ax2 = f.add_axes([0.85, 0.05, 0.05, 0.4])
    f.colorbar(cf4, cax=cbar_ax2)
    plt.savefig('writeup/figures/task_d_eta.png')

def plot_analytical(X, Y, eta_st, u_st, v_st):
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

def arakawa_c_figure(m=6, n=6, L=1e3):
    #fig = plt.figure()
    
    #ax = fig.add_axes([-1, -1, m + 1, n + 1])
    #ax.axis('off')
    plt.clf()
    dx = L/(m - 1)
    dy = L/(n - 1)
    plt.xlim((-1 * dx, m * dx))
    plt.ylim((-1 * dy, n * dy))
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')

    plt.plot(0, 0, 'kx', label='$\eta$')
    plt.plot(dx / 2, 0, 'ko', label='$u$')
    plt.plot(0, dy / 2, 'k^', label='$v$')
    plt.legend(numpoints=1, bbox_to_anchor=[1.05, 1.1])

    for i in range(m):
        for j in range(n):
            plt.plot(dx * i, dy * j, 'kx')
    for i in range(m + 1):
        for j in range(n):
            if i == 0 or i == m:
                fmt = 'ro'
            else:
                fmt = 'ko'
            plt.plot(dx * (i - 0.5), dy * j, fmt)
    for i in range(m):
        for j in range(n + 1):
            if j == 0 or j == m:
                fmt = 'r^'
            else:
                fmt = 'k^'
            plt.plot(dx * i, dy * (j - 0.5), fmt)
    plt.savefig('writeup/figures/arakawa_c_grid.png')
