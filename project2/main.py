from __future__ import division
from collections import OrderedDict

import pylab as plt

from gyresim import init_settings, gyre_sim, gyre_sim_semi_lag, energy
from gyreplot import analyse_one_day_result, analyse_diff_res, plot_analytical, plot_timestep
from gyreresults import ResultsManager


def tasks(run_controls):
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
            eta_st, u_st, v_st = plot_analytical(eta[0, eta.shape[1]/2], X, Y, **settings)

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


if __name__ == '__main__':
    plt.ion()
    run_controls = [('A', 'gyre_sim', 86400, 2e4, 139), 
                    ('C', 'gyre_sim', 86400 * 100, 2e4, 139),
                    #('C', 'gyre_sim', 86400 * 100, 1e4, 69), # Unstable!
		    # N.B. This is stable, it meets the more
		    # stringent stability requirements in Beckers and Deleersnijder.
                    ('C', 'gyre_sim', 86400 * 100, 1e4, 30), 
                    ('D', 'gyre_sim_semi_lag', 86400, 2e4, 139),
                    ('D', 'gyre_sim_semi_lag', 86400 * 50, 2e4, 139)]
    results = tasks(run_controls)
    #print('Analyse one day: {}'.format(results.keys()[0]))
    #analyse_one_day_result(results.values()[0])
    print('Analyse 2 res: {}, {}'.format(results.keys()[1], results.keys()[2]))
    analyse_diff_res(results.values()[1], results.values()[2])
    #print('Analyse 2 res: {}, {}'.format(results.keys()[1], results.keys()[4]))
    #analyse_diff_res(results.values()[1], results.values()[4])
