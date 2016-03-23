from __future__ import division
from collections import OrderedDict

import pylab as plt

from gyresim import run_tasks
from gyreanalysis import (analyse_one_day_result, analyse_diff_res, 
			  analyse_diff_res2, arakawa_c_figure)


if __name__ == '__main__':
    # plt.ion()
    run_controls = [('A', 'gyre_sim', 86400, 2e4, 139), 
                    ('C', 'gyre_sim', 86400 * 100, 2e4, 139),
                    ('C', 'gyre_sim', 86400 * 200, 2e4, 49),
                    #('C', 'gyre_sim', 86400 * 100, 1e4, 69), # Unstable!
		    # N.B. This is stable, it meets the more
		    # stringent stability requirements in Beckers and Deleersnijder.
                    ('C', 'gyre_sim', 86400 * 200, 1e4, 49), 
                    ('D', 'gyre_sim', 86400, 2e4, 75), 
                    ('D', 'gyre_sim', 86400 * 100, 2e4, 75),
                    ('D', 'gyre_sim_semi_lag', 86400, 2e4, 75),
                    ('D', 'gyre_sim_semi_lag', 86400 * 100, 2e4, 75)]
    results = run_tasks(run_controls)

    arakawa_c_figure()
    # print('Analyse one day: {}'.format(results.keys()[0]))
    analyse_one_day_result(results.values()[0])
    # print('Analyse 2 res: {}, {}'.format(results.keys()[2], results.keys()[3]))
    analyse_diff_res(results.values()[2], results.values()[3])
    # print('Analyse 2 res: {}, {}'.format(results.keys()[0], results.keys()[4]))
    analyse_diff_res2(results.values()[4], results.values()[6], results.values()[5], results.values()[7])

