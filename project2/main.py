'''Main entry point for running all tasks'''
import pylab as plt

from gyresim import run_tasks
from gyreanalysis import (analyse_one_day_result, analyse_diff_res, 
			  analyse_diff_res2, arakawa_c_figure)


if __name__ == '__main__':
    # plt.ion()
    # Task, scheme, length, dx, dt
    run_controls = [('A', 'gyre_sim', 86400, 2e4, 139), 
                    ('C', 'gyre_sim', 86400 * 100, 2e4, 139),
                    ('C', 'gyre_sim', 86400 * 200, 2e4, 49),
                    #('C', 'gyre_sim', 86400 * 200, 1e4, 69), # Unstable! After 100 days.
		    # N.B. This is stable, it meets the more
		    # stringent stability requirements in Beckers and Deleersnijder.
                    ('C', 'gyre_sim', 86400 * 200, 1e4, 49), 
                    ('D', 'gyre_sim', 86400, 2e4, 75), 
                    ('D', 'gyre_sim', 86400 * 100, 2e4, 75),
                    ('D', 'gyre_sim_semi_lag', 86400, 2e4, 75),
                    ('D', 'gyre_sim_semi_lag', 86400 * 100, 2e4, 75)]
    results = run_tasks(run_controls)

    # Produce all figures.
    arakawa_c_figure()
    analyse_one_day_result(results.values()[0])
    analyse_diff_res(results.values()[2], results.values()[3])
    analyse_diff_res2(results.values()[4], results.values()[6], results.values()[5], results.values()[7])

