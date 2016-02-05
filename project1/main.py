from __future__ import division
import sys
from collections import OrderedDict

import numpy as np

import ensosim as sim
from ensoplot import plot_output, clear_plots

if __name__ == '__main__':
    # Allows easy running of individual tasks by e.g. python main.py A B
    # defaults to all.
    if len(sys.argv) == 1:
        tasks = OrderedDict(A=True, B=True, C=True, D=True, E=True, F=True)
    else:
        tasks = OrderedDict(A=False, B=False, C=False,
                            D=False, E=False, F=False)
        for arg in sys.argv[1:]:
            tasks[arg] = True
    print(tasks)

    # Calculate some periods.
    omega_c = np.sqrt(3/32)
    tau_c = 2 * np.pi / omega_c
    one_period = tau_c * sim.TIME_SCALE
    five_periods = one_period * 5
    nt = 10000
    print('one period: {}'.format(one_period))

    if tasks['A']:
        settings = sim.init_settings()
        plot_output(*sim.enso_oscillator(one_period, nt, **settings))
        raw_input('Press enter to continue')

    if tasks['B']:
        settings = sim.init_settings(mu0=0.66)
        plot_output(*sim.enso_oscillator(five_periods, nt, **settings),
                    clear=False)

        settings = sim.init_settings(mu0=0.67)
        plot_output(*sim.enso_oscillator(five_periods, nt, **settings),
                    clear=False)
        raw_input('Press enter to continue')

    if tasks['C']:
        settings = sim.init_settings(epsilon=0.1)
        plot_output(*sim.enso_oscillator(five_periods, nt, **settings))
        raw_input('Press enter to continue')

        settings = sim.init_settings(epsilon=0.1, mu0=0.67)
        plot_output(*sim.enso_oscillator(five_periods, nt, **settings))
        raw_input('Press enter to continue')

    if tasks['D']:
        tau=12/sim.TIME_SCALE # N.B. non-dimensionalised
        settings = sim.init_settings(epsilon=0.1, mu0=0.75, mu_ann=0.2, tau=tau,
                                     mode='annual_cycle')
        plot_output(*sim.enso_oscillator(one_period * 4, nt, **settings))
        raw_input('Press enter to continue')


    if tasks['E']:
        tau=12/sim.TIME_SCALE # N.B. non-dimensionalised
        tau_cor=1/30/sim.TIME_SCALE # N.B. non-dimensionalised
        settings = sim.init_settings(epsilon=0.1, mu0=0.75, mu_ann=0.2, tau=tau,
                                     f_ann=0.02, f_ran=0.2, tau_cor=tau_cor,
                                     mode='annual_cycle')
        plot_output(*sim.enso_oscillator(one_period * 4, nt, **settings))
        raw_input('Press enter to continue')

    if tasks['F']:
        clear_plots()
        nperiods = 10
        # Set dt to one day by calculating nt correspondingly.
        nt = int(one_period * nperiods * 30)
        tau=12/sim.TIME_SCALE # N.B. non-dimensionalised
        tau_cor=1/30/sim.TIME_SCALE # N.B. non-dimensionalised
        settings = sim.init_settings(epsilon=0.1, mu0=0.75, mu_ann=0.2, tau=tau,
                                     f_ann=0.02, f_ran=0.2, tau_cor=tau_cor,
                                     mode='annual_cycle', debug=False)
        for h0 in np.linspace(-0.1, 0.1, 21):
            print(h0)
            settings['h0'] = h0
            plot_output(*sim.enso_oscillator(one_period * nperiods, nt, **settings), clear=False)
        raw_input('Press enter to continue')

