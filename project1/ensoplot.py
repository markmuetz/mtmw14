'''Plotting tools'''
from itertools import cycle

import pylab as plt

import ensosim

# Picks a cyclical colour.
cycol = cycle('bgrcmkyw').next

def clear_plots():
    plt.figure(1)
    plt.clf()
    plt.figure(2)
    plt.clf()

def plot_output(ts, Ts, hs, ylim=None, colour=None, clear=True):
    '''Takes non-dimensional values of time, T and h and plots them.
    Can be used to overlay plots if clear=False'''
    # Re-dimensionalise:
    ts = ts * ensosim.TIME_SCALE
    Ts = Ts * ensosim.T_SCALE
    hs = hs * ensosim.H_SCALE
    if not colour:
	colour = cycol()

    fig = plt.figure(1)
    if clear:
        plt.clf()

    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.plot(ts, Ts, '{}--'.format(colour))
    ax2.plot(ts, hs, '{}-'.format(colour))
    if not ylim:
	ax2.set_ylim((-20, 20))
    else:
	ax2.set_ylim(ylim)

    if clear:
	ax1.set_ylabel('Temperature (K)')
	ax2.set_ylabel('thermocline depth (m)')
	ax1.set_xlabel('time (months)')

    plt.figure(2)
    if clear:
        plt.clf()
    plt.plot(Ts, hs, '{}-'.format(colour))
    if clear:
	plt.xlabel('Temperature (K)')
	plt.ylabel('thermocline depth (m)')


def plot_ensemble_output(ensemble_results):
    '''Loop over all ensemble results and plot each one'''
    plot_output(*ensemble_results[0], ylim=(-80, 80))
    for result in ensemble_results[1:]:
        plot_output(*result, ylim=(-80, 80), clear=False)
