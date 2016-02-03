import pylab as plt

def clear_plots():
    plt.figure(1)
    plt.clf()
    plt.figure(2)
    plt.clf()

def plot_output(ts, Ts, hs, clear=True):
    fig = plt.figure(1)
    if clear:
        plt.clf()

    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.plot(ts, Ts, 'r--')
    ax2.plot(ts, hs, 'b-')

    ax1.set_ylabel('Temperature (K)', color='r')
    ax2.set_ylabel('thermocline depth (m)', color='b')
    ax1.set_xlabel('time (months)')

    plt.figure(2)
    if clear:
        plt.clf()
    plt.plot(Ts, hs, 'k-')
    plt.xlabel('Temperature (K)')
    plt.ylabel('thermocline depth (m)')

