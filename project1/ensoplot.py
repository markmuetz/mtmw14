import pylab as plt

def clear_plots():
    plt.figure(1)
    plt.clf()
    plt.figure(2)
    plt.clf()

def plot_output(ts, Ts, hs, clear=True):
    plt.figure(1)
    if clear:
        plt.clf()
    plt.plot(ts, Ts)
    plt.plot(ts, hs)

    plt.figure(2)
    if clear:
        plt.clf()
    plt.plot(Ts, hs)
