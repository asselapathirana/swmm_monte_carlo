import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats
from matplotlib.animation import FuncAnimation

backend = "Qt5Aggx"
try:
    matplotlib.use(backend)
except BaseException:
    print("Backend %s is not available. Using default %s." %
          (backend, matplotlib.get_backend()))


fig, ax = plt.subplots()
xdata, ydata = [], []
ax2 = ax.twinx()
ln2, = ax2.plot([], [], '')
ln, = ax.plot([], [], 'ro')
ln3 = []

dist_names = sys.argv[2:]
outputfile = sys.argv[1]

for dist_name in dist_names:
    dist = getattr(scipy.stats, dist_name)
    v, = ax2.plot([], [], label=dist_name)
    ln3.append(v)
plt.legend(loc='upper left')


def init():
    ax.set_ylim(0, 1)
    return ln, ln2,


def update(frame):

    try:
        data = np.loadtxt(outputfile)
        X2 = np.sort(data)
        N = len(X2)
        ax.set_xlim(X2[0], X2[-1])
        F2 = np.array(range(N)) / float(N)
        ln.set_data(X2, F2)

        n, bins = np.histogram(data, bins=100, density=1)
        pdfx = np.zeros(n.size)
        pdfy = np.zeros(n.size)
        for k in range(n.size):
            pdfx[k] = 0.5 * (bins[k] + bins[k + 1])
            pdfy[k] = n[k]
        ln2.set_data(pdfx, pdfy)
        ax2.set_ylim(0, np.max(pdfy))

        for i in range(len(ln3)):
            dist = getattr(scipy.stats, dist_names[i])
            param = dist.fit(data)
            pdf_fitted = dist.pdf(
                pdfx, *param[:-2], loc=param[-2], scale=param[-1])
            ln3[i].set_data(pdfx, pdf_fitted)

    finally:
        return ln3.extend((ln, ln2))


ani = FuncAnimation(fig, update,
                    init_func=init, interval=1000, frames=10, repeat=True)
plt.show()
