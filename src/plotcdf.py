
import numpy as np
import matplotlib
backend="Qt5Aggx"
try:
    matplotlib.use(backend)
except:
    print("Backend %s is not available. Using default %s." % (backend, matplotlib.get_backend()))
          
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln2, = plt.plot([],[], '')
ln, = plt.plot([], [], 'ro')


def init():
    ax.set_ylim(0, 1)
    return ln, ln2,

def update(frame):
  
    try:
        data=np.loadtxt("../run/output.dat")
        X2 = np.sort(data)
        N=len(X2)
        ax.set_xlim(X2[0],X2[-1])
        F2 = np.array(range(N))/float(N)    
        ln.set_data(X2, F2)
        
        n, bins = np.histogram(data,  bins=100, density=1)
        pdfx = np.zeros(n.size)
        pdfy = np.zeros(n.size)
        for k in range(n.size):
            pdfx[k] = 0.5*(bins[k]+bins[k+1])
            pdfy[k] = n[k]/np.max(n)        
        ln2.set_data(pdfx,pdfy)
    finally:
        return ln, ln2,

ani = FuncAnimation(fig, update, 
                    init_func=init,  interval=1000, frames=10, repeat=True)
plt.show()