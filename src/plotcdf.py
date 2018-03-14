
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro', animated=False)

def init():
    ax.set_ylim(0, 1)
    return ln,

def update(frame):
  
    try:
        data=np.loadtxt("../run/output.dat")
        X2 = np.sort(data)
        print(X2[5])
        N=len(X2)
        ax.set_xlim(X2[0],X2[-1])
        F2 = np.array(range(N))/float(N)    
        ln.set_data(X2, F2)
    finally:
        return ln,

ani = FuncAnimation(fig, update, 
                    init_func=init, blit=True, interval=100, frames=10, repeat=True)
plt.show()