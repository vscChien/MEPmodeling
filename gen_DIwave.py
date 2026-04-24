import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, x0, r, a):
    y = a / (1+ np.exp(r * (x0 - x)))
    return y

def gen_DIwave(t, intensity):
    t0 = 5
    T = 1.5
    width = 0.25

    DIwave = np.zeros(np.size(t))

    # D, I1, I2, I3, I4
    x0 = [1.36192637, 1.04127548, 1.16603639, 1.03733872, 1.45405986]
    r =  [18.50774852, 9.26210842, 5.91559859, 17.7805388, 425.51252596]
    a =  [0.34532065, 1.0, 0.80577286, 0.46054753, 0.27828232]

    for i in range(5):
        DIwave = DIwave + np.exp(-(t-t0 -i * T)**2 /2 /width**2) * sigmoid(intensity, x0[i], r[i], a[i])
    

    # ----- plotting -----
    if False:
        plt.figure()
        plt.plot(t, DIwave)
        plt.grid(True)
        plt.xlabel("Time (ms)")
        plt.ylabel("Normalised amplitude")
        plt.xlim([t[0], t[-1]])
        plt.title(f"DI wave (at {intensity} RMT)")
        plt.show()

    return DIwave