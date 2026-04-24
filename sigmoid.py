import numpy as np

def sigmoid(x, x0, r, a):
    return a / (1 + np.exp(r * (x0 - x)))