import numpy as np

def cal_R2(y, simMEP):
    return 1 - np.linalg.norm(y-simMEP) / np.linalg.norm(y-np.mean(y))