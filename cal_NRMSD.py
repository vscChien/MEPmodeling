import numpy as np

def cal_NRMSD(y, simMEP):
    return np.linalg.norm(y-simMEP) / np.sqrt(len(y)) / (np.max(y) - np.min(y))