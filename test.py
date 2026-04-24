# Checked files
# gen_DIwave (qualitatively, just the plots)
# load_MEP (qualitatively, just the plots)
# load_muap (qualitatively, just the plots)
# generate_EP (qualitatively, just the plots)
# deconv_DIwave (qualitatively, just the plots)
# gen_kernels (qualitatively, just the plots)
# config_model_bio (qualitatively, just the plots)



# shape dv  (3,)
# shape dv2  (1, 3)
# shape tau  (3, 2)
# shape input vec  (3,)
# shape h  (3, 1)
# shape v  (3, 500)
# shape v2  (3, 500)


# import numpy as np
# tau = np.ones((3,2))
# input_vec = np.ones((3,))
# h = np.ones((3,1))
# v = np.ones((3,500))
# v2 = np.ones((3,500))
# tt=0

# print("h.T * input_vec ", np.shape(h.T * input_vec))
# print("tau[:, 0] + tau[:, 1] ", np.shape(tau[:, 0] + tau[:, 1]))
# print("tau[:, 0] * tau[:, 1] ", np.shape(tau[:, 0] * tau[:, 1]))
# print("v2[:, tt] * (tau[:, 0] + tau[:, 1]) ", np.shape(v2[:, tt] * (tau[:, 0] + tau[:, 1])))
# print("v2[:, tt] * (tau[:, 0] + tau[:, 1]) / (tau[:, 0] * tau[:, 1]) ", np.shape(v2[:, tt] * (tau[:, 0] + tau[:, 1]) / (tau[:, 0] * tau[:, 1])))
# print("- v[:, tt] / (tau[:, 0] * tau[:, 1] ", np.shape(- v[:, tt] / (tau[:, 0] * tau[:, 1])))
# print("(h.T * input_vec - v2[:, tt] * (tau[:, 0] + tau[:, 1]) / (tau[:, 0] * tau[:, 1]) - v[:, tt] / (tau[:, 0] * tau[:, 1])) ", np.shape((h.T * input_vec - v2[:, tt] * (tau[:, 0] + tau[:, 1]) / (tau[:, 0] * tau[:, 1]) - v[:, tt] / (tau[:, 0] * tau[:, 1]))))


# dv2 = (h.T * input_vec - v2[:, tt] * (tau[:, 0] + tau[:, 1]) / (tau[:, 0] * tau[:, 1]) - v[:, tt] / (tau[:, 0] * tau[:, 1]))




import os
import numpy as np
import matplotlib.pyplot as plt

# from scipy.io import loadmat, savemat
# from MEPmodel_bio import MEPmodel_bio
# from config_model_bio import config_model_bio

# root = os.getcwd()

# # model setting
# # Treat 'ref' as a dictionary with tuple keys for nested fields
# ref = config_model_bio(1, 1, [])

# result_file = os.path.join(root, ref['resultname'])
# print(result_file)

# # run GA or load existing results
# if os.path.exists(result_file) and not 0:
#     print(f"Use fitted result: \n{ref['resultname']}")
#     tmp = loadmat(result_file)
#     # Flattening to ensure it's a 1D array as expected in Python
#     p_post = tmp['p_post'].flatten()
# # show result
# plotOn = 1
# MEPmodel_bio(p_post, ref, plotOn)
#############################################################
# from config_model_bio import config_model_bio
# config_model_bio(1,1,0.5)
#############################################################
# from gen_kernels import gen_kernels
# tlength = 50 # ms
# dt = 0.1 # ms
# gen_kernels(dt,tlength)
#############################################################
from generate_EP import generate_EP
generate_EP(0.1, 1,2)
generate_EP(0.1, 1)
#############################################################
# from load_muap import load_muap
# load_muap(1)
#############################################################
# from load_MEP import load_MEP
# subj=1
# intensity_idx = np.arange(0, 10)
# load_MEP(subj, intensity_idx, [20, 50], 1)
#############################################################
# from gen_DIwave import gen_DIwave
# from deconv_DIwave import deconv_DIwave
# import numpy as np

# tlength = 50 # ms
# dt = 0.1 # ms
# t = np.arange(0, tlength, dt)
# intensities=[29., 32., 35., 38., 41., 44., 47., 50., 53., 56.]
# RMT=32
# DIwave0=np.zeros((len(intensities), len(t)))
# for i in range(len(intensities)):
#     DIwave0[i,:] = gen_DIwave(t, intensities[i] / RMT)
# DIwave = deconv_DIwave(t, DIwave0, ref={})

# plt.figure()
# plt.plot(t, DIwave.T)
# plt.grid(True)
# plt.xlabel("Time (ms)")
# plt.ylabel("Normalised mplitude")
# plt.xlim([t[0], t[-1]])

# plt.show()
