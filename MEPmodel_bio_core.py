import numpy as np
from sigmoid import sigmoid
from scipy.interpolate import interp1d

def MEPmodel_bio_core(model):
    # Mapping dictionary keys from the model structure
    DIwave         = model['DIwaveConv']
    t              = model['t']
    muaps          = model['muaps']
    tmuap          = model['tmuap']
    R              = model['R']             # membrane resistance of MNs
    Wexc           = model['Wexc']
    RWinh          = model['RWinh']         # R * Winh
    dt             = model['dt']
    tauLIF         = model['tauLIF']
    Eexc           = model['Eexc']
    Einh           = model['Einh']
    V_rest         = model['V_rest']
    V_thr          = model['V_thr']
    T_ref          = model['T_ref']
    Tmu            = model['Tmu']
    withRC         = model['withRC']
    maxES          = model['maxES']
    fastAChRweight = model['fastAChRweight']
    rc             = model['rc']
    tau            = np.array(model[('kernel', 'tau')]) # Nested structure handled as dict
    h              = np.array(model[('kernel', 'h')])

    # ------------record for all TMS intensities--------------
    nIntensities = DIwave.shape[0]
    simMEP = np.zeros((nIntensities, len(t)))
    spike_times = np.full((100, maxES, nIntensities), np.nan)
    spike_times2 = [None] * nIntensities      # all spike times (list of objects)

    gexc_all = np.zeros((nIntensities, len(t)))
    ginh_all = np.zeros((nIntensities, len(t)))
    Rm_all   = np.zeros((nIntensities, len(t)))
    Vm_all   = np.zeros((nIntensities, len(t)))
    mRC_all  = np.zeros((nIntensities, len(t)))
    mMN_all  = np.zeros((nIntensities, len(t)))
    vRC_all  = np.zeros((nIntensities, len(t)))
    Iexc_all = np.zeros((nIntensities, len(t)))
    Iinh_all = np.zeros((nIntensities, len(t)))

    for i in range(nIntensities):
       
        # ----- Initialization -----
        gexc   = DIwave[i, :]                  # synaptic conductances on MNs
        ginh   = np.zeros(len(t))              # synaptic conductances on MNs
        Iexc   = np.zeros((100, len(t)))       # synaptic current to MNs
        Iinh   = np.zeros((100, len(t)))       # synaptic current to MNs
        Vm     = np.ones((100, len(t))) * V_rest 
        Vm_lag = np.ones((100, len(t))) * V_rest 
        sMN    = np.zeros((100, len(t)))       # MN spikes (0/1) 
        TR     = np.zeros(100)                 # Timer of refractory period
        mMN    = np.zeros(len(t))              # MN firing rate
        mRC    = np.zeros(len(t))              # RC firing rate   
        v      = np.zeros((3, len(t)))         # average postsynaptic potential
        v2     = np.zeros((3, len(t)))         # derivative helper
           
        # ----- simulation -----
        for tt in range(len(t) - 1):

            # -----MN rate-----
            mMN[tt] = np.mean(Wexc * sMN[:, tt]) # Wexc' in MATLAB is handled by element-wise if Wexc is 1D

            # -----RC rate-----
            if withRC:
                rescale = 100 
                # Assuming sigmoid is defined elsewhere as a custom function
                mRC[tt] = sigmoid(v[0, tt] * fastAChRweight + v[1, tt] * (1 - fastAChRweight), 
                                  rc['v_thr'] / rescale, rc['r'] * rescale, rc['fmax'])
            
            # -----PSPs (biexponential kernel)----- 
            input_vec = np.array([mMN[tt], mMN[tt], mRC[tt]])

            dv  = v2[:, tt]
            # print("shape tau ", np.shape(tau))
            # print("shape input vec ", np.shape(input_vec))
            # print("shape h ", np.shape(h))
            # print("shape v ", np.shape(v))
            # print("shape v2 ", np.shape(v2))
            # print("tt", tt)

            h = h.flatten()


            dv2 = (h * input_vec - v2[:, tt] * (tau[:, 0] + tau[:, 1]) / (tau[:, 0] * tau[:, 1]) - v[:, tt] / (tau[:, 0] * tau[:, 1]))

            v[:, tt + 1]  = v[:, tt]  + dv * dt
            v2[:, tt + 1] = v2[:, tt] + dv2 * dt

            # print("shape dv ", np.shape(dv))
            # print("shape dv2 ", np.shape(dv2))




            if withRC: 
                ginh[tt] = v[2, tt]
                Iexc[:, tt] = (Eexc - Vm[:, tt]) * gexc[tt]         
                Iinh[:, tt] = (Einh - Vm[:, tt]) * ginh[tt]
                Vm[:, tt + 1] = Vm[:, tt] + (dt / tauLIF) * (V_rest - Vm[:, tt] + 
                                R * Iexc[:, tt] + RWinh * Iinh[:, tt])
            else: 
                Iexc[:, tt] = (Eexc - Vm[:, tt]) * gexc[tt]
                Vm[:, tt + 1] = Vm[:, tt] + (dt / tauLIF) * (V_rest - Vm[:, tt] + 
                                R * Iexc[:, tt])

            # -----refractory period-----
            idx_tr = TR > 0
            Vm[idx_tr, tt + 1] = V_rest
            TR[idx_tr] -= 1
    
            # -----spike of 100 motor neurons----- 
            idx_spike = Vm[:, tt + 1] > V_thr
            Vm_lag[idx_spike, tt + 1] = Vm[idx_spike, tt + 1] 
            sMN[idx_spike, tt + 1] = 1 
            Vm[idx_spike, tt + 1] = V_rest 
            TR[idx_spike] = round(T_ref / dt)
    
        # ----- spike times -----  
        MEPcomps = np.zeros((100, len(t)))
        T_idx = Tmu / dt

        for n in range(100):
            # Find all spikes for neuron n
            idx = np.where(Vm_lag[n, :] > V_thr)[0]
            
            # -----collect effective spikes-----
            idxES = []
            refractory_idx = 0
            counter = 0
            
            while len(idx) > 0 and counter < maxES:
                if idx[0] > refractory_idx:
                    idxES.append(idx[0])
                    refractory_idx = idx[0] + T_idx
                    counter += 1
                idx = idx[1:]
            
            # ---------------------------------
            for k in range(len(idxES)):
                curr_idx = idxES[k]
                # Linear interpolation for precise spike time
                spike_t = (dt / (Vm_lag[n, curr_idx] - Vm[n, curr_idx - 1]) * (V_thr - Vm[n, curr_idx - 1]) + t[curr_idx - 1])
                
                spike_times[n, k, i] = spike_t
                
                # Cumulative interpolation of MUAP
                f_interp = interp1d((tmuap + spike_t).flatten(), (muaps[:, n]).flatten(), kind='linear', 
                                    bounds_error=False, fill_value=0)
                MEPcomps[n, :] += f_interp(t)
        
        simMEP[i, :] = np.sum(MEPcomps, axis=0)

        # ---------for record--------
        gexc_all[i, :] = R[0] * gexc  
        Rm_all[i, :]   = sMN[0, :] 
        Vm_all[i, :]   = Vm[0, :] 
        
        spike_rows, spike_cols = np.where(sMN == 1)
        spike_times2[i] = np.column_stack((spike_rows + 1, t[spike_cols])) 

        mMN_all[i, :]  = mMN
        vRC_all[i, :]  = v[0, :] * fastAChRweight + v[1, :] * (1 - fastAChRweight)
        Iexc_all[i, :] = Iexc[0, :]
        Iinh_all[i, :] = Iinh[0, :]

        if withRC:
            mRC_all[i, :]  = mRC
            ginh_all[i, :] = RWinh[0] * ginh

    # Construct output dictionary
    sim = {
        't': t,
        'simMEP': simMEP,
        'spike_times': spike_times,
        'gexc_all': gexc_all,
        'ginh_all': ginh_all,
        'Rm_all': Rm_all,
        'Vm_all': Vm_all,
        'mRC_all': mRC_all,
        'mMN_all': mMN_all,
        'vRC_all': vRC_all,
        'spike_times2': spike_times2,
        'Iexc_all': Iexc_all,
        'Iinh_all': Iinh_all
    }
    return sim