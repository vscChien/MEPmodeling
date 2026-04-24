import numpy as np

def gen_kernels(dt, tlength):
    """
    Generate AMPA and NMDA kernels (in msec) for convolution.
    Parameters and refs are based on Roese's report.
    """
    
    # t = 0:dt:(tlength-dt)
    t = np.arange(0, tlength, dt)
    
    # kernels=zeros(5,tlength/dt)
    # Using t.size to ensure the dimensions match the time vector
    kernels = np.zeros((5, t.size))

    # tau = [rise, decay]
    tau = np.array([
        [1.0,   5.0],   # Index 0: AMPA (In->MN)
        [3.0,   50.0],  # Index 1: NMDA (In->MN)
        [0.5,   3.6],   # Index 2: fast AChR
        [1.8,   20.2],  # Index 3: slow AChR
        [1.0,   6.0]    # Index 4: GlyR (RC->MN)
    ])
    
    # h = normalizing terms
    h = np.array([1.8692, 1.2731, 1.5977, 1.3908, 1.7175]) * 1e-2
    
    # for i=1:5
    for i in range(5):
        # kernels(i,:) = h(i) * (exp(-t /tau(i,2))-exp(-t/tau(i,1)))
        kernels[i, :] = h[i] * (np.exp(-t / tau[i, 1]) - np.exp(-t / tau[i, 0]))
    
    # AMPA=kernels(1,:)
    AMPA = kernels[0, :]
    # NMDA=kernels(2,:)
    NMDA = kernels[1, :]

    # Optional plotting logic (equivalent to 'if 0' in MATLAB)
    plotOn = False
    if plotOn:
        import matplotlib.pyplot as plt
        labels = [
            'In->MN(AMPA | 1ms, 5ms)',
            'In->MN(NMDA | 3ms, 50ms)',
            'MN->RC(AChR | 0.5ms, 3.6ms)',
            'MN->RC(AChR | 1.8ms, 20.2ms)',
            'RC->MN(GlyR | 1ms, 6ms)'
        ]
        plt.figure()
        plt.plot(t, kernels.T, linewidth=2)
        plt.ylim([0, 0.015])
        plt.grid(True)
        plt.legend(labels)
        plt.xlabel('ms')
        plt.title('Biexponential synaptic kernels (analytical)')
        plt.show()

    return AMPA, NMDA