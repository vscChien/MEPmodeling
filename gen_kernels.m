% generate AMPA and NMDA kernels (in msec) for convolution
% For parameters and refs, check Roese's report
% https://drive.google.com/drive/folders/1sHNjkKEKYdNQDEcrcgHUjLxilNphZrS5
%
% (MN->RC)
%   AMPA: rise(~0.9ms), decay(~6ms)
%   NMDA: rise(~4.7ms), decay(~54.6ms)
%   fast AChR (α7): rise(~0.5ms), decay(~3.6ms)
%   slow AChR (α4β2): rise(~1.8ms), decay(~20.2ms)
%   [1] Moore, Niall J. (2015) https://discovery.ucl.ac.uk/id/eprint/1458605/1/njm_thesis_final_corrected.pdf  
% (RC->MN)
%   GlyR: rise(0.6~0.9 ms), decay (3~11 ms)
%   [2] Beato, Marco. (2008) https://www.jneurosci.org/content/jneuro/28/29/7412.full.pdf   
%   [3] Pitt, Samantha J., Lucia G. Sivilotti, and Marco Beato. (2008) https://www.jneurosci.org/content/jneuro/28/45/11454.full.pdf  

function [AMPA, NMDA] = gen_kernels(dt,tlength)

    t  = 0:dt:(tlength-dt);
    kernels=zeros(5,tlength/dt);


    tau = [1,   5; % rise, decay
           3,   50;
           0.5, 3.6;
           1.8, 20.2;
           1,   6];
    h   = [1.8692, 1.2731, 1.5977, 1.3908, 1.7175]*1e-2; % normalizing terms
    
    for i=1:5
        kernels(i,:) = h(i) * (exp(-t /tau(i,2))-exp(-t/tau(i,1)));
    end
    AMPA=kernels(1,:);
    NMDA=kernels(2,:);


    if 0
        labels={'In->MN(AMPA | 1ms, 5ms)',...
                'In->MN(NMDA | 3ms, 50ms)',...
                'MN->RC(AChR | 0.5ms, 3.6ms)',...
                'MN->RC(AChR | 1.8ms, 20.2ms)',...
                'RC->MN(GlyR | 1ms, 6ms)'};
        figure;
        plot(t,kernels','linewidth',2);
        ylim([0 1]);grid on;
        legend(labels);xlabel('ms')
        title('Biexponential synaptic kernels (analytical)')
    end

end
