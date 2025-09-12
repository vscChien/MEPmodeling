function sim = MEPmodel_pheno_core(ref,p)

    y0=ref.y0';
    t0=ref.t0;
    muaps=ref.muaps;
    tmuap=ref.tmuap;
    %---------------------

    nIntensities = size(y0,1);
    simMEP      = zeros(nIntensities,length(t0));
    for i=1:nIntensities
        shape = p(1); 
        scale = p(2); 
        tshift = p(3);
        N = round(p(3+i)); 
        spike_times{i} = gen_spike_times_gamma(shape,scale,N) + (nIntensities-i)*tshift;   
        MEPcomps = zeros(N,length(t0));
        for n=1:N    
            MEPcomps(n,:) = interp1(tmuap+spike_times{i}(n)+t0(1),muaps(:,n),t0,'linear',0);
        end
        simMEP(i,:) = sum(MEPcomps,1);     

    end
    % align peaks of simMEP and MEP
    [~,i1]=max(y0(end,:)); % largest TMS intensity
    [~,i2]=max(simMEP(end,:));
    axonalDelay=max(0,t0(i1)-t0(i2)); % d>=0 ms
    simMEP = interp1(t0+axonalDelay,simMEP',t0,'linear',0)'; % shift d
    for i=1:nIntensities, spike_times{i} = spike_times{i} + axonalDelay; end

    %----------------------
    sim.simMEP=simMEP;
    sim.spike_times=spike_times;
    sim.MEPcomps=MEPcomps;
end

%==========================================================================
% Generate spike times 
function spike_times = gen_spike_times_gamma(shape,scale,N)

     spike_times = gaminv(linspace(0, 0.99, N), shape, 1/scale);

end