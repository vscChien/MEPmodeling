function sim = MEPmodel_bio_core(model)

    DIwave         = model.DIwaveConv;
    t              = model.t;
    muaps          = model.muaps;
    tmuap          = model.tmuap;
    R              = model.R;      % membrane resistance of MNs
    Wexc           = model.Wexc;
    RWinh          = model.RWinh;   % R.*Winh
    dt             = model.dt;
    tauLIF         = model.tauLIF;
    Eexc           = model.Eexc;
    Einh           = model.Einh;
    V_rest         = model.V_rest;
    V_thr          = model.V_thr;
    T_ref          = model.T_ref;
    Tmu            = model.Tmu;
    withRC         = model.withRC;
    maxES          = model.maxES;
    fastAChRweight = model.fastAChRweight;
    rc             = model.rc;
    tau            = model.kernel.tau;
    h              = model.kernel.h;

    %------------record for all TMS intensities--------------
    nIntensities=size(DIwave,1);
    simMEP=zeros(nIntensities,length(t));
    spike_times=nan(100,maxES,nIntensities); 
    spike_times2=cell(nIntensities,1);         % all spike times

    gexc_all   = zeros(nIntensities,length(t)); % (for record) 
    ginh_all  = zeros(nIntensities,length(t)); % (for record) 
    Rm_all  = zeros(nIntensities,length(t)); % (for record) 
    Vm_all  = zeros(nIntensities,length(t)); % (for record) 
    mRC_all = zeros(nIntensities,length(t)); % RC rate (for record) 
    mMN_all = zeros(nIntensities,length(t)); % MN rate (for record) 
    vRC_all = zeros(nIntensities,length(t)); % EPSP (MN->RC) (for record)
    Iexc_all = zeros(nIntensities,length(t)); % EPSP (MN->RC) (for record)
    Iinh_all = zeros(nIntensities,length(t)); % EPSP (MN->RC) (for record)

    for i=1:nIntensities
       
        %----- Initialization -----
        gexc   = DIwave(i,:);                  % synaptic conductances on MNs
        ginh   = zeros(1,length(t));           % synaptic conductances on MNs
        Iexc   = zeros(100,length(t));         % synaptic current to MNs
        Iinh   = zeros(100,length(t));         % synaptic current to MNs
        Vm     = ones(100,length(t)) * V_rest; % MN membrane potentials
        Vm_lag = ones(100,length(t)) * V_rest; % MN membrane potentials (a copy for interpolation)
        sMN    = zeros(100,length(t));         % MN spikes (0/1) 
        TR     = zeros(100,1);                 % Timer of refractory period
        mMN    = zeros(1,length(t));           % MN firing rate
        mRC    = zeros(1,length(t));           % RC firing rate   
        v      = zeros(3,length(t));           % average postsynaptic potential
        v2     = zeros(3,length(t));           %   EPSPs (MN->RC): fast AchR (rise 0.5 ms, decay 3.6 ms) 
                                               %                   slow AchR (rise 1.8 ms, decay 20.2 ms)  
                                               %   IPSP (RC->MN):  GlyR (rise 1 ms, decay 6 ms) 
           
        %----- simulation -----
        for tt=1:length(t)-1

            %-----MN rate-----
            mMN(tt) = mean(Wexc'.*sMN(:,tt)); % mean MN firing rate 

            %-----RC rate-----
            if withRC
                rescale = 100; %N=100
                mRC(tt) = sigmoid(v(1,tt)*fastAChRweight + v(2,tt)*(1-fastAChRweight), ...
                                  rc.v_thr/rescale, rc.r*rescale, rc.fmax);
            end
            %-----PSPs (biexponential kernel)----- 
            in = [mMN(tt); % (MN->RC, fast AchR)
                  mMN(tt); % (MN->RC, slow AchR)
                  mRC(tt)];% (RC->MN, GlyR)

            dv  = v2(:,tt);
            dv2 = h.*in - ...
                  v2(:,tt).*(tau(:,1)+tau(:,2))./tau(:,1)./tau(:,2) - ...
                  v(:,tt)./tau(:,1)./tau(:,2); 

            v(:,tt+1)  = v(:,tt)  + dv*dt;
            v2(:,tt+1) = v2(:,tt) + dv2*dt;

            
            if withRC 
                ginh(tt)  =v(3,tt);
                Iexc(:,tt)=(Eexc-Vm(:,tt))*gexc(tt);         
                Iinh(:,tt)=(Einh-Vm(:,tt))*ginh(tt);
                Vm(:,tt+1) = Vm(:,tt) + ...
                             dt / tauLIF * (V_rest - Vm(:,tt) + ...
                                            R'.*Iexc(:,tt) + ... % excitatory input (from DI-waves)
                                            RWinh'.*Iinh(:,tt));  % inhibitory input (from RC)
            else % no RC 
                Iexc(:,tt)=(Eexc-Vm(:,tt))*gexc(tt);
                Vm(:,tt+1) = Vm(:,tt) + ...
                             dt / tauLIF * (V_rest - Vm(:,tt) + ...
                                            R'.*Iexc(:,tt)); % excitatory input (from DI-waves)
            end
            %-----refractory period-----
            idx = TR>0;
            Vm(idx,tt+1) = V_rest; % rest
            TR(idx,1)=TR(idx,1)-1;
    
            %-----spike of 100 motor neurons----- 
            idx = Vm(:,tt+1) > V_thr;
            Vm_lag(idx,tt+1) = Vm(idx,tt+1); % just to record the Vm right before spiking (for time interpolation)
            sMN(idx,tt+1) = 1; % spike
            Vm(idx,tt+1) = V_rest; % rest
            TR(idx,1)=round(T_ref/dt);
    
        end
        
        %----- spike times -----  
        MEPcomps = zeros(100,length(t));
        T=Tmu/dt;

        for n=1:100
            [~,idx]=find(Vm_lag(n,:)> V_thr);  % all spikes
            %-----collect effective spikes-----
            idxES=nan(1,maxES); 
            refractory_idx=0;
            counter=1;
            while ~isempty(idx) && counter<=maxES 
                if idx(1)>refractory_idx
                  idxES(counter)=idx(1);
                  refractory_idx=idx(1)+T;
                  counter=counter+1;
                end
                idx(1)=[];
            end
            idxES(isnan(idxES))=[];
            %---------------------------------
            for k=1:length(idxES) 
                spike_times(n,k,i) = dt/(Vm_lag(n,idxES(k))-Vm(n,idxES(k)-1)) *(V_thr-Vm(n, idxES(k)-1))+ t(idxES(k)-1);
                MEPcomps(n,:) = MEPcomps(n,:) + interp1(tmuap+spike_times(n,k,i),muaps(:,n),t,'linear',0); %cumulative
            end          
        end
        simMEP(i,:) = sum(MEPcomps,1); % [10 x  500]

        %---------for record--------
        gexc_all(i,:)=R(1)*gexc;  % conductance (Cortex->MN1)    
        Rm_all(i,:)=sMN(1,:); % spike of the first MN
        Vm_all(i,:)=Vm(1,:); % spike of the first MN
        [tmp1,tmp2]=find(sMN);
        spike_times2{i}=[tmp1,t(tmp2)']; 


        mMN_all(i,:)=mMN(:); % MN rate
        vRC_all(i,:)=v(1,:)*fastAChRweight+v(2,:)*(1-fastAChRweight); % EPSP (MN->RC)
        Iexc_all(i,:)=Iexc(1,:); % excitatoty input to MN1 (from DIwaves)
        Iinh_all(i,:)=Iinh(1,:); % inhibitory input to MN1 (from RC)

        if withRC
            mRC_all(i,:)=mRC(:); % RC rate
            ginh_all(i,:)=RWinh(1)*ginh;%v(3,:); % conductance (RC->MN1)
            
        end
    end

    sim.t=t;
    sim.simMEP=simMEP;
    sim.spike_times=spike_times;
    sim.gexc_all=gexc_all;
    sim.ginh_all=ginh_all;
    sim.Rm_all=Rm_all;
    sim.Vm_all=Vm_all;
    sim.mRC_all=mRC_all;
    sim.mMN_all=mMN_all;
    sim.vRC_all=vRC_all;
    sim.spike_times2=spike_times2;
    sim.Iexc_all=Iexc_all;
    sim.Iinh_all=Iinh_all;
end