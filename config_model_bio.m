function ref = config_model_bio(subj,withRC,AMPAweight)

    %----------------------------------
    model.AMPAweight    = AMPAweight;
    model.withRC        = withRC;  
    model.maxES         = 5;  % take first 5 spikes

    %-----subject MEP (target) -----
    switch subj
        case {1,8},     intensity_idx = 1:10;  % TMS intensities
        case {2,3,4,9}, intensity_idx = 1:7;   % TMS intensities
        case {5,7},     intensity_idx = 1:8;   % TMS intensities
        case {6,10},    intensity_idx = 1:6;   % TMS intensities
    end
    ref.subj               = subj;
    ref.intensity_idx      = intensity_idx;
    ref.tcrop=[20,50];     % ms, time window of interest
    [y0, t0, ~, intensities, ~] = load_MEP(subj,intensity_idx,ref.tcrop,0);
    ref.intensities=intensities;
    ref.t0 = t0;
    ref.y0 = y0';     % [t x N]
    [model.muaps, model.tmuap] = load_muap();


    %-----subject RMT ---------
    % (by eyeballing the IO curve)
    switch ref.subj
        case {1,8},  ref.RMT=32; % RMT
        case {2,10}, ref.RMT=35; 
        case {3,9},  ref.RMT=38;
        case 4,      ref.RMT=41;
        case {5,6},  ref.RMT=47;
        case 7,      ref.RMT=44;
    end

    %----- simulated DI wave -----
    tlength = 50;  % ms  
    dt      = 0.1; % ms
    t       = 0:dt:(tlength-dt);  
    model.DIwave0 = zeros(length(ref.intensities),length(t));
    for i=1:length(ref.intensities)
        model.DIwave0(i,:) = gen_DIwave(t,ref.intensities(i)/ref.RMT);
    end
    model.DIwave=deconv_DIwave(t,model.DIwave0,ref);

    %----- AMPA, NMDA kernels -----
    [model.AMPA,model.NMDA] = gen_kernels(dt,tlength);

    %----- AchR, GlyR kernels -----
     model.kernel.tau = [0.5, 3.6; % EPSPs (MN->RC): fast AchR (rise 0.5 ms, decay 3.6 ms)
                         1.8, 20.2;%                 slow AchR (rise 1.8 ms, decay 20.2 ms)
                         1,   6];  % IPSP (RC->MN):  GlyR (rise 1 ms, decay 6 ms) 
     model.kernel.h   = [1.5977; 
                         1.3908; 
                         1.7175]; % normalizing terms
    %----- Motor neuron -----
    model.tauLIF    = 10;   % ms
    model.Eexc      = 0;    % mV
    model.Einh      = -75;  % mV
    model.V_rest    = -65;  % mV
    model.V_thr     = -55;  % mV
    model.T_ref     = 2;    % ms, refractory period
    model.fastAChRweight = 0.5; % AchR (MN->RC)
    
    %----- tDIwave (input) -----
    model.DIwaveConv=model.DIwave;
    fr=1;
    for i=1:length(ref.intensities)
      g = conv(fr*model.DIwave(i,:),model.AMPA);
      model.DIwaveConv_AMPA(i,:)=g(1:length(t));
      g = conv(fr*model.DIwave(i,:),model.NMDA);
      model.DIwaveConv_NMDA(i,:)=g(1:length(t));
    end
    model.t=t;
    model.dt=dt;
   
    plot_DIwave(model,ref); 
 
    %----- search boundary-----
    switch withRC
        case 0
            nParams = 7;
            boundary = zeros(nParams,2); % [lower,upper]
            boundary(1,:) = [0,10];    % R1
            boundary(2,:) = [0,10];    % R2
            boundary(3,:) = [0,10];    % R3
            boundary(4,:) = [0,10];    % R4
            boundary(5,:) = [0,10];    % R5
            boundary(6,:) = [5,10];    % Tmuap
            boundary(7,:) = [0,1];     % AMPAweight
            if ~isempty(AMPAweight) 
                boundary(7,:) = [1,1]*AMPAweight; % fixed value
            end
            model.boundarytext = {'R1','R2','R3','R4','R5','MU.T','AMPAw'};
        case 1
            nParams = 12;
            boundary = zeros(nParams,2); % [lower,upper]
            boundary(1,:) = [0,10];     % R1
            boundary(2,:) = [0,10];     % R2
            boundary(3,:) = [0,10];     % R3
            boundary(4,:) = [0,10];     % R4
            boundary(5,:) = [0,10];     % R5
            boundary(6,:) = [0,20];     % E1 (MN1->RC)
            boundary(7,:) = [0,20];     % E2 (MN100->RC)
            boundary(8,:) = [0,10];     % I1 (RC->MN1)
            boundary(9,:) = [0,10];     % I2 (RC->MN100)
            boundary(10,:) = [1,10];    % RC.th
            boundary(11,:) = [5,10];    % Tmu
            boundary(12,:) = [0,1];     % AMPAweight
            if ~isempty(AMPAweight) 
                boundary(12,:) = [1,1]*AMPAweight; % fixed value
            end
            model.boundarytext = {'R1','R2','R3','R4','R5','E1','E2','I1','I2','RC.th','MU.T','AMPAw'};
    end

    model.boundary       = boundary;     
    ref.model = model;


    %----- fitted result -----
    if withRC        
        if isempty(AMPAweight)
            ref.resultname = fullfile('fitted_results','bio',...
                                      sprintf('result_bio_s%d.mat',subj));
        else
            ref.resultname = fullfile('fitted_results','bio','fixed_AMPAweight',...
                                      sprintf('result_bio_s%d[%g].mat',subj,AMPAweight));
        end
    else 
        ref.resultname = fullfile('fitted_results','bioNoRC',...
                                  sprintf('result_bioNoRC_s%d.mat',subj));

    end
    ref.figname=[ref.resultname(1:end-4) '.svg']; 


end
