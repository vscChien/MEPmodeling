function ref = config_model_pheno(subj)
    %----- MEP (target) -----
    switch subj
        case {1,8},     intensity_idx=1:10;  % TMS intensities
        case {2,3,4,9}, intensity_idx=1:7;   % TMS intensities
        case {5,7},     intensity_idx=1:8;   % TMS intensities
        case {6,10},    intensity_idx=1:6;   % TMS intensities
    end
    ref.tcrop=[20,50];    % ms, time window of interest
    [ref.y0, ref.t0, ~, ref.intensities, ~] = load_MEP(subj,intensity_idx,ref.tcrop,0);
    ref.y0 = ref.y0'; % [t x N]
    [ref.muaps, ref.tmuap] = load_muap();
    ref.subj               = subj;
    ref.intensity_idx      = intensity_idx;
    %======================================================================
    %-----search boundary-----
    % curveType: Gamma + tshift
    nParams = 3+length(intensity_idx);
    boundary = zeros(nParams,2); % [lower,upper]
    boundary(1,:) = [0.1,5];    % mu
    boundary(2,:) = [0.1,5];    % lambda
    boundary(3,:) = [0,2];      % tshift (msec), increased latency per decreased TMS intensity
    ref.boundarytext = {'shape','scale','tshift'};
    for i=1:length(intensity_idx)
        boundary(3+i,:) = [4,100];  % N1
        ref.boundarytext=[ref.boundarytext,sprintf('N%d',i)];
    end
    ref.boundary           = boundary;             
    
    ref.resultname = fullfile('fitted_results','pheno',...
                         sprintf('result_pheno_s%d.mat',subj)); 
    ref.figname=[ref.resultname(1:end-4) '.svg'];

end