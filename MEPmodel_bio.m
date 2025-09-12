%---------12 free parameters (with RC population)------------
% p(1:5): membrane resistances R of MNs
% p(6:7): connectivity W_exc (MNs->RC)
% p(8:9): connectivity R.*W_inh (RC->MNs)
% p(10) : RC.th (threshold of RC sigmoid function)
% p(11) : Tmu (refractory period of muscle unit)
% p(12) : AMPA weight (for DI-wave input)
%
%---------7 free parameters (without RC population)--------
% p(1:5): membrane resistances R of MNs
% p(6)  : Tmu (refractory period of muscle unit)
% p(7)  : AMPA weight (for DI-wave input)

function [simMEP,ref] = MEPmodel_bio(p,ref,plotOn)

    if nargin<3
        plotOn=0;
    end
   
    ref = update_model_bio(ref,p);
    sim = MEPmodel_bio_core(ref.model); % simulation
    ref = cal_error(ref,sim);
    simMEP = ref.sim.simMEP2;
    
    if plotOn        

        % show fitted parameters in the search boundary
        plot_param_panel(p,ref); 

        % summarize goodness-of-fit of MEPs across TMS intensities
        plot_summary(p,ref); 

        % select a TMS intensity to show MN-RC interaction
        idx=length(ref.intensities); %1; 
        plot_TMScondition(ref,idx); 
    end
end
%==========================================================================
function ref = update_model_bio(ref,p)
    switch ref.model.withRC
        case 0
            R     = p(1:5);
            E1    = 0; 
            E2    = 0; 
            I1    = 0; 
            I2    = 0;
            RCth  = 10;
            Tmu   = p(6);
            AMPAw = p(7);
        case 1
            R     = p(1:5);
            E1    = p(6); 
            E2    = p(7); 
            I1    = p(8); 
            I2    = p(9);
            RCth  = p(10);
            Tmu   = p(11);
            AMPAw = p(12);
    end
    %----- membrane resistances of MNs -----
    ref.model.R = gen_resistance_mono(R);
    %----- Wexc (MNs -> RC) -----
    ref.model.Wexc = linspace(E1,E2,100);   
    %----- R*Winh (RC->MNs)-----
    ref.model.RWinh = linspace(I1,I2,100); 
    %----- RC sigmoid ------
    ref.model.rc = RC_setting(RCth);
    %----- MUAP-----
    ref.model.Tmu       = Tmu;  % ms, refractory period 
    %----- DI-waves -----
    ref.model.AMPAweight= AMPAw;
    ref.model.DIwaveConv= ref.model.DIwaveConv_AMPA* ref.model.AMPAweight + ...
                          ref.model.DIwaveConv_NMDA* (1 - ref.model.AMPAweight);

end
%==========================================================================
% Reshaw cell population params
function rc=RC_setting(p)
   
    % RC sigmoid function
    rc.r     = 10;
    rc.v_thr = p;
    rc.fmax  = 1;

    % synaptic kernel (RC->MN, Glycine receptor)
    taur=1; % ms rise
    tauf=6; % ms fall
    rc.tau1=taur*tauf/(taur+tauf); 
    rc.tau2=tauf;
    rc.h=(rc.tau2-rc.tau1)/rc.tau1/rc.tau2; 

end

%==========================================================================
function ref = cal_error(ref,sim)

    y0=ref.y0;
    t0=ref.t0;
    simMEP=sim.simMEP'; % [500 x 10]
    t=sim.t;
    %---------------------
    % align peaks of simMEP and MEP
    [~,i1]=max(y0(:,end)); % largest TMS intensity
    [~,i2]=max(simMEP(:,end));
    axonalDelay=max(0,t0(i1)-t(i2)); % >=0 ms 
    simMEP = interp1(t + axonalDelay,simMEP,t0,'linear',0); % shift d
    %------------------------------------
    if max(simMEP(:))>0 % avoid devide by zero
        simMEP2=simMEP/max(simMEP(:))*max(y0(:));
    else
        simMEP2=simMEP;
    end
    error=simMEP2(:)-y0(:);
    NRMSD=norm(y0(:)-simMEP2(:))/sqrt(length(t0))/(max(y0(:))-min(y0(:)));
    R2=1-sumsqr(y0(:)-simMEP2(:))/sumsqr(y0(:)-mean(y0(:)));
    %---------------------
    ref.NRMSD=NRMSD;
    sim.simMEP2=simMEP2;
    ref.model.axonalDelay=axonalDelay;
    ref.R2=R2;
    ref.error=error;
    ref.sim=sim;
end
