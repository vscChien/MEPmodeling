%---------3+i free parameters------------
% p(1): alpha, shape of gamma distribution
% p(2): lamda, rate of gamma distribution
% p(3): d, time lag between TMS conditions
% p(3+i): active MNs (range [4-100]) by TMS intensity i 

function [simMEP,ref] = MEPmodel_pheno(p,ref,plotOn)

    if nargin<3
        plotOn=0; 
    end

    %----- simulate MEP -----
    sim = MEPmodel_pheno_core(ref,p);
    ref = cal_error(ref,sim); 
    simMEP = ref.sim.simMEP2;

    if plotOn
        plot_param_panel_pheno(p,ref);  
        plot_summary_pheno(p,ref);
    end
end

%==========================================================================
function ref = cal_error(ref,sim)
    y0=ref.y0;
    t0=ref.t0;
    simMEP=sim.simMEP';
    %------------------
    simMEP2=simMEP/max(simMEP(:))*max(y0(:));
    error=simMEP2(:)-y0(:);
    NRMSD=norm(y0(:)-simMEP2(:))/sqrt(length(t0))/(max(y0(:))-min(y0(:)));
    R2=1-sumsqr(y0(:)-simMEP2(:))/sumsqr(y0(:)-mean(y0(:)));
    %-------------------
    ref.NRMSD=NRMSD; 
    ref.error=error;
    ref.R2=R2;
    sim.simMEP2=simMEP2;
    ref.sim=sim;
end

