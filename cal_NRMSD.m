% NRMSD (Normalized Root Mean Square Deviation)
function NRMSD = cal_NRMSD(y,simMEP)

    NRMSD = norm(y(:)-simMEP(:))/sqrt(length(y))/(max(y(:))-min(y(:)));

end