% R-Squared (Coefficient of Determination)
function R2 = cal_R2(y,simMEP) 

    R2 = 1 - sumsqr(y(:)-simMEP(:))/sumsqr(y(:)-mean(y(:)));

end