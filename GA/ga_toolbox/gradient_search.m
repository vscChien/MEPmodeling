% run gradient search on the solutions
% input
%   P: [N x nParams] solutions
%   r: [timepoints x N] residual of solutions
%   stop_crit:   gradient search stop crit.
% output
%   P_post: [N x nParams] solutions after gradient search
%   fit_post: fit [=sumsqr(error)] after gradient search
%   r_post: [timepoints x N] residual of solutions after gradient search

function [P_post,fit_post,r_post] = gradient_search(P,r,conf,stop_crit)

[N,nParams] = size(P);
fit_post = zeros(1,N);
P_post = zeros(N,nParams);
for i = 1:N
   fprintf('G%d:\n',i);
   [fit_post(i),P_post(i,:),r_post(:,i)] = gauss_newton_slow(conf.op,P(i,:)',r(:,i),conf.myfunc,conf.y_goal,conf.gL,conf.gU,conf.gT,conf.gLoop,conf.gTol,conf.LR,conf.UR,stop_crit);
end

end

