% Generate N random solutions within the boundary
%
% input:
%    N: number of random solutions to generate
%    nParams: number of parameters
%    LR: lower boundary [nParams x 1]
%    UR: upper boundary [nParams x 1]
% output:
%    P: [N x nParams] random solutions
%
function P = population(N,nParams,LR,UR)

    P = zeros(N,nParams);
    for i = 1:nParams
        P(:,i) = (UR(i)-LR(i))*rand(N,1)+LR(i);
    end

end