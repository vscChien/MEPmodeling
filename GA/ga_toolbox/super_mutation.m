function P1 = super_mutation(P,n,t,LR,UR)
% P : population
% n : return n new population
% t : stand diviation for the zero-mean normal noise
% LR: low boundary
% UR: high boundary

% P1 new population
[~,dim] = size(P);  % get parameter space dimension
P1 = ones(n,dim).*P;

if length(LR)==1  % all parameter share the same boundary
    %P1 = P1 + rand(n,dim)*t;
    P1 = P1 +  normrnd(0,t,n,dim); % normal distribution zero mean, stand div t
    % check boundary
    P1(P1<LR) = LR;
    P1(P1>UR) = UR;
 
else     
    P1 = P1 +  normrnd(0,t,n,dim); % normal distribution zero mean, stand div t   
    for i = 1:dim
        P_ = P1(:,i);
        P_(P_<LR(i)) = LR(i);
        P_(P_>UR(i)) = UR(i);
        P1(:,i) = P_;
    end  
end



