function [YY1, YY2] = selection_uniq(P1,B,p,r,op,LR,UR)
% P1: population
% B : fitness value
% p : population size
% s : select top s chromosomes

% YY1 : selected population
% YY2 : the trun fitness of YY1

%f = S(find(rand<cumsum(w),1,'first'))

% trun min. to max.
B = op*B;
dim = size(P1,2);

%first kill inf
index = isinf(B);
B(index) = [];
P1(index,:) = [];
index = isnan(B);
B(index) = [];
P1(index,:) = [];

% check firt the unique, keep unique
[P1,ip,~] = unique(P1,'rows');
B = B(ip);


% sort from high to low , get the best
[E,index] = sort(B,'descend'); % E all sorted cost
P1 = P1(index,:); % all sorted pop.

% recheck the length, min shold be p
len_counter = length(E); % total length
if len_counter < p
    new_len = p-len_counter;
    P1 = [P1;population(new_len,dim,LR,UR)];
    E = [E,nan(1,new_len)];
end
% select best
% randon draw the rest
if len_counter > p
    P1_best = P1(1:r,:);
    E_best = E(1:r);
    P2 = P1(r+1:end,:);
    E2 = E(r+1:end);
    index = randperm(length(E2));
    P2 =P2(index,:);
    E2 =E2(index);
    P2 = P2(1:p-r,:);
    E2 = E2(1:p-r);
    P1 = [P1_best;P2];
    E = [E_best,E2];
    YY1 = P1(1:p,:);
    YY2 = op*E(1:p);  % turn back
else
    YY1 = P1(1:p,:);
    YY2 = op*E(1:p);  % turn back
end


end


  
    
 