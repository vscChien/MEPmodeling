function mutateP = mutationV(P,lowchance,highchance,LR,UR)
%mutation all position by chance based on cost
% P: ranked population , with the best cost at the fisrt row

mutateP = P;  %[n_pop x n_parameter]
mutateChance = linspace(lowchance,highchance,size(mutateP,1));
mask = rand(size(mutateP)) < repmat(mutateChance',1,size(mutateP,2));
tmp = population(size(mutateP,1),size(mutateP,2),LR,UR);
mutateP(mask)=tmp(mask);


end