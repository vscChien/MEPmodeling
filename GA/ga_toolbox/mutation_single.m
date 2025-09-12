% single-parameter mutation on the best solution
% input 
%    solution : [1 x nParams], best solution with nParams parameters
%    LR: low boundary
%    UR: high boundary
% output
%    P: [nParams x nParams], nParams mutated solutions
%
function P = mutation_single(solution,LR,UR)

nParams = length(solution); 
P = repmat(solution,nParams,1);
P(eye(nParams)==1)= population(1,nParams,LR,UR);
    
end



