function [YY1, YY2, YY3] = selection_best(P,E,R,p,op)
% P: population            {n_pop x n_parameter}
% E : fitness value        {n_pop x n_parameter}
% R:  residual, y-houtput  {n_data sample x n_pop}
% p : population size      the number of pop want to be returened
% op: -1: select minimum, 1 select maximum


% YY1 : selected population                  
% YY2 : the retrun of fitness of YY1         
% YY3 : the return of risidual of YY1


% trun min. to max. if necessary
E = op*E;

% sort from high to low , get the best
[E,index] = sort(E,'descend');
P = P(index,:);
R = R(:,index);

YY1 = P(1:p,:);
YY2 = op*E(1:p);  % turn back
YY3 = R(:,1:p);




