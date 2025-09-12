function [Fit,Error,Houtput] = evaluation(X,func,y_goal)
% X:  population
%  
[x,~] = size(X); % d: number of dimensions (parameter numbers), x: realizations in population
Fit = zeros(1,x);
total_pop = num2str(x);

% here is objective function
reverseStr = '   ';
for j = 1:x
    %%%%%%% objective_function here
    P = X(j,:);         %parameter
    tic
    [error,houtput] = func(P,y_goal);  % input function to objective function must be 1xp
    %fit = fitness_function(y_goal.MUA(:),houtput.simMUA(:));
    fit = sumsqr(error);
    gettime = toc;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    msg = sprintf('simulation time: %3.5f --> %d/%s ',gettime,j,total_pop); 
    fprintf([reverseStr, msg]);
    reverseStr = repmat(sprintf('\b'), 1, length(msg));
    newline;
%     disp(['check running......' num2str(j)]);
    %%%%%%%%%%
    Fit(1,j) = fit;  
    Error(:,j) = error;
    Houtput{j} = houtput;
end


end
