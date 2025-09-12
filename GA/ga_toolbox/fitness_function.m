function fit = fitness_function(y_goal,h_output)
% compute the goodness of fit 1-(var(y-h_output)/var(y))
tmp=y_goal-h_output;
fit = var(tmp(:))/var(y_goal(:)); %1-R2
%fit = -1*sum(var(y_goal).*log(var(h_output)) + (1-var(y_goal)).*log(1-var(h_output)))/length(y_goal);
%fit = sqrt(sum((y_goal-h_output).^2))/length(y_goal);
%fit = corr2(y_goal,h_output);

end