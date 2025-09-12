function [fit_after_g,Para_E_after_g,error_after_g]=gauss_newton_slow(op,Para_E_test,r_test,func,y_goal,reg0,reg1,steps,loop,tol,LR,UR,fit_crit)
% Para_E_test: current parameter  [nx1]
% r_test:      current residual output
% func:        cost-function
% y_goal:      goal output for cost-function 
% reg0:        
% reg1: 
% loop:  num. of loops
% LR,UR, low and high boundary for graidient_repair
% fit_crit:  the loop stop crit.

%Para_E_after_g :  updated-parameter [nx1] 
j = 1;
 while j <= loop
     %fprintf('step %d :\n',j);
     J = NMM_diff_A_lfm(Para_E_test,r_test,func,y_goal);
     Para_E_new_group = multi_lavenberg_regulization(steps,reg0,reg1,Para_E_test,J,r_test,LR,UR);
%      if sum(isnan(Para_E_new_group(:)))
%          keyboard;
%      end
     fprintf('[%d/%d] ',j,loop);
     [fit_grp,error_grp] = evaluation(Para_E_new_group,func,y_goal);
     [Para_E_new, fit_new, error_new] = selection_best(Para_E_new_group,fit_grp,error_grp,1,op);
     r_test = error_new;
     Para_E_test = Para_E_new';
     
  
     disp(fit_new);
     fit_(j) = fit_new;
     Para_E_(j,:) = Para_E_test;
     error_(:,j) = r_test;
     j = j+1;
     
     % need check
%      if (op <0) && (fit_new < fit_crit-tol)
%          break
%      end
%      if (op>0) && (fit_new > fit_crit+tol)
%          break
%      end
%      if isinf(r_test)
%          break
%      end
     if length(fit_)>1 && op*fit_(end)-op*fit_(end-1)< tol
         fprintf('Quit: improvement < tol(%g)\n',tol)
         break
     end
     
 end
 

% Para_E_after_g = Para_E_test;
% r_after_g = r_test;
if j==loop
    [Para_E_after_g,fit_after_g,error_after_g] = selection_best(Para_E_,fit_,error_,1,op);
    Para_E_after_g = Para_E_after_g';
else
    Para_E_after_g = Para_E_test;
    fit_after_g = fit_new;
    error_after_g = r_test;
end