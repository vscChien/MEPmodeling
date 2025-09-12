function [P_post]=run_gradient(P,target,boundary,conf)

    if nargin<4
        conf.op = -1;   % -1: find global minimum
                        %  1: find global maximum
        conf.y_goal = target;
        conf.myfunc = @objective_function;
        conf.LR = boundary(:,1);
        conf.UR = boundary(:,2);   
        conf.gLoop = 64;%5
        conf.gL = -12;
        conf.gU = 12;
        conf.gT = abs(conf.gU-conf.gL)+1;
        conf.gTol = 0.05;%0.01;
    end


    % get test ----------------
    %rng(6666)
    Para_E = P;

    % run object function
    [fit,error] = evaluation(Para_E,conf.myfunc,conf.y_goal);%para_E as group, should be n_pop x n_p
    disp(['fit: ' , num2str(fit)]); % fit is sumsquare (error)

    % gradient search
    disp('gradient search....')
    [fit_new,P_post,error_new]=gauss_newton_slow(conf.op,Para_E',error,conf.myfunc,conf.y_goal,conf.gL,conf.gU,conf.gT,conf.gLoop,conf.gTol,conf.LR,conf.UR,error);
    %disp(fit_new)

end