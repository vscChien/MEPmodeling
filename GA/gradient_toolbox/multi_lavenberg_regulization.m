function Y = multi_lavenberg_regulization(n,reg0,reg1,Para_E,J,h_output,LR,UR)
% n: return n populations
% Para_E: current parameter for update  [nx1]
% J jacobbian for update
% reg0, min.reg
% reg1, max. reg

% Y: n population x Parameter

Y = zeros(n,length(Para_E));  % return n population x p parameter

reg = 10.^linspace(reg0,reg1,n);

% if isinf(h_output(:)) || isnan(h_output(:))
%     Y = repmat(Para_E',n,1);
% else
    for i = 1: length(reg)
        % lavenberg-regulization
        try
        D = pinv(J'*J+reg(i)*eye(length(Para_E)));
        catch
            return
        end
        d = -D*J'*h_output;  % using in case when f(x)-y_goal
        if sum(isnan(d))
            keyboard;
        end
        Para_E_new = Para_E + d;
        % check if gradient over run the boundary
        Y(i,:) = gradient_repair(Para_E_new,LR,UR);
    end
% end

end