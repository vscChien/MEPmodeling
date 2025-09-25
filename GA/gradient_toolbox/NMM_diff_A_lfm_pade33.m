function j = NMM_diff_A_lfm_pade33(para, houtput, myfunc,y_goal)
    % Estimate Jacobian using Padé [3/3] derivative for each parameter
    % Inputs:
    %   para     - 1×P parameter vector
    %   houtput  - T×1 or T×n output at para
    %   myfunc   - function handle: @(para) → output (same size as houtput)
    % Output:
    %   j        - Jacobian (T×P)

    [T, ~] = size(houtput);
    P = length(para);
    j = zeros(T, P);

    h = linspace(-0.02, 0.02, 7)';  % 7 points for [3/3] (m + n + 1 = 7)

    for p = 1:P
        % Sample perturbed parameter values
        h_vec = para(p) + h;
        Y = zeros(T, length(h_vec));

        for k = 1:length(h_vec)
            para_perturbed = para;
            para_perturbed(p) = h_vec(k);
            Y(:, k) = myfunc(para_perturbed,y_goal);
        end

        % Compute derivative for each output dimension
        df_p = zeros(T, 1);

        for t = 1:T
            y_vals = Y(t, :)';
            H = h;

            % Build design matrix for Padé [3/3]:
            % [1 h h^2 h^3 | -y*h -y*h^2 -y*h^3]
            A = [H.^0, H.^1, H.^2, H.^3, ...
                 -y_vals.*H, -y_vals.*H.^2, -y_vals.*H.^3];

            rhs = y_vals;

            %coeffs = A \ rhs;
            lambda = 1e-6; % Small regularization parameter
            coeffs = (A.' * A + lambda * eye(size(A,2))) \ (A.' * rhs);

            a = coeffs(1:4);         % a0, a1, a2, a3
            b = [1; coeffs(5:7)];    % b0 = 1, b1, b2, b3

            % f'(0) = (P'(0)*Q(0) - P(0)*Q'(0)) / Q(0)^2
            P0 = a(1);
            Q0 = b(1);

            Pp0 = sum((1:3)'.*a(2:4));   % a1 + 2*a2 + 3*a3
            Qp0 = sum((1:3)'.*b(2:4));   % b1 + 2*b2 + 3*b3

            df_p(t) = (Pp0 * Q0 - P0 * Qp0) / Q0^2;
        end

        j(:, p) = df_p;
    end

    j(isnan(j))=0; %
    j(isinf(j))=0; %
end