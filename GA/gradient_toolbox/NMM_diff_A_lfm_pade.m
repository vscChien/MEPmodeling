function j = NMM_diff_A_lfm_pade(para, houtput, myfunc,y_goal)
    % Estimate Jacobian of a black-box function using Padé [2/2] derivative
    % Inputs:
    %   para     - 1×P parameter vector
    %   houtput  - T×1 or T×n output at para (reference function value)
    %   myfunc   - function handle: @(para) → output (same size as houtput)
    % Output:
    %   j        - Jacobian (T×P), with each column df/dp_i via Padé

    [T, ~] = size(houtput);
    P = length(para);
    j = zeros(T, P);

    h = linspace(-0.02, 0.02, 5)';  % 5-point symmetric stencil for [2/2]

    for p = 1:P
        % Sample values around para(p)
        h_vec = para(p) + h;  % perturbed values
        Y = zeros(T, length(h_vec));

        for k = 1:length(h_vec)
            para_perturbed = para;
            para_perturbed(p) = h_vec(k);
            Y(:, k) = myfunc(para_perturbed,y_goal);
        end

        % Fit Padé [2/2] to each output dimension at once
        % Use same h for all outputs
        df_p = zeros(T, 1);  % one column of Jacobian

        for t = 1:T
            y_vals = Y(t, :)';  % y(h) for this output dim
            H = h;

            % Build design matrix: [1 h h^2 | -y*h -y*h^2]
            A = [H.^0, H.^1, H.^2, -y_vals.*H, -y_vals.*H.^2];
            rhs = y_vals;

            %coeffs = A \ rhs;
            lambda = 1e-6; % Small regularization parameter
            coeffs = (A.' * A + lambda * eye(size(A,2))) \ (A.' * rhs);


            a = coeffs(1:3);
            b = [1; coeffs(4:5)];

            % f'(0) = (P'(0)*Q(0) - P(0)*Q'(0)) / Q(0)^2
            P0 = a(1);
            Pp0 = a(2);
            Q0 = b(1);
            Qp0 = b(2);

            df_p(t) = (Pp0 * Q0 - P0 * Qp0) / Q0^2;
        end

        j(:, p) = df_p;
    end

    j(isnan(j))=0; %
    j(isinf(j))=0; %
end