function state = kbmtl_semisupervised_classification_variational_train(K, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(K, 1);
    N = size(K, 2);
    T = size(Y, 2);
    R = parameters.R;
    sigma_h = parameters.sigma_h;
    sigma_w = parameters.sigma_w;

    Lambda.alpha = (parameters.alpha_lambda + 0.5) * ones(D, R);
    Lambda.beta = parameters.beta_lambda * ones(D, R);
    A.mu = randn(D, R);
    A.sigma = repmat(eye(D, D), [1, 1, R]);
    H.mu = randn(R, N);
    H.sigma = repmat(eye(R, R), [1, 1, N]);

    W.mu = randn(R, T);
    W.sigma = repmat(eye(R, R), [1, 1, T]);

    F.mu = (abs(randn(N, T)) + parameters.margin) .* sign(Y);
    F.sigma = ones(N, T);

    KKT = K * K';

    lower = -1e40 * ones(N, T);
    lower(Y > 0) = +parameters.margin;
    upper = +1e40 * ones(N, T);
    upper(Y < 0) = -parameters.margin;

    for iter = 1:parameters.iteration
        if mod(iter, 1) == 0
            fprintf(1, '.');
        end
        if mod(iter, 10) == 0
            fprintf(1, ' %5d\n', iter);
        end

        %%%% update Lambda
        for s = 1:R
            Lambda.beta(:, s) = 1 ./ (1 / parameters.beta_lambda + 0.5 * (A.mu(:, s).^2 + diag(A.sigma(:, :, s))));
        end
        %%%% update A
        for s = 1:R
            A.sigma(:, :, s) = (diag(Lambda.alpha(:, s) .* Lambda.beta(:, s)) + KKT / sigma_h^2) \ eye(D, D);
            A.mu(:, s) = A.sigma(:, :, s) * (K * H.mu(s, :)' / sigma_h^2);
        end
        %%%% update H
        for i = 1:N
            indices = ~isnan(Y(i, :));
            H.sigma(:, :, i) = (eye(R, R) / sigma_h^2 + W.mu(:, indices) * W.mu(:, indices)' + sum(W.sigma(:, :, indices), 3)) \ eye(R, R);
            H.mu(:, i) = H.sigma(:, :, i) * (A.mu' * K(:, i) / sigma_h^2 + W.mu(:, indices) * F.mu(i, indices)');
        end

        %%%% update W
        for t = 1:T
            indices = ~isnan(Y(:, t));
            W.sigma(:, :, t) = (eye(R, R) / sigma_w^2 + H.mu(:, indices) * H.mu(:, indices)' + sum(H.sigma(:, :, indices), 3)) \ eye(R, R);
            W.mu(:, t) = W.sigma(:, :, t) * (H.mu(:, indices) * F.mu(indices, t));
        end

        %%%% update F
        output = H.mu' * W.mu;
        alpha_norm = lower - output;
        beta_norm = upper - output;
        normalization = normcdf(beta_norm) - normcdf(alpha_norm);
        normalization(normalization == 0) = 1;
        F.mu = output + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;
        F.sigma = 1 + (alpha_norm .* normpdf(alpha_norm) - beta_norm .* normpdf(beta_norm)) ./ normalization - (normpdf(alpha_norm) - normpdf(beta_norm)).^2 ./ normalization.^2;
    end

    state.Lambda = Lambda;
    state.A = A;
    state.W = W;
    state.parameters = parameters;
end
